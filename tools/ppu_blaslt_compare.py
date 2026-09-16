#!/usr/bin/env python3
"""Same-input PPU actlize vs acBLASLt INT8 comparison; quantization is untimed."""

import argparse
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from comfy_kitchen.backends import ppu
from ppu_int8_check import assert_bits, oracle
from ppu_int8_sweep import measure


class Unsupported(RuntimeError):
    """Only the native bridge's explicit vendor NOT_SUPPORTED maps here."""


def check_status(rc, error):
    if rc == 2:
        raise Unsupported(error())
    if rc:
        raise RuntimeError(error())


def compare_envelopes(ours, vendor):
    verdict = "UNRESOLVED"
    if ours["max_us"] < vendor["min_us"]:
        verdict = "ACTLIZE-WINS"
    elif vendor["max_us"] < ours["min_us"]:
        verdict = "ACBLASLT-WINS"
    return {
        "verdict": verdict,
        "vendor_over_ours": vendor["median_us"] / ours["median_us"],
        "vendor_minus_ours_us": vendor["median_us"] - ours["median_us"],
    }


def load_bridge(path):
    lib = C.CDLL(str(Path(path).resolve()), mode=os.RTLD_NOW | os.RTLD_LOCAL | os.RTLD_DEEPBIND)
    lib.comfy_lt_error.restype = C.c_char_p
    lib.comfy_lt_version.restype = C.c_size_t
    lib.comfy_lt_library_path.restype = C.c_char_p
    lib.comfy_lt_create.argtypes = [
        C.c_int64,
        C.c_int64,
        C.c_int64,
        C.c_size_t,
        C.c_int,
        C.POINTER(C.c_void_p),
    ]
    lib.comfy_lt_destroy.argtypes = [C.c_void_p]
    lib.comfy_lt_destroy.restype = None
    lib.comfy_lt_count.argtypes = [C.c_void_p]
    lib.comfy_lt_info.argtypes = [C.c_void_p, C.c_int]
    lib.comfy_lt_info.restype = C.c_char_p
    lib.comfy_lt_run.argtypes = [
        C.c_void_p,
        C.c_int,
        C.POINTER(ppu.runtime.GemmArgs),
        C.c_void_p,
        C.c_void_p,
        C.c_void_p,
        C.c_int,
    ]
    return lib


class Trial:
    def __init__(self, lib, q, w, xs, ws, bias, dtype, workspace, heuristics):
        self.lib, self.workspace = lib, workspace
        self.q, self.w, self.xs, self.ws, self.bias = q, w, xs, ws, bias
        m, k = q.shape
        n = w.shape[0]
        self.acc = torch.empty((m, n), dtype=torch.int32, device=q.device)
        self.output = torch.empty((m, n), dtype=dtype, device=q.device)
        self.args = ppu.runtime.GemmArgs(
            m,
            n,
            k,
            q.data_ptr(),
            w.data_ptr(),
            xs.data_ptr(),
            ws.data_ptr(),
            None if bias is None else bias.data_ptr(),
            self.output.data_ptr(),
            ppu.runtime.DTYPES[dtype],
            int(ws.numel() == 1),
            0,
        )
        self.stream = torch.cuda.current_stream().cuda_stream
        self.plan = C.c_void_p()
        self.check(lib.comfy_lt_create(m, n, k, workspace.numel(), heuristics, C.byref(self.plan)))
        self.candidates = [
            json.loads(lib.comfy_lt_info(self.plan, i))
            for i in range(lib.comfy_lt_count(self.plan))
        ]

    def check(self, rc):
        check_status(rc, lambda: self.lib.comfy_lt_error().decode(errors="replace"))

    def close(self):
        if self.plan:
            torch.cuda.synchronize()
            self.lib.comfy_lt_destroy(self.plan)
            self.plan = C.c_void_p()

    def lt(self, slot, mode=1):
        self.check(
            self.lib.comfy_lt_run(
                self.plan,
                slot,
                C.byref(self.args),
                self.acc.data_ptr(),
                self.workspace.data_ptr(),
                self.stream,
                mode,
            )
        )
        return self.acc if mode == 0 else self.output

    def ours(self, config):
        self.args.config = config
        ppu.runtime.check(
            ppu.runtime.library().comfy_ppu_int8_gemm(C.byref(self.args), self.stream)
        )
        return self.output


def independent_admission(lib, workspace, dtype, heuristics):
    # Non-square dimensions are essential: the target 4096^3 alone could hide
    # an incorrectly transposed descriptor. Entire result uses CPU int64 truth.
    q = torch.randint(-17, 18, (64, 96), dtype=torch.int8, device="cuda")
    w = torch.randint(-13, 14, (128, 96), dtype=torch.int8, device="cuda")
    xs = (torch.arange(64, device="cuda") % 7 + 1).float() / 32
    ws = (torch.arange(128, device="cuda") % 11 + 1).float() / 64
    bias = ((torch.arange(128, device="cuda") % 7 - 3).float() / 16).to(dtype)
    want_acc = q.cpu().to(torch.int64) @ w.cpu().to(torch.int64).T
    want = oracle(q, w, xs, ws, bias, dtype)
    trial = Trial(lib, q, w, xs, ws, bias, dtype, workspace, heuristics)
    admitted = 0
    try:
        for candidate in trial.candidates:
            try:
                trial.acc.fill_(0x55555555)
                trial.output.fill_(float("nan"))
                got = trial.lt(candidate["slot"])
                assert_bits(trial.acc, want_acc.to(torch.int32))
                assert_bits(got, want)
                admitted += 1
            except Unsupported as e:
                print(f"[PPU Lt admission] slot={candidate['slot']} SKIP reason={e}", flush=True)
        if not admitted:
            raise Unsupported("no acBLASLt algorithm passed the non-square INT8 admission")
        # Detection power: both descriptor/index errors and missing scale must
        # be visible in this exact same fixture, not in a different easy case.
        wrong_order = want_acc.T.contiguous().reshape_as(want_acc).to(torch.int32)
        for name, bad, good in (
            ("transposed-output", wrong_order, want_acc.to(torch.int32)),
            ("missing-row-scale", oracle(q, w, torch.ones_like(xs), ws, bias, dtype), want),
        ):
            try:
                assert_bits(bad, good)
            except AssertionError:
                print(f"[PPU Lt negative] {name} EXPECTED-RED/PASS", flush=True)
            else:
                raise AssertionError(f"negative control escaped: {name}")
        print(
            f"[PPU Lt admission] 64x128x96 all-output INT32+epilogue RAW-BIT/PASS candidates={admitted}",
            flush=True,
        )
    finally:
        trial.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library", required=True)
    p.add_argument("--out", required=True)
    for name, default in (
        ("m", 4096),
        ("n", 4096),
        ("k", 4096),
        ("samples", 7),
        ("iterations", 20),
        ("warmup", 5),
        ("heuristics", 32),
        ("workspace-mib", 64),
    ):
        p.add_argument("--" + name, type=int, default=default)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--convrot", action="store_true")
    p.add_argument("--peak-tops", type=float, default=1000.0)
    args = p.parse_args()
    if min(args.m, args.n, args.k, args.samples, args.iterations, args.heuristics) <= 0:
        p.error("extents and sample counts must be positive")
    if (
        args.warmup < 0
        or args.workspace_mib < 0
        or not math.isfinite(args.peak_tops)
        or args.peak_tops <= 0
        or args.heuristics > 256
    ):
        p.error("invalid warmup/workspace/peak/heuristics")
    if args.k % 32 or args.k > 131040 or (args.convrot and args.k % 256):
        p.error("K must satisfy the shipping INT8/ConvRot contract")
    if not ppu.runtime.is_ppu_device("cuda:0"):
        raise RuntimeError("PPU required: local NVIDIA hardware is NOT this baseline")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / (time.strftime("blaslt-%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{os.getpid()}.json")
    lib = load_bridge(args.library)
    vendor_path = lib.comfy_lt_library_path()
    if not vendor_path:
        raise RuntimeError("cannot bind the actually loaded acBLASLt binary")
    vendor_path = Path(os.fsdecode(vendor_path)).resolve()
    metadata = {
        "args": vars(args),
        "device": torch.cuda.get_device_name(0),
        "device_properties": str(torch.cuda.get_device_properties(0)),
        "torch": torch.__version__,
        "ppu_versions": ppu.runtime.versions(),
        "acblaslt_version": lib.comfy_lt_version(),
        "sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ),
        "binary_sha256": {
            str(f): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in (ppu.runtime.library_path(), Path(args.library).resolve(), vendor_path)
        },
        "cache": "warm/reused allocations",
        "allocation": "preallocated for BOTH sides",
        "timing": "aggregate current-stream device events; includes host launch idle",
        "layout": "zero-copy TN: Yt[N,M]_col = Wt[K,N]_col^T * At[K,M]_col",
        "quantization": "identical shared inputs; ALL quantization excluded",
        "claim_scope": "heuristic candidates + library default; NOT exhaustive vendor autotuning",
        "peak_scope": "user-specified dense INT8 TOPS; not measured ACU Compute SOL",
    }
    results, skips = [], []
    document = {"metadata": metadata, "results": results, "skips": skips, "complete": False}

    def save():
        path.write_text(json.dumps(document, indent=2) + "\n")

    def record(role, identity, stats):
        tops = 2 * args.m * args.n * args.k / stats["median_us"] / 1e6
        row = {
            "role": role,
            "identity": identity,
            **stats,
            "logical_tops": tops,
            "utilization_pct": tops / args.peak_tops * 100,
        }
        results.append(row)
        print("[PPU Lt compare] " + json.dumps(row), flush=True)
        save()
        return row

    save()
    torch.manual_seed(901)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    workspace = torch.empty(args.workspace_mib * 1024**2, dtype=torch.uint8, device="cuda")
    try:
        independent_admission(lib, workspace, dtype, args.heuristics)
        # Reset RNG after admission: same generator as the existing INT8 sweep.
        torch.manual_seed(901)
        x = torch.randint(-8, 9, (args.m, args.k), device="cuda").to(dtype) / 8
        w = torch.randint(-8, 9, (args.n, args.k), dtype=torch.int8, device="cuda")
        ws = (torch.arange(args.n, device="cuda") % 11 + 1).float() / 64
        bias = ((torch.arange(args.n, device="cuda") % 7 - 3).float() / 16).to(dtype)
        q, xs = ppu._quantize(x, convrot=args.convrot)
        anchor = ppu.int8_gemm(q, w, xs, ws, bias, dtype, config=0)
        rows = torch.linspace(0, args.m - 1, min(4, args.m)).long()
        cols = torch.linspace(0, args.n - 1, min(17, args.n)).long()
        qr, wc = q[rows.to(q.device)], w[cols.to(w.device)]
        want_acc = qr.cpu().to(torch.int64) @ wc.cpu().to(torch.int64).T
        assert_bits(
            anchor[rows.to(q.device)][:, cols.to(w.device)],
            oracle(
                qr, wc, xs[rows.to(q.device)], ws[cols.to(w.device)], bias[cols.to(w.device)], dtype
            ),
        )
        trial = Trial(lib, q, w, xs, ws, bias, dtype, workspace, args.heuristics)
        metadata["vendor_candidates"] = trial.candidates
        metadata["output_sha256"] = hashlib.sha256(
            anchor.cpu().view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        metadata["input_sha256"] = {
            name: hashlib.sha256(t.cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
            for name, t in (("q", q), ("w", w), ("xs", xs), ("ws", ws), ("bias", bias))
        }
        names = ppu.configurations()
        work = [("ours", i) for i in range(len(names))] + [
            ("lt", r["slot"]) for r in trial.candidates
        ]
        random.Random(901).shuffle(work)
        metadata["screening_order"] = work
        print("[PPU Lt compare metadata] " + json.dumps(metadata), flush=True)
        try:
            for backend, index in work:
                trial.output.fill_(float("nan"))
                if backend == "ours":
                    assert_bits(trial.ours(index), anchor)
                    stats, last = measure(
                        lambda: trial.ours(index), args.warmup, args.samples, args.iterations
                    )
                    assert_bits(last, anchor)
                    record("actlize-fused", {"config": index, "name": names[index]}, stats)
                    continue
                try:
                    trial.acc.fill_(0x55555555)
                    assert_bits(trial.lt(index), anchor)
                    assert_bits(
                        trial.acc[rows.to(q.device)][:, cols.to(w.device)], want_acc.to(torch.int32)
                    )
                except Unsupported as e:
                    row = {"backend": "acblaslt", "slot": index, "reason": str(e)}
                    skips.append(row)
                    print("[PPU Lt SKIP] " + json.dumps(row), flush=True)
                    save()
                    continue
                for role, mode in (
                    ("acblaslt-int32-only-DIAGNOSTIC", 0),
                    ("acblaslt-scale-bias", 1),
                ):
                    stats, last = measure(
                        lambda: trial.lt(index, mode), args.warmup, args.samples, args.iterations
                    )
                    if mode == 0:
                        assert_bits(
                            last[rows.to(q.device)][:, cols.to(w.device)], want_acc.to(torch.int32)
                        )
                        assert_bits(trial.lt(index, 2), anchor)
                    else:
                        assert_bits(last, anchor)
                    record(role, trial.candidates[index], stats)
            ours = min(
                (r for r in results if r["role"] == "actlize-fused"), key=lambda r: r["median_us"]
            )
            vendors = [r for r in results if r["role"] == "acblaslt-scale-bias"]
            if not vendors:
                raise Unsupported("no acBLASLt INT8 algorithm supported the target shape")
            vendor = min(vendors, key=lambda r: r["median_us"])
            cfg, slot = ours["identity"]["config"], vendor["identity"]["slot"]
            # Fresh interleaved confirmation: don't compare an early measurement
            # with a later one while clocks/temperature may have drifted.
            arms = {
                "actlize-fused": lambda: trial.ours(cfg),
                "acblaslt-scale-bias": lambda: trial.lt(slot),
            }
            samples = {role: [] for role in arms}
            rng = random.Random(1901)
            for _ in range(args.samples):
                order = list(arms)
                rng.shuffle(order)
                for role in order:
                    stats, last = measure(arms[role], args.warmup, 1, args.iterations)
                    assert_bits(last, anchor)
                    samples[role].extend(stats["samples_us"])
            confirmed = {}
            for role, values in samples.items():
                identity = ours["identity"] if role == "actlize-fused" else vendor["identity"]
                confirmed[role] = record(
                    "confirmed-" + role,
                    identity,
                    {
                        "median_us": statistics.median(values),
                        "min_us": min(values),
                        "max_us": max(values),
                        "samples_us": values,
                    },
                )
            document["comparison"] = compare_envelopes(
                confirmed["actlize-fused"], confirmed["acblaslt-scale-bias"]
            )
            document["comparison"][
                "scope"
            ] = "same BF16/FP16/FP32 output including both scales and bias; quantization excluded"
            stats, last = measure(
                lambda: trial.lt(slot, 2), args.warmup, args.samples, args.iterations
            )
            assert_bits(last, anchor)
            # No GEMM TOPS/MFU for a pure elementwise kernel.
            document["epilogue_only_diagnostic"] = stats
            # Public-API control binds the new prepared-call protocol to the old
            # sweep without silently comparing unlike host/allocation overhead.
            stats, last = measure(
                lambda: ppu.int8_gemm(q, w, xs, ws, bias, dtype, config=cfg),
                args.warmup,
                args.samples,
                args.iterations,
            )
            assert_bits(last, anchor)
            record("actlize-public-api-DIAGNOSTIC", ours["identity"], stats)
            document.update(complete=True, status="PASS")
            print("[PPU Lt verdict] " + json.dumps(document["comparison"]), flush=True)
        finally:
            trial.close()
    except Unsupported as e:
        document.update(status="SKIP", reason=str(e))
        print(f"[PPU Lt compare] SKIP: {e}; no speed claim", flush=True)
        return 2
    except Exception as e:
        document.update(status="FAIL", reason=str(e))
        raise
    finally:
        save()
        print(f"artifacts: {path}", flush=True)
    return 0


if __name__ == "__main__":
    with torch.inference_mode():
        raise SystemExit(main())
