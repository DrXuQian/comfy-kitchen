#!/usr/bin/env python3
"""Sequential PPU config sweep. Core and quant+core are separate measurements."""

import argparse
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from ppu_int8_admission import device_limits, resource_reason
from ppu_int8_check import assert_bits, oracle

from comfy_kitchen.backends import ppu


def measure(fn, warmup, samples, iterations):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(samples):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            result = fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / iterations)
    return {
        "median_us": statistics.median(times),
        "min_us": min(times),
        "max_us": max(times),
        "samples_us": times,
    }, result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--m", type=int, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--convrot", action="store_true")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--samples", type=int, default=7)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument(
        "--confirm-top",
        type=int,
        default=0,
        help="Fresh interleaved confirmation of top N plus legacy config5, per role",
    )
    p.add_argument("--confirm-samples", type=int, help="Default: same as --samples")
    p.add_argument("--confirm-iterations", type=int, help="Default: same as --iterations")
    p.add_argument(
        "--confirm-config",
        action="append",
        default=[],
        metavar="CONFIG_NAME",
        help="Also confirm this coordinate name, even outside the screened top N",
    )
    p.add_argument(
        "--peak-tops", type=float, help="Optional calibrated INT8 peak (not fp16 TFLOPS)"
    )
    p.add_argument("--out", default="/workspace/comfy-kitchen-ppu-sweep")
    args = p.parse_args()
    if args.confirm_samples is None:
        args.confirm_samples = args.samples
    if args.confirm_iterations is None:
        args.confirm_iterations = args.iterations
    if (
        min(
            args.m,
            args.n,
            args.k,
            args.samples,
            args.iterations,
            args.confirm_samples,
            args.confirm_iterations,
        )
        <= 0
    ):
        p.error("extents/samples/iterations must be positive")
    if args.confirm_config and not args.confirm_top:
        p.error("--confirm-config requires --confirm-top")
    if (
        args.warmup < 0
        or args.confirm_top < 0
        or (args.peak_tops is not None and args.peak_tops <= 0)
    ):
        p.error("warmup must be nonnegative and peak TOPS positive")
    if not ppu.runtime.is_ppu_device("cuda:0"):
        raise RuntimeError("PPU required; no performance result")
    names = ppu.configurations()
    for name in args.confirm_config:
        if name not in names:
            raise RuntimeError(f"confirmation config absent from loaded binary: {name}")
    controls = {5, *(names.index(name) for name in args.confirm_config)}
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    # Never overwrite a prior run's measurements.
    result_path = outdir / (
        "sweep-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{os.getpid()}.json"
    )
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    torch.manual_seed(901)
    x = torch.randint(-8, 9, (args.m, args.k), device="cuda").to(dtype) / 8
    w = torch.randint(-8, 9, (args.n, args.k), device="cuda", dtype=torch.int8)
    ws = (torch.arange(args.n, device="cuda") % 11 + 1).float() / 64
    bias = ((torch.arange(args.n, device="cuda") % 7 - 3).float() / 16).to(dtype)
    q, xs = ppu._quantize(x, convrot=args.convrot)
    rows = torch.linspace(0, args.m - 1, min(4, args.m)).long()
    cols = torch.linspace(0, args.n - 1, min(17, args.n)).long()
    expected = oracle(
        q[rows.to(q.device)],
        w[cols.to(w.device)],
        xs[rows.to(xs.device)],
        ws[cols.to(ws.device)],
        bias[cols.to(bias.device)],
        dtype,
    )
    order = list(range(len(names)))
    random.Random(901).shuffle(order)
    metadata = {
        "sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ),
        "binary_sha256": hashlib.sha256(ppu.runtime.library_path().read_bytes()).hexdigest(),
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "device_properties": str(torch.cuda.get_device_properties(0)),
        "ppu_versions": ppu.runtime.versions(),
        "torch_cuda": torch.version.cuda,
        "args": vars(args),
        "order": order,
        "cache": "warm/reused allocations",
        "timing": "aggregate device events; includes launch idle",
        "correctness": "sampled independent int64 + all-output raw equality vs config0 (run smoke first)",
    }
    results, skips, confirmations = [], [], []
    limits = device_limits(ppu.runtime)
    metadata["device_limits"] = limits
    metadata["candidate_count"] = len(names)
    manifest_path = ppu.runtime.library_path().with_suffix(".so.json")
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest["binary_sha256"] != metadata["binary_sha256"]:
            raise RuntimeError("build manifest does not bind the loaded binary")
        metadata["config_census"] = manifest.get("config_census")

    def save(complete=False, winners=None):
        result_path.write_text(
            json.dumps(
                {
                    "metadata": metadata,
                    "results": results,
                    "skips": skips,
                    "confirmations": confirmations,
                    "complete": complete,
                    "winners": winners or {},
                },
                indent=2,
            )
            + "\n"
        )

    save()
    anchor = ppu.int8_gemm(q, w, xs, ws, bias, dtype, config=0)
    for config in order:
        resources = ppu.resources(config, dtype)
        reason = resource_reason(resources, limits)
        if reason:
            skipped = {"config": config, "name": names[config], "reason": reason, **resources}
            skips.append(skipped)
            print("[PPU INT8 SKIP] " + json.dumps(skipped), flush=True)
            save()
            continue
        os.environ["COMFY_KITCHEN_PPU_INT8_CONFIG"] = str(config)
        core = lambda: ppu.int8_gemm(q, w, xs, ws, bias, dtype, config=config)
        e2e = lambda: ppu.int8_linear(x, w, ws, bias, dtype, convrot=args.convrot)
        check = core()
        assert_bits(check[rows.to(check.device)][:, cols.to(check.device)], expected)
        assert_bits(check, anchor)
        for role, fn in (("prequantized-core", core), ("quant+core", e2e)):
            stats, last = measure(fn, args.warmup, args.samples, args.iterations)
            assert_bits(last, anchor)
            tops = 2 * args.m * args.n * args.k / stats["median_us"] / 1e6
            row = {
                "config": config,
                "name": names[config],
                "role": role,
                **stats,
                "logical_tops": tops,
                **resources,
            }
            if args.peak_tops:
                row["utilization_pct"] = 100 * tops / args.peak_tops
            results.append(row)
            print("[PPU INT8 sweep] " + json.dumps(row), flush=True)
            save()
    if len(results) // 2 + len(skips) != len(names):
        raise RuntimeError("candidate denominator is incomplete")
    if len(results) < 4:
        raise RuntimeError("fewer than two admissible candidates; no winner comparison")
    if args.confirm_top:
        rng = random.Random(1901)
        for role in ("prequantized-core", "quant+core"):
            ranked = sorted((r for r in results if r["role"] == role), key=lambda r: r["median_us"])
            selected = ranked[: args.confirm_top]
            for config in sorted(controls):
                control = next((r for r in ranked if r["config"] == config), None)
                if control is not None and control not in selected:
                    selected.append(control)
            if len(selected) < 2:
                raise RuntimeError("confirmation needs at least two configs")
            timings = {r["config"]: [] for r in selected}
            for round_id in range(args.confirm_samples):
                configs = list(timings)
                rng.shuffle(configs)
                for config in configs:
                    os.environ["COMFY_KITCHEN_PPU_INT8_CONFIG"] = str(config)
                    fn = (
                        (lambda: ppu.int8_gemm(q, w, xs, ws, bias, dtype, config=config))
                        if role == "prequantized-core"
                        else (lambda: ppu.int8_linear(x, w, ws, bias, dtype, convrot=args.convrot))
                    )
                    stats, last = measure(fn, args.warmup, 1, args.confirm_iterations)
                    assert_bits(last, anchor)
                    timings[config].extend(stats["samples_us"])
                print(
                    f"[PPU INT8 confirm] role={role} round={round_id + 1}/{args.confirm_samples} candidates={len(configs)}",
                    flush=True,
                )
            for original in selected:
                values = timings[original["config"]]
                median = statistics.median(values)
                row = {
                    **original,
                    "median_us": median,
                    "min_us": min(values),
                    "max_us": max(values),
                    "samples_us": values,
                    "logical_tops": 2 * args.m * args.n * args.k / median / 1e6,
                }
                if args.peak_tops:
                    row["utilization_pct"] = 100 * row["logical_tops"] / args.peak_tops
                confirmations.append(row)
                print("[PPU INT8 confirmed] " + json.dumps(row), flush=True)
                save()
    winners = {}
    for role in ("prequantized-core", "quant+core"):
        authority = confirmations if args.confirm_top else results
        ranked = sorted((r for r in authority if r["role"] == role), key=lambda r: r["median_us"])
        a, b = ranked[:2]
        # Conservative envelope test, fixed before data: overlapping envelopes
        # cannot establish a unique winner even if medians differ.
        confirmed_ids = {r["config"] for r in ranked}
        outside = [r for r in results if r["role"] == role and r["config"] not in confirmed_ids]
        competitors = ranked[1:] + outside
        resolved = all(a["max_us"] < r["min_us"] for r in competitors)
        winners[role] = {
            "best_config": a["config"],
            "median_us": a["median_us"],
            "logical_tops": a["logical_tops"],
            "best": a["name"],
            "runner_up": b["name"],
            "gap_us": b["median_us"] - a["median_us"],
            "verdict": "RESOLVED" if resolved else "UNRESOLVED",
            "binding": "all-screened+top-confirmed" if args.confirm_top else "screening-only",
        }
        if args.peak_tops:
            winners[role]["utilization_pct"] = a["utilization_pct"]
        control = next((r for r in ranked if r["config"] == 5), None)
        if control:
            winners[role]["legacy_config5_median_us"] = control["median_us"]
            winners[role]["speedup_vs_legacy5"] = control["median_us"] / a["median_us"]
        print("[PPU INT8 winner] " + role + " " + json.dumps(winners[role]), flush=True)
    save(complete=True, winners=winners)
    print(
        f"[PPU INT8 denominator] candidates={len(names)} measured={len(results) // 2} resource_SKIP={len(skips)} FAIL=0",
        flush=True,
    )
    print(f"artifacts: {result_path}")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
