#!/usr/bin/env python3
"""Sequential PPU config sweep. Core and quant+core are separate measurements."""

import argparse
import hashlib
import json
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
        "--peak-tops", type=float, help="Optional calibrated INT8 peak (not fp16 TFLOPS)"
    )
    p.add_argument("--out", default="/workspace/comfy-kitchen-ppu-sweep")
    args = p.parse_args()
    if min(args.m, args.n, args.k, args.samples, args.iterations) <= 0:
        p.error("extents/samples/iterations must be positive")
    if args.warmup < 0 or (args.peak_tops is not None and args.peak_tops <= 0):
        p.error("warmup must be nonnegative and peak TOPS positive")
    if not ppu.runtime.is_ppu_device("cuda:0"):
        raise RuntimeError("PPU required; no performance result")
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
    names = ppu.configurations()
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
    results = []
    anchor = ppu.int8_gemm(q, w, xs, ws, bias, dtype, config=0)
    for config in order:
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
                **ppu.resources(config, dtype),
            }
            if args.peak_tops:
                row["utilization_pct"] = 100 * tops / args.peak_tops
            results.append(row)
            print("[PPU INT8 sweep] " + json.dumps(row), flush=True)
            result_path.write_text(
                json.dumps({"metadata": metadata, "results": results, "complete": False}, indent=2)
                + "\n"
            )
    winners = {}
    for role in ("prequantized-core", "quant+core"):
        ranked = sorted((r for r in results if r["role"] == role), key=lambda r: r["median_us"])
        a, b = ranked[:2]
        # Conservative envelope test, fixed before data: overlapping envelopes
        # cannot establish a unique winner even if medians differ.
        resolved = a["max_us"] < b["min_us"]
        winners[role] = {
            "best": a["name"],
            "runner_up": b["name"],
            "gap_us": b["median_us"] - a["median_us"],
            "verdict": "RESOLVED" if resolved else "UNRESOLVED",
        }
        print("[PPU INT8 winner] " + role + " " + json.dumps(winners[role]), flush=True)
    result_path.write_text(
        json.dumps(
            {"metadata": metadata, "results": results, "complete": True, "winners": winners},
            indent=2,
        )
        + "\n"
    )
    print(f"artifacts: {result_path}")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
