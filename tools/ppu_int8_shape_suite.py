#!/usr/bin/env python3
"""Sweep controlled large-M shapes using one existing, hash-bound PPU binary.

These are heuristic probes, not an asserted MiniMax model-layer inventory.
No kernel compilation, concurrent GPU sweeps, or automatic selector changes.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from ppu_int8_config_space import validate_census

ROOT = Path(__file__).resolve().parents[1]
ROLES = ("prequantized-core", "quant+core")
DEFAULT_SHAPES = (
    (4096, 4096, 4096),
    (16384, 4096, 4096),
    (73774, 4096, 4096),
    (73774, 8192, 4096),
    (73774, 4096, 8192),
    (73774, 8192, 8192),
    (73774, 16384, 4096),
    (73774, 4096, 16384),
)
# Names, not sweep-local IDs: expansion can renumber non-legacy configs.
CONTROL_NAMES = (
    "128x128x64_w64x64_s3",  # Product fallback for M >= 128 (config1).
    "128x256x128_w64x64_s2",  # Original six-row winner (config5).
    "64x256x128_w64x32_s2",  # Measured 4096-cube winner (currently config94).
)


def minimax_h3_shapes(tokens):
    # Official FL2VA config: hidden=5376, heads*head_dim=56*128,
    # ffn=14336. ComfyUI fuses Q/K/V and gate/up, respectively.
    hidden, attention, ffn = 5376, 56 * 128, 14336
    return (
        (tokens, 3 * attention, hidden),
        (tokens, hidden, attention),
        (tokens, 2 * ffn, hidden),
        (tokens, hidden, ffn),
    )


def shape_arg(text):
    try:
        values = tuple(int(v) for v in text.lower().replace("x", ",").split(","))
        if len(values) != 3 or min(values) <= 0:
            raise ValueError
        if values[2] % 256 or values[2] > 131040:
            raise ValueError
        return values
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "shape must be positive M,N,K; this ConvRot suite requires K % 256 == 0 and K <= 131040"
        ) from error


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_name(row):
    return f"{row['tm']}x{row['tn']}x{row['tk']}_w{row['wm']}x{row['wn']}_s{row['stages']}"


def bind_library(library):
    path = Path(library).resolve()
    if not path.is_file():
        raise ValueError(f"library not found: {path}; reuse the 285-config sweep's _native.so")
    manifest_path = path.with_suffix(".so.json")
    if not manifest_path.is_file():
        raise ValueError(f"missing binary manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    digest = sha256(path)
    if manifest.get("binary_sha256") != digest:
        raise ValueError("binary SHA256 differs from its build manifest")
    census = manifest.get("config_census")
    if not census:
        raise ValueError("this suite requires the expanded config binary, not the six-row build")
    validate_census(census)
    names = [config_name(row) for row in census["accepted"]]
    if any(name not in names for name in CONTROL_NAMES):
        raise ValueError("binary census lacks a required comparison config")
    return {
        "path": str(path),
        "sha256": digest,
        "source_sha": manifest["sha"],
        "manifest_sha256": sha256(manifest_path),
        "config_census": census,
    }


def validate_result(data, shape, binding, protocol):
    """Reject partial sweeps, mixed binaries, and silently changed denominators."""
    if data.get("complete") is not True:
        raise ValueError("shape sweep is incomplete")
    meta = data["metadata"]
    if meta["binary_sha256"] != binding["sha256"]:
        raise ValueError("shape sweep used a different binary")
    args = meta["args"]
    if tuple(args[key] for key in ("m", "n", "k")) != tuple(shape):
        raise ValueError("shape identity changed")
    if args["dtype"] != "bf16" or args["convrot"] is not True:
        raise ValueError("dtype/ConvRot contract changed")
    for key, value in protocol.items():
        if args[key] != value:
            raise ValueError(f"timing protocol changed: {key}")
    census = binding["config_census"]
    if meta["config_census"] != census:
        raise ValueError("shape config census changed")
    names = [config_name(row) for row in census["accepted"]]
    if meta["candidate_count"] != len(names):
        raise ValueError("candidate denominator changed")
    skips = data["skips"]
    if any(not row.get("reason") for row in skips):
        raise ValueError("unreasoned resource SKIP")
    for row in data["results"] + skips + data["confirmations"]:
        if row["config"] not in range(len(names)) or row["name"] != names[row["config"]]:
            raise ValueError("config coordinate/ID mismatch")
    for role in ROLES:
        measured = [row for row in data["results"] if row["role"] == role]
        ids = [row["config"] for row in measured + skips]
        if sorted(ids) != list(range(len(names))):
            raise ValueError(f"candidate denominator incomplete or duplicated: {role}")
        confirmed = [row for row in data["confirmations"] if row["role"] == role]
        if len({row["config"] for row in confirmed}) != len(confirmed):
            raise ValueError("duplicate confirmation")
        required = set(CONTROL_NAMES) - {row["name"] for row in skips}
        if not required.issubset({row["name"] for row in confirmed}):
            raise ValueError("control missing from fresh confirmation")
        for row in confirmed:
            if len(row["samples_us"]) != protocol["confirm_samples"]:
                raise ValueError("confirmation sample denominator changed")
        winner = data["winners"][role]
        if winner["binding"] != "all-screened+top-confirmed":
            raise ValueError("winner did not receive fresh confirmation")
        if winner["verdict"] not in ("RESOLVED", "UNRESOLVED"):
            raise ValueError("unrecognized winner verdict")
    return meta


def write_summary(outdir, summary):
    # Complete raw per-shape results are embedded: one upload preserves every
    # sample, exclusion, coordinate, and device/binary identity for analysis.
    (outdir / "shape-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# PPU INT8 shape sweep",
        "",
        f"Completed {len(summary['runs'])}/{len(summary['shapes'])}; "
        f"complete={summary['complete']}. No selector was changed.",
        "",
        f"Scope: {summary['scope']}. BF16, ConvRot, warm/reused allocations; "
        "public-API aggregate events. Activation functions are not timed.",
        "",
        "| M,N,K | Role | Best | us | TOPS | % of declared INT8 peak | "
        "Runner-up gap (us) | Verdict | config1 / best | cube winner / best |",
        "|---|---|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for run in summary["runs"]:
        shape = ",".join(map(str, run["shape"]))
        data = run["data"]
        for role in ROLES:
            winner = data["winners"][role]
            controls = {
                r["name"]: r["median_us"] for r in data["confirmations"] if r["role"] == role
            }
            ratios = [
                f"{controls[name] / winner['median_us']:.4f}x"
                if name in controls
                else "resource-SKIP"
                for name in (CONTROL_NAMES[0], CONTROL_NAMES[2])
            ]
            lines.append(
                f"| {shape} | {role} | {winner['best']} | {winner['median_us']:.3f} | "
                f"{winner['logical_tops']:.3f} | {winner['utilization_pct']:.2f} | "
                f"{winner['gap_us']:.3f} | {winner['verdict']} | {' | '.join(ratios)} |"
            )
    lines += [
        "",
        "Ratios are descriptive medians, not proof of a resolved pairwise win. "
        "UNRESOLVED remains UNRESOLVED; use the raw envelopes in shape-summary.json.",
    ]
    (outdir / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", default=os.environ.get("COMFY_KITCHEN_PPU_LIBRARY"))
    parser.add_argument("--shapes", nargs="+", type=shape_arg)
    parser.add_argument("--suite", choices=("probe", "minimax-h3"), default="probe")
    parser.add_argument(
        "--tokens", type=int, default=73774, help="Packed tokens for --suite minimax-h3"
    )
    parser.add_argument(
        "--bias", choices=("vector", "none"), help="Default: none for H3, vector for probes"
    )
    parser.add_argument("--samples", type=int, default=3, help="Full-table screening samples")
    parser.add_argument("--iterations", type=int, default=3, help="Launches per screening sample")
    parser.add_argument("--confirm-samples", type=int, default=7)
    parser.add_argument("--confirm-iterations", type=int, default=20)
    parser.add_argument("--confirm-top", type=int, default=8)
    parser.add_argument("--peak-tops", type=float, default=1000)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan without importing torch or loading a GPU library",
    )
    args = parser.parse_args()
    if args.shapes is not None and args.suite != "probe":
        parser.error("custom --shapes and --suite minimax-h3 are mutually exclusive")
    if args.tokens <= 0:
        parser.error("--tokens must be positive")
    scope = "synthetic-large-M-heuristic-probes-not-model-inventory"
    if args.suite == "minimax-h3":
        args.shapes = minimax_h3_shapes(args.tokens)
        scope = "MiniMax-H3-ComfyUI-fused-QKV-out-FFN-up-down; activation-functions-excluded"
    elif args.shapes is None:
        args.shapes = DEFAULT_SHAPES
    if args.bias is None:
        args.bias = "none" if args.suite == "minimax-h3" else "vector"
    if len(set(args.shapes)) != len(args.shapes):
        parser.error("duplicate shapes")
    if (
        min(
            args.samples,
            args.iterations,
            args.confirm_samples,
            args.confirm_iterations,
            args.confirm_top,
            args.peak_tops,
        )
        <= 0
    ):
        parser.error("sample/iteration/confirmation counts and peak must be positive")
    print(
        f"[PPU shape suite] {scope}; BF16 + ConvRot bias={args.bias}; no selector change",
        flush=True,
    )
    for i, (m, n, k) in enumerate(args.shapes, 1):
        print(
            f"[PPU shape plan] {i}/{len(args.shapes)} M={m} N={n} K={k} "
            f"output_MiB={m * n * 2 / 2**20:.3f} logical_Top={2 * m * n * k / 1e12:.6f}",
            flush=True,
        )
    print(
        f"[PPU shape protocol] screen={args.samples}x{args.iterations} "
        f"confirm={args.confirm_samples}x{args.confirm_iterations} top={args.confirm_top} "
        f"plus_controls={','.join(CONTROL_NAMES)} sequential=1 compile=0",
        flush=True,
    )
    if args.dry_run:
        return 0
    if not args.library:
        parser.error("--library or COMFY_KITCHEN_PPU_LIBRARY is required; no implicit rebuild")
    binding = bind_library(args.library)
    outdir = args.out or Path("/workspace") / (
        "comfy-kitchen-ppu-shapes-"
        + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        + f"-{os.getpid()}"
    )
    outdir.mkdir(parents=True, exist_ok=True)
    protocol = {
        key: getattr(args, key)
        for key in (
            "samples",
            "iterations",
            "confirm_samples",
            "confirm_iterations",
            "confirm_top",
            "peak_tops",
            "bias",
        )
    }
    summary = {
        "schema": 1,
        "scope": scope,
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ),
        "binding": binding,
        "protocol": protocol,
        "shapes": args.shapes,
        "controls": CONTROL_NAMES,
        "runs": [],
        "complete": False,
    }
    # Exclusive creation also prevents a second runner from clobbering this run.
    with (outdir / "plan.json").open("x") as stream:
        stream.write(json.dumps(summary, indent=2) + "\n")
    env = dict(os.environ, COMFY_KITCHEN_PPU_LIBRARY=binding["path"])
    device_identity = None
    write_summary(outdir, summary)
    try:
        for index, shape in enumerate(args.shapes, 1):
            m, n, k = shape
            directory = outdir / f"{index:02d}-{m}x{n}x{k}"
            directory.mkdir()
            cmd = [
                sys.executable,
                str(ROOT / "tools/ppu_int8_sweep.py"),
                "--m",
                str(m),
                "--n",
                str(n),
                "--k",
                str(k),
                "--dtype",
                "bf16",
                "--convrot",
                "--out",
                str(directory),
            ]
            for key, value in protocol.items():
                cmd += ["--" + key.replace("_", "-"), str(value)]
            for name in CONTROL_NAMES:
                cmd += ["--confirm-config", name]
            print(f"[PPU shape suite] RUN {index}/{len(args.shapes)} {m}x{n}x{k}", flush=True)
            with (directory / "sweep.log").open("w") as log:
                log.write("command=" + json.dumps(cmd) + "\n")
                log.flush()
                with subprocess.Popen(
                    cmd,
                    cwd=ROOT,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                ) as process:
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end="", flush=True)
                    if process.wait() != 0:
                        raise RuntimeError(f"shape {shape} failed; see {directory / 'sweep.log'}")
            paths = list(directory.glob("sweep-*.json"))
            if len(paths) != 1:
                raise ValueError("expected exactly one raw result for this shape")
            data = json.loads(paths[0].read_text())
            meta = validate_result(data, shape, binding, protocol)
            identity = {
                key: meta[key] for key in ("device", "device_properties", "ppu_versions", "torch")
            }
            if device_identity is not None and identity != device_identity:
                raise ValueError("device/runtime identity changed between shapes")
            device_identity = identity
            summary["runs"].append(
                {"shape": shape, "result_sha256": sha256(paths[0]), "data": data}
            )
            write_summary(outdir, summary)
    except Exception as error:
        summary["failure"] = str(error)
        write_summary(outdir, summary)
        raise
    summary["complete"] = True
    write_summary(outdir, summary)
    print(
        f"[PPU shape suite] PASS {len(summary['runs'])}/{len(args.shapes)}; "
        f"upload={outdir / 'shape-summary.json'} summary={outdir / 'summary.md'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
