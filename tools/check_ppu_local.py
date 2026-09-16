#!/usr/bin/env python3
"""Host proof with actual shipping types and deliberate negative controls."""

import argparse
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def run(sdk, output, extended=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    compiler = Path(sdk) / "bin/hgcc"
    if not compiler.is_file():
        print(f"SKIP: actual-type host proof requires PPU compiler: {compiler}")
        return 77
    obj, binary = output / "ownership.o", output / "ownership"
    extra = []
    if extended:
        from ppu_int8_config_space import generate

        table, _, _ = generate(output / "generated")
        extra.append(f'-DCOMFY_PPU_INT8_CONFIGS="{table.resolve()}"')
    command = [
        str(compiler),
        "-arch=ppu_10",
        "-x",
        "cu",
        "-std=c++17",
        "-O1",
        "--expt-relaxed-constexpr",
        "-DUSE_CLANG",
        "-DCUTLASS_USE_PACKED_TUPLE=1",
        "-DCUTE_USE_PACKED_TUPLE=1",
        f"-I{ROOT}",
        f"-I{ROOT / 'third_party/actlize/include'}",
        f"-I{ROOT / 'comfy_kitchen/backends/ppu'}",
        f"-I{Path(sdk) / 'include'}",
        *extra,
        "-c",
        str(ROOT / "tests/ppu_ownership.cu"),
        "-o",
        str(obj),
    ]
    subprocess.run(command, check=True)
    symbols = subprocess.check_output(["nm", "-u", str(obj)], text=True)
    allowed = {"__hggcRegisterFatBinary", "__hggcUnregisterFatBinary", "__hggcRegisterVar"}
    for line in symbols.splitlines():
        name = line.split()[-1]
        if ("hggc" in name or "cuda" in name) and name not in allowed:
            raise RuntimeError(f"host proof accidentally depends on device execution: {name}")
    subprocess.run(
        ["g++", str(obj), str(ROOT / "tests/ppu_host_registration.cpp"), "-o", str(binary)],
        check=True,
    )
    subprocess.run([str(binary)], check=True)
    for plant in ("--plant-owner", "--plant-scale"):
        result = subprocess.run([str(binary), plant], capture_output=True, text=True)
        if result.returncode != 1 or "[PPU host] FAIL:" not in result.stderr:
            raise RuntimeError(f"negative control did not fail normally: {plant}: {result}")
        print(f"[PPU host negative] {plant}: {result.stderr.strip()} EXPECTED-RED/PASS")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sdk", default=os.environ.get("PPU_SDK", "/usr/local/PPU_SDK"))
    p.add_argument("--out", default="/workspace/comfy-kitchen-ppu-local")
    p.add_argument("--extended", action="store_true")
    args = p.parse_args()
    raise SystemExit(run(args.sdk, args.out, args.extended))
