#!/usr/bin/env python3
"""Compile/link the benchmark-only Lt bridge; never changes the shipping library."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

from build_ppu import ROOT, check_native_linkage, git_output, link_command


def build(sdk, out):
    sdk, out = Path(sdk).resolve(), Path(out).resolve()
    compiler, blas = sdk / "bin/hgcc", sdk / "lib/libacblasLt.so"
    for path in (compiler, blas, sdk / "include/acblasLt.h"):
        if not path.is_file():
            raise RuntimeError(f"required PPU SDK component missing: {path}")
    out.mkdir(parents=True, exist_ok=True)
    source = ROOT / "tools/ppu_blaslt_probe.cu"
    obj, library = out / "blaslt.o", out / "_blaslt_probe.so"
    cxx = shutil.which(os.environ.get("CXX", "g++"))
    if not cxx:
        raise RuntimeError("host C++ linker missing")
    commands = [
        [
            str(compiler),
            "-arch=ppu_10",
            "-x",
            "cu",
            "-std=c++17",
            "-O3",
            "-Xcompiler",
            "-fPIC",
            "-fmad=false",
            f"-I{sdk / 'include'}",
            "-c",
            str(source),
            "-o",
            str(obj),
        ],
        link_command(cxx, [str(obj), str(blas)], sdk, library),
    ]
    for command in commands:
        print(shlex.join(command), flush=True)
        subprocess.run(command, check=True)
    needed = check_native_linkage(library)
    if "libacblasLt.so" not in needed:
        raise RuntimeError(f"benchmark is not linked to acBLASLt: {needed}")
    manifest = {
        "scope": "COMPILED_AND_LINKED_ONLY_DEVICE_NOT_RUN",
        "commands": commands,
        "sha": git_output(["rev-parse", "HEAD"], ROOT),
        "git_status": git_output(["status", "--porcelain"], ROOT),
        "compiler": subprocess.check_output([compiler, "--version"], text=True),
        "binary_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "acblaslt_sha256": hashlib.sha256(blas.read_bytes()).hexdigest(),
        "sources": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (source, ROOT / "comfy_kitchen/backends/ppu/api.h")
        },
        "elf_needed": needed,
    }
    library.with_suffix(".so.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[PPU Lt build] COMPILED+LINKED: {library}; device NOT RUN")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", default=os.environ.get("PPU_SDK", "/usr/local/PPU_SDK"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    build(args.sdk, args.out)
