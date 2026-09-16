#!/usr/bin/env python3
"""Compile and *link* the explicit PPU source graph, without PyTorch C++ ABI.

No device is needed. This is not a numerical or performance PASS.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("int8_gemm.cu", "quantize.cu")


def link_command(cxx, objects, sdk, output):
    # HGGC's native runtime exports every symbol used by these objects. The
    # 2.1.1 wrapper still dlopens libhggcrt.12.0.so, although the SDK's own
    # libhggcrt1.so resolves to SONAME 13.0. Never involve that dispatch shim.
    return [
        cxx,
        "-shared",
        "-Wl,-z,defs",
        *objects,
        f"-Wl,-rpath,{sdk / 'lib'}",
        str(sdk / "lib/libhggcrt1.so"),
        str(sdk / "lib/libhggc.so"),
        "-ldl",
        "-o",
        str(output),
    ]


def check_native_linkage(output):
    dynamic = subprocess.check_output(["readelf", "-d", str(output)], text=True)
    needed = re.findall(r"\(NEEDED\).*?\[(.*?)\]", dynamic)
    if any("wrapper" in name for name in needed):
        raise RuntimeError(f"PPU extension must not depend on a runtime wrapper: {needed}")
    if not any(name.startswith("libhggcrt") for name in needed):
        raise RuntimeError(f"PPU extension has no direct native runtime dependency: {needed}")
    return needed


def git_output(arguments, directory):
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=directory, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except subprocess.CalledProcessError:
        return "UNAVAILABLE(source archive)"


def build(output, build_dir, sdk=None):
    sdk = Path(sdk or os.environ.get("PPU_SDK", "/usr/local/PPU_SDK")).resolve()
    compiler = sdk / "bin/hgcc"
    if not compiler.is_file():
        raise RuntimeError(f"PPU SDK compiler not found: {compiler}; set PPU_SDK")
    actlize = ROOT / "third_party/actlize/include"
    if not (actlize / "ppu_include.hpp").is_file():
        raise RuntimeError("run git submodule update --init third_party/actlize")
    build_dir, output = Path(build_dir).resolve(), Path(output).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    flags = [
        str(compiler),
        "-arch=ppu_10",
        "-x",
        "cu",
        "-std=c++17",
        "-O3",
        "-Xcompiler",
        "-fPIC",
        "--expt-relaxed-constexpr",
        "-DUSE_CLANG",
        "-DCUTLASS_USE_PACKED_TUPLE=1",
        "-DCUTE_USE_PACKED_TUPLE=1",
        "-DUSE_PPU=1",
        "-DUSE_AIU=1",
        "-fmad=false",
        f"-I{actlize}",
        f"-I{sdk / 'include'}",
        f"-I{ROOT / 'comfy_kitchen/backends/cuda'}",
    ]
    commands, objects = [], []
    for source in SOURCES:
        src = ROOT / "comfy_kitchen/backends/ppu" / source
        obj = build_dir / (src.stem + ".o")
        command = flags + ["-c", str(src), "-o", str(obj)]
        commands.append(command)
        print(shlex.join(command), flush=True)
        subprocess.run(command, check=True)
        objects.append(str(obj))
    # A real multi-TU link: unresolved symbols are errors, not deferred to box.
    cxx = shutil.which(os.environ.get("CXX", "g++"))
    if not cxx:
        raise RuntimeError("host C++ linker not found")
    command = link_command(cxx, objects, sdk, output)
    commands.append(command)
    print(shlex.join(command), flush=True)
    subprocess.run(command, check=True)
    needed = check_native_linkage(output)
    manifest = {
        "schema": 1,
        "scope": "COMPILE_AND_LINK_ONLY_NO_DEVICE_VERDICT",
        "sources": list(SOURCES),
        "commands": commands,
        "compiler": subprocess.check_output([compiler, "--version"], text=True),
        "sha": git_output(["rev-parse", "HEAD"], ROOT),
        "worktree_status": git_output(["status", "--porcelain"], ROOT),
        "actlize_sha": git_output(["rev-parse", "HEAD"], actlize.parent),
        "binary_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "runtime_linkage": {"mode": "direct-native-HGGC", "elf_needed": needed},
    }
    inputs = [
        ROOT / "comfy_kitchen/backends/ppu" / name
        for name in (
            *SOURCES,
            "api.h",
            "int8_config.hpp",
            "int8_configs.inc",
            "int8_epilogue.hpp",
            "runtime_compat.cuh",
        )
    ]
    inputs += [
        ROOT / "comfy_kitchen/backends/cuda" / name
        for name in ("ops/int8_linear.cu", "utils.cuh", "dtype_dispatch.cuh", "input_act_codes.h")
    ]
    manifest["source_sha256"] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in inputs
    }
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[PPU build] COMPILED+LINKED: {output}; device correctness/performance NOT RUN")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk")
    parser.add_argument("--build-dir", default="/workspace/comfy-kitchen-ppu-build")
    parser.add_argument("--output", default=str(ROOT / "comfy_kitchen/backends/ppu/_native.so"))
    args = parser.parse_args()
    build(args.output, args.build_dir, args.sdk)
