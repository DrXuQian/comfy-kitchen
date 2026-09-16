#!/usr/bin/env python3
"""Compile and *link* the explicit PPU source graph, without PyTorch C++ ABI.

No device is needed. This is not a numerical or performance PASS.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def build(output, build_dir, sdk=None, extended=False, jobs=1):
    if jobs < 1:
        raise ValueError("jobs must be positive")
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
        f"-I{ROOT / 'comfy_kitchen/backends/ppu'}",
    ]
    source_paths = [ROOT / "comfy_kitchen/backends/ppu" / source for source in SOURCES]
    generated, config_census = [], None
    if extended:
        from ppu_int8_config_space import generate

        table, generated, config_census = generate(build_dir / "generated")
        flags += [
            f'-DCOMFY_PPU_INT8_CONFIGS="{table}"',
            "-DCOMFY_PPU_INT8_SHARDED=1",
            f"-I{table.parent}",
        ]
        source_paths += generated
        print(
            f"[PPU sweep build] raw={config_census['raw_count']} "
            f"accepted={len(config_census['accepted'])} rejected={len(config_census['rejected'])} "
            f"shards={len(generated)} dtypes=3 legacy_ids=0..5/UNCHANGED",
            flush=True,
        )
    commands, objects = [], []
    for src in source_paths:
        obj = build_dir / (src.stem + ".o")
        command = flags + ["-c", str(src), "-o", str(obj)]
        commands.append(command)
        objects.append(str(obj))

    def compile_one(command):
        src = Path(command[-3])
        log = build_dir / (src.stem + ".compile.log")
        print(f"[PPU build] compiling {src.name}; log={log}", flush=True)
        with log.open("w") as stream:
            stream.write(shlex.join(command) + "\n")
            stream.flush()
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"compile failed: {src}\n{log.read_text(errors='replace')[-12000:]}")
        return src.name

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(compile_one, command) for command in commands]
        for count, future in enumerate(as_completed(futures), 1):
            print(f"[PPU build] compiled {count}/{len(futures)} {future.result()}", flush=True)
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
        "generated_sources": [str(p) for p in generated],
        "config_census": config_census,
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
            "int8_dispatch.hpp",
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
    if extended:
        manifest["generated_sha256"] = {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (build_dir / "generated").iterdir()
            if p.is_file()
        }
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[PPU build] COMPILED+LINKED: {output}; device correctness/performance NOT RUN")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk")
    parser.add_argument("--build-dir", default="/workspace/comfy-kitchen-ppu-build")
    parser.add_argument("--output", default=str(ROOT / "comfy_kitchen/backends/ppu/_native.so"))
    parser.add_argument(
        "--extended", action="store_true", help="Build the explicit expanded sweep table"
    )
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    build(args.output, args.build_dir, args.sdk, args.extended, args.jobs)
