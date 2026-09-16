# SPDX-License-Identifier: Apache-2.0
"""Versioned C ABI; no nanobind, ATen, or PyTorch C++ ABI dependency."""

import ctypes as C
from functools import lru_cache
import os
from pathlib import Path

import torch

DTYPES = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}


class GemmArgs(C.Structure):
    _fields_ = (
        [(key, C.c_int64) for key in ("m", "n", "k")]
        + [(key, C.c_void_p) for key in ("a", "b", "x_scale", "w_scale", "bias", "output")]
        + [(key, C.c_int32) for key in ("dtype", "scalar_weight_scale", "config")]
    )


def library_path():
    explicit = os.environ.get("COMFY_KITCHEN_PPU_LIBRARY")
    if explicit:
        path = Path(explicit).resolve()
        if not path.is_file():
            raise RuntimeError(f"COMFY_KITCHEN_PPU_LIBRARY not found: {path}")
        return path
    paths = sorted(Path(__file__).parent.glob("_native*.so"))
    if len(paths) != 1:
        raise RuntimeError(
            f"expected one PPU library, found {len(paths)}; build with COMFY_KITCHEN_BUILD_PPU=1"
        )
    return paths[0]


def is_ppu_device(device):
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    # The PPU torch runtime exposes CUDA-compatible tensors. CUDA capability
    # numbers alone are NOT a backend identity test (nor is SDK presence).
    return "PPU" in torch.cuda.get_device_name(device).upper()


@lru_cache(maxsize=1)
def library():
    # A globally loaded SDK wrapper (e.g. via torch) can interpose even our
    # versioned HGGC references. This library exchanges only pointers/scalars
    # through a C ABI, so bind its private SDK dependency before global shims.
    # No Python/ATen/C++ objects cross this boundary.
    if not hasattr(os, "RTLD_DEEPBIND"):
        raise RuntimeError("PPU C-ABI loader requires Linux RTLD_DEEPBIND")
    lib = C.CDLL(str(library_path()), mode=os.RTLD_NOW | os.RTLD_LOCAL | os.RTLD_DEEPBIND)
    lib.comfy_ppu_abi_version.restype = C.c_int
    if lib.comfy_ppu_abi_version() != 1:
        raise RuntimeError("unsupported comfy PPU native ABI")
    lib.comfy_ppu_last_error.restype = C.c_char_p
    lib.comfy_ppu_int8_gemm.argtypes = [C.POINTER(GemmArgs), C.c_void_p]
    lib.comfy_ppu_quantize_int8.argtypes = [
        C.c_void_p,
        C.c_void_p,
        C.c_void_p,
        C.c_int64,
        C.c_int64,
        C.c_int,
        C.c_int,
        C.c_int,
        C.c_int,
        C.c_uint64,
        C.c_void_p,
    ]
    lib.comfy_ppu_int8_config_name.argtypes = [C.c_int]
    lib.comfy_ppu_int8_config_name.restype = C.c_char_p
    lib.comfy_ppu_int8_resources.argtypes = [
        C.c_int,
        C.c_int,
        C.POINTER(C.c_int),
        C.POINTER(C.c_int),
        C.POINTER(C.c_int),
    ]
    return lib


def check(rc):
    if rc:
        raise RuntimeError(library().comfy_ppu_last_error().decode("utf-8", errors="replace"))


def configurations():
    lib = library()
    return [
        lib.comfy_ppu_int8_config_name(i).decode() for i in range(lib.comfy_ppu_int8_config_count())
    ]


def resources(config, dtype=torch.bfloat16):
    threads, smem, occ = C.c_int(), C.c_int(), C.c_int()
    check(
        library().comfy_ppu_int8_resources(
            config, DTYPES[dtype], C.byref(threads), C.byref(smem), C.byref(occ)
        )
    )
    return {
        "threads": threads.value,
        "shared_bytes": smem.value,
        "maximum_active_blocks": occ.value,
    }


def versions():
    result = {}
    for name, symbol in (("driver", "hggcDriverGetVersion"), ("runtime", "hggcRuntimeGetVersion")):
        try:
            fn = getattr(library(), symbol)
            fn.argtypes = [C.POINTER(C.c_int)]
            value = C.c_int()
            rc = fn(C.byref(value))
            result[name] = value.value if rc == 0 else f"UNAVAILABLE(rc={rc})"
        except AttributeError:
            result[name] = "UNAVAILABLE(symbol not exported by SDK)"
    return result
