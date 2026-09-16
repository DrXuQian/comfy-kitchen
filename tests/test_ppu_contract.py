# SPDX-License-Identifier: Apache-2.0
"""CPU-only contracts; no mocked arithmetic is called device validation."""

import ctypes
from pathlib import Path
import runpy

import pytest
import torch

from comfy_kitchen.backends import ppu
from comfy_kitchen.backends.ppu import runtime


def operands():
    return (
        torch.zeros(3, 96, dtype=torch.int8),
        torch.zeros(7, 96, dtype=torch.int8),
        torch.arange(3, dtype=torch.float32),
        torch.arange(7, dtype=torch.float32),
    )


def test_contract_accepts_tail():
    ppu._gemm_contract(*operands(), torch.zeros(7), torch.bfloat16)


@pytest.mark.parametrize(
    "index,replacement",
    [
        (0, torch.zeros(3, 97, dtype=torch.int8)),
        (1, torch.zeros(7, 96, dtype=torch.float16)),
        (2, torch.ones(1)),
        (3, torch.ones(6)),
    ],
)
def test_rejects_wrong_units_or_denominator(index, replacement):
    args = list(operands())
    args[index] = replacement
    with pytest.raises(ValueError):
        ppu._gemm_contract(*args, None, torch.float16)


def test_cpu_never_routes_to_ppu():
    assert not ppu._admit({"x": torch.ones(1, 32)}).success
    with pytest.raises(ValueError, match="requires a PPU"):
        ppu.int8_gemm(*operands())


def test_nvidia_device_is_not_ppu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda dev: "NVIDIA GeForce RTX 5090")
    assert not runtime.is_ppu_device("cuda:0")
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda dev: "PPU-ZW810")
    assert runtime.is_ppu_device("cuda:1")


@pytest.mark.parametrize("k,group", [(128, 256), (256, 64), (131072, 256)])
def test_convrot_rejects_unsupported_contract(k, group):
    with pytest.raises(ValueError):
        ppu._row_contract(torch.zeros(1, k), True, group)


def test_pointer_abi_offsets():
    assert ctypes.sizeof(runtime.GemmArgs) == 88
    assert runtime.GemmArgs.a.offset == 24
    assert runtime.GemmArgs.output.offset == 64
    assert runtime.GemmArgs.dtype.offset == 72


def test_explicit_source_graph():
    root = Path(__file__).resolve().parents[1]
    builder = runpy.run_path(str(root / "tools/build_ppu.py"))
    assert builder["SOURCES"] == ("int8_gemm.cu", "quantize.cu")
    # NVIDIA GEMM/attention TUs are never globbed into the PPU build.
    assert "rglob" not in (root / "tools/build_ppu.py").read_text()


def test_missing_explicit_library_fails(monkeypatch):
    monkeypatch.setenv("COMFY_KITCHEN_PPU_LIBRARY", "/nonexistent/comfy-ppu.so")
    with pytest.raises(RuntimeError, match="not found"):
        runtime.library_path()


def test_int32_bound_is_conservative():
    assert 131040 * 128 * 128 <= 2**31 - 1
    assert 131072 * 128 * 128 > 2**31 - 1
