"""Shipping selector, table-ID seams, and explicit-override precedence."""

import ctypes
import importlib.util
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from comfy_kitchen.backends.ppu import runtime

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ROOT / "comfy_kitchen/backends/ppu"


def space_module():
    spec = importlib.util.spec_from_file_location("space", ROOT / "tools/ppu_int8_config_space.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("extended", [False, True])
def test_real_native_selector_boundaries_and_regression_plant(tmp_path, extended):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("C++ compiler unavailable: cannot execute the actual native selector")
    extra = []
    if extended:
        table, _, _ = space_module().generate(tmp_path / "generated")
        extra += [f'-DCOMFY_PPU_INT8_CONFIGS="{table}"']
    binary = tmp_path / "selector"
    command = [
        compiler,
        "-std=c++17",
        "-O2",
        f"-I{INCLUDE}",
        *extra,
        str(ROOT / "tests/ppu_selector.cpp"),
        "-o",
        str(binary),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    assert "cases=2187" in result.stdout
    assert f"balanced_id={94 if extended else 6}" in result.stdout
    assert "long_k_id=5" in result.stdout
    plant = subprocess.run([str(binary), "--plant-old-default"], capture_output=True, text=True)
    assert plant.returncode == 1 and "selector mismatch 4096x4096x4096 dtype=2 id=1" in plant.stderr


def test_missing_selected_geometry_is_compile_failure(tmp_path):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("C++ compiler unavailable for the missing-config negative control")
    table = tmp_path / "wrong.inc"
    table.write_text(
        "\n".join(
            line
            for line in (INCLUDE / "int8_configs.inc").read_text().splitlines()
            if not line.startswith("COMFY_PPU_INT8_CONFIG(6,")
        )
    )
    result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            f"-I{INCLUDE}",
            f'-DCOMFY_PPU_INT8_CONFIGS="{table}"',
            "-fsyntax-only",
            str(ROOT / "tests/ppu_selector.cpp"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "selector references a config absent from the compiled table" in result.stderr


def test_product_added_row_does_not_renumber_the_measured_sweep():
    space = space_module()
    rows = space.product_rows()
    assert len(rows) == 7
    assert rows[6] == (64, 256, 128, 64, 32, 2)
    data = space.census()
    assert len(data["accepted"]) == 285
    assert tuple(data["accepted"][94][k] for k in space.AXES) == rows[6]
    assert tuple(data["accepted"][5][k] for k in space.AXES) == rows[5]


def test_native_selector_query_and_old_library_fail_closed(monkeypatch):
    calls = []
    lib = SimpleNamespace(
        comfy_ppu_int8_select_config=lambda *a: calls.append(a) or 6,
        comfy_ppu_last_error=lambda: b"invalid selection",
    )
    monkeypatch.setattr(runtime, "library", lambda: lib)
    assert runtime.select_config(4096, 4096, 4096) == 6
    assert calls == [(4096, 4096, 4096, 2)]
    lib.comfy_ppu_int8_select_config = lambda *a: -1
    with pytest.raises(ValueError, match="invalid selection"):
        runtime.select_config(-1, 4096, 4096)
    monkeypatch.setattr(runtime, "library", SimpleNamespace)
    with pytest.raises(RuntimeError, match="predates the measured selector"):
        runtime.select_config(4096, 4096, 4096)


def test_argument_env_auto_precedence_and_coordinate_names(monkeypatch):
    auto = []
    monkeypatch.setattr(runtime, "select_config", lambda *args: auto.append(args) or 6)
    monkeypatch.setattr(runtime, "_config_ids_by_name", lambda: {"64x256x128_w64x32_s2": 6})
    monkeypatch.delenv("COMFY_KITCHEN_PPU_INT8_CONFIG", raising=False)
    shape = (77777, 7168, 6144)
    assert runtime.resolve_config(None, *shape) == 6
    assert auto == [(*shape, torch.bfloat16)]
    monkeypatch.setenv("COMFY_KITCHEN_PPU_INT8_CONFIG", "5")
    assert runtime.resolve_config(None, *shape) == 5
    assert runtime.resolve_config(0, *shape) == 0
    assert runtime.resolve_config("64x256x128_w64x32_s2", *shape) == 6
    assert len(auto) == 1
    assert runtime.resolve_config(-1, *shape) == 6  # Explicit auto overrides the env too.
    assert len(auto) == 2
    for invalid in ("no-such-config", -2, 1.5):
        with pytest.raises(ValueError, match="unknown PPU INT8 config"):
            runtime.resolve_config(invalid, *shape)


def test_new_native_query_does_not_change_pointer_abi():
    assert ctypes.sizeof(runtime.GemmArgs) == 88
