"""CPU tests for the multi-shape measurement contract; no device performance claims."""

import argparse
import copy
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import ppu_int8_shape_suite as suite  # noqa: E402
from ppu_int8_check import assert_bits  # noqa: E402
from ppu_int8_config_space import census  # noqa: E402


def result_fixture():
    space = json.loads(json.dumps(census()))
    binding = {"sha256": "binary-fixture", "config_census": space}
    protocol = {
        "samples": 3,
        "iterations": 3,
        "confirm_samples": 7,
        "confirm_iterations": 20,
        "confirm_top": 8,
        "peak_tops": 1000.0,
        "bias": "vector",
    }
    rows = [
        {"config": r["id"], "name": suite.config_name(r), "role": role}
        for r in space["accepted"]
        for role in suite.ROLES
    ]
    confirmed = [
        {**r, "samples_us": [1.0] * 7, "median_us": 1.0}
        for r in rows
        if r["name"] in suite.CONTROL_NAMES
    ]
    data = {
        "complete": True,
        "metadata": {
            "binary_sha256": binding["sha256"],
            "config_census": space,
            "candidate_count": len(space["accepted"]),
            "device": "CPU contract fixture, no PPU measurement",
            "device_properties": "fixture-device-1",
            "torch": "fixture-version",
            "ppu_versions": {"runtime": "fixture-runtime"},
            "args": {
                "m": 73774,
                "n": 4096,
                "k": 4096,
                "dtype": "bf16",
                "convrot": True,
                **protocol,
            },
        },
        "results": rows,
        "skips": [],
        "confirmations": confirmed,
        "winners": {
            role: {
                "binding": "all-screened+top-confirmed",
                "verdict": "UNRESOLVED",
                "best": suite.CONTROL_NAMES[2],
                "median_us": 1.0,
                "logical_tops": 1.0,
                "utilization_pct": 0.1,
                "gap_us": 0.0,
            }
            for role in suite.ROLES
        },
    }
    return data, binding, protocol


def test_plan_separates_m_n_k_and_has_a_large_tail():
    shapes = suite.DEFAULT_SHAPES
    assert len(set(shapes)) == len(shapes) == 8
    assert {(m, 4096, 4096) for m in (4096, 16384, 73774)}.issubset(shapes)
    assert {
        (73774, n, k)
        for n, k in ((8192, 4096), (4096, 8192), (8192, 8192), (16384, 4096), (4096, 16384))
    }.issubset(shapes)
    assert 73774 % 256 != 0


def test_h3_geometry_uses_hidden_not_attention_width_for_the_residual_stream():
    assert suite.minimax_h3_shapes(73774) == (
        (73774, 21504, 5376),
        (73774, 5376, 7168),
        (73774, 28672, 5376),
        (73774, 5376, 14336),
    )
    assert all(shape[2] % 256 == 0 for shape in suite.minimax_h3_shapes(73774))
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/ppu_int8_shape_suite.py"),
            "--dry-run",
            "--suite",
            "minimax-h3",
            "--tokens",
            "16384",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "M=16384 N=21504 K=5376" in result.stdout
    assert "bias=none" in result.stdout


def test_complete_sweep_and_unresolved_are_distinct():
    data, binding, protocol = result_fixture()
    suite.validate_result(data, (73774, 4096, 4096), binding, protocol)
    assert all(w["verdict"] == "UNRESOLVED" for w in data["winners"].values())


@pytest.mark.parametrize(
    "plant",
    [
        "missing",
        "duplicate",
        "wrong-binary",
        "wrong-shape",
        "missing-control",
        "short-confirmation",
        "mixed-protocol",
        "false-coordinate",
        "not-complete",
    ],
)
def test_invalid_evidence_is_not_a_success(plant):
    data, binding, protocol = result_fixture()
    if plant == "missing":
        data["results"].pop()
    elif plant == "duplicate":
        data["results"][-1] = copy.deepcopy(data["results"][1])
    elif plant == "wrong-binary":
        data["metadata"]["binary_sha256"] = "other-binary"
    elif plant == "wrong-shape":
        data["metadata"]["args"]["m"] = 4096
    elif plant == "missing-control":
        data["confirmations"].pop()
    elif plant == "short-confirmation":
        data["confirmations"][0]["samples_us"].pop()
    elif plant == "mixed-protocol":
        data["metadata"]["args"]["confirm_iterations"] = 3
    elif plant == "false-coordinate":
        data["results"][0]["name"] = "another-shape"
    else:
        data["complete"] = False
    with pytest.raises(ValueError):
        suite.validate_result(data, (73774, 4096, 4096), binding, protocol)


def test_reasoned_resource_skip_keeps_the_denominator():
    data, binding, protocol = result_fixture()
    removed = data["results"].pop(0)
    data["results"].pop(0)
    data["skips"] = [removed | {"reason": "OCCUPANCY_API_ZERO_ACTIVE_BLOCKS"}]
    suite.validate_result(data, (73774, 4096, 4096), binding, protocol)
    data["skips"][0]["reason"] = ""
    with pytest.raises(ValueError, match="unreasoned"):
        suite.validate_result(data, (73774, 4096, 4096), binding, protocol)


def test_library_binding_uses_bytes_not_filename(tmp_path):
    path = tmp_path / "_native.so"
    path.write_bytes(b"not-a-device-library-CPU-contract-only")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"binary_sha256": digest, "sha": "source-fixture", "config_census": census()}
    path.with_suffix(".so.json").write_text(json.dumps(manifest))
    assert suite.bind_library(path)["sha256"] == digest
    path.write_bytes(b"wrong-binary")
    with pytest.raises(ValueError, match="SHA256"):
        suite.bind_library(path)


@pytest.mark.parametrize("shape", ["0,4096,4096", "1,2", "1,4096,96", "1,4096,131072"])
def test_invalid_shape_fails_before_device_work(shape):
    with pytest.raises(argparse.ArgumentTypeError):
        suite.shape_arg(shape)


def test_cli_dry_run_needs_no_library_and_custom_shapes_work():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/ppu_int8_shape_suite.py"),
            "--dry-run",
            "--shapes",
            "73774,7168,7168",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "M=73774 N=7168 K=7168" in result.stdout
    assert "compile=0" in result.stdout


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_byte_comparator_catches_last_byte_and_signed_zero(dtype):
    expected = torch.zeros(17, 35, dtype=dtype)
    assert_bits(expected, expected.clone())
    wrong = expected.clone()
    wrong.view(torch.uint8).flatten()[-1] ^= 1
    with pytest.raises(AssertionError, match="raw byte mismatch"):
        assert_bits(wrong, expected)
    wrong = expected.clone()
    wrong[-1, -1] = -0.0
    with pytest.raises(AssertionError, match="raw byte mismatch"):
        assert_bits(wrong, expected)


def test_byte_comparator_handles_noncontiguous_and_rejects_schema_change():
    expected = torch.arange(40, dtype=torch.float32).reshape(4, 10).T
    assert_bits(expected, expected.contiguous())
    with pytest.raises(AssertionError, match="shape and dtype"):
        assert_bits(expected, expected.to(torch.bfloat16))


@pytest.mark.parametrize("changed_device", [False, True])
def test_supervisor_is_sequential_and_refuses_mixed_device_evidence(
    tmp_path, monkeypatch, changed_device
):
    data, binding, _ = result_fixture()
    binding["path"] = "cpu-fixture-only.so"
    monkeypatch.setattr(suite, "bind_library", lambda _: binding)
    calls = []

    class CpuFixtureProcess:
        def __init__(self, cmd, **kwargs):
            assert kwargs["env"]["COMFY_KITCHEN_PPU_LIBRARY"] == binding["path"]
            calls.append(cmd)
            payload = copy.deepcopy(data)
            for key in ("m", "n", "k"):
                payload["metadata"]["args"][key] = int(cmd[cmd.index("--" + key) + 1])
            if changed_device and len(calls) == 2:
                payload["metadata"]["device_properties"] = "fixture-device-2"
            output = Path(cmd[cmd.index("--out") + 1]) / "sweep-fixture.json"
            output.write_text(json.dumps(payload))
            self.stdout = io.StringIO("[fixture] supervisor CPU contract only\n")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stdout.close()

        def wait(self):
            return 0

    monkeypatch.setattr(suite.subprocess, "Popen", CpuFixtureProcess)
    monkeypatch.setattr(suite.subprocess, "check_output", lambda *a, **kw: "source-fixture\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "suite",
            "--library",
            "cpu-fixture-only.so",
            "--out",
            str(tmp_path),
            "--shapes",
            "73774,4096,4096",
            "73774,8192,8192",
        ],
    )
    if changed_device:
        with pytest.raises(ValueError, match="identity changed"):
            suite.main()
    else:
        assert suite.main() == 0
    summary = json.loads((tmp_path / "shape-summary.json").read_text())
    assert summary["complete"] == (not changed_device)
    assert len(summary["runs"]) == (1 if changed_device else 2)
    assert len(calls) == 2
    assert "UNRESOLVED" in (tmp_path / "summary.md").read_text()
    # A competing invocation must not overwrite an existing bundle.
    before = (tmp_path / "shape-summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        suite.main()
    assert (tmp_path / "shape-summary.json").read_bytes() == before
