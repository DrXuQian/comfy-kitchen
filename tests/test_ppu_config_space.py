"""Explicit search denominator and fail-closed resource admission."""

import copy
import importlib.util
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_cartesian_census_has_exact_denominator_and_legacy_prefix():
    space = module("ppu_int8_config_space")
    data = space.census()
    assert data["raw_count"] == 324
    assert len(data["accepted"]) == 285
    assert len(data["rejected"]) == 39
    assert [tuple(r[k] for k in space.AXES) for r in data["accepted"][:6]] == space.legacy_rows()


@pytest.mark.parametrize("plant", ["missing", "duplicate", "false-reason", "legacy-changed"])
def test_census_negative_controls(plant):
    space = module("ppu_int8_config_space")
    data = copy.deepcopy(space.census())
    if plant == "missing":
        data["accepted"].pop()
    elif plant == "duplicate":
        data["accepted"][-1] = copy.deepcopy(data["accepted"][-2])
    elif plant == "false-reason":
        data["rejected"][0]["reasons"] = ["BECAUSE_SLOW"]
    else:
        data["accepted"][0]["stages"] = 2
    with pytest.raises(ValueError):
        space.validate_census(data)


def test_runtime_errors_are_not_resource_skips():
    admission = module("ppu_int8_admission")
    limits = {"max_threads_per_block": 1024, "max_optin_shared_bytes": 262144}
    good = {"threads": 256, "shared_bytes": 98304, "maximum_active_blocks": 2}
    assert admission.resource_reason(good, limits) is None
    assert (
        admission.resource_reason(good | {"maximum_active_blocks": 0}, limits)
        == "OCCUPANCY_API_ZERO_ACTIVE_BLOCKS"
    )
    with pytest.raises(RuntimeError, match="query failed"):
        admission.resource_reason(good | {"maximum_active_blocks": -1}, limits)
    assert (
        admission.resource_reason(
            good | {"shared_bytes": 300000, "maximum_active_blocks": -1}, limits
        )
        == "SHARED_BYTES_EXCEED_DEVICE_OPTIN_LIMIT"
    )


def test_shard_generation_covers_each_case_once(tmp_path):
    space = module("ppu_int8_config_space")
    table, sources, data = space.generate(tmp_path)
    assert len(sources) == 32
    assert table.read_text().count("COMFY_PPU_INT8_CONFIG(") == 285
    cases = []
    for source in sources:
        cases.extend(
            int(v)
            for v in re.findall(
                r"case (\d+): return comfy::ppu::run_int8_config", source.read_text()
            )
        )
    assert cases == list(range(len(data["accepted"])))
