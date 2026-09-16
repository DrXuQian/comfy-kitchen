"""CPU contracts for the benchmark; not device correctness/performance claims."""

import ast
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def bench():
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        spec = importlib.util.spec_from_file_location(
            "ppu_blaslt_compare", ROOT / "tools/ppu_blaslt_compare.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(ROOT / "tools"))


def test_only_explicit_unsupported_is_skip(bench):
    bench.check_status(0, lambda: "unused")
    with pytest.raises(bench.Unsupported, match="vendor unsupported"):
        bench.check_status(2, lambda: "vendor unsupported")
    for rc in (1, 3, -1):
        with pytest.raises(RuntimeError, match="real failure") as caught:
            bench.check_status(rc, lambda: "real failure")
        assert not isinstance(caught.value, bench.Unsupported)


def test_numerical_negative_is_fail_never_skip(bench):
    with pytest.raises(AssertionError, match="raw byte mismatch") as caught:
        bench.assert_bits(torch.tensor([1.0]), torch.tensor([2.0]))
    assert not isinstance(caught.value, bench.Unsupported)


@pytest.mark.parametrize(
    "ours,vendor,verdict",
    [
        ((250, 245, 255), (300, 290, 310), "ACTLIZE-WINS"),
        ((300, 290, 310), (250, 245, 255), "ACBLASLT-WINS"),
        ((250, 245, 255), (251, 248, 257), "UNRESOLVED"),
        ((250, 245, 255), (260, 255, 270), "UNRESOLVED"),
    ],
)
def test_verdict_cannot_ignore_resolution(bench, ours, vendor, verdict):
    def stats(values):
        return dict(zip(("median_us", "min_us", "max_us"), values))

    assert bench.compare_envelopes(stats(ours), stats(vendor))["verdict"] == verdict


def test_signed_nonsquare_fixture_detects_wrong_transpose_and_scale(bench):
    torch.manual_seed(1901)
    a = torch.randint(-17, 18, (64, 96), dtype=torch.int8)
    w = torch.randint(-13, 14, (128, 96), dtype=torch.int8)
    xs = (torch.arange(64) % 7 + 1).float() / 32
    ws = (torch.arange(128) % 11 + 1).float() / 64
    bias = ((torch.arange(128) % 7 - 3).float() / 16).bfloat16()
    acc = a.long() @ w.long().T
    want = bench.oracle(a, w, xs, ws, bias, torch.bfloat16)
    with pytest.raises(AssertionError):
        bench.assert_bits(acc.T.contiguous().reshape_as(acc), acc)
    with pytest.raises(AssertionError):
        bench.assert_bits(bench.oracle(a, w, torch.ones_like(xs), ws, bias, torch.bfloat16), want)


def test_benchmark_does_not_enter_shipping_source_graph():
    text = (ROOT / "tools/build_ppu.py").read_text()
    assert 'SOURCES = ("int8_gemm.cu", "quantize.cu")' in text
    assert "blaslt_probe" not in text


def test_int32_only_is_not_the_complete_comparison():
    text = (ROOT / "tools/ppu_blaslt_compare.py").read_text()
    assert '"acblaslt-int32-only-DIAGNOSTIC"' in text
    assert 'r["role"] == "acblaslt-scale-bias"' in text

    def roles(source):
        calls = [
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "compare_envelopes"
        ]
        assert len(calls) == 1
        return [ast.literal_eval(argument.slice) for argument in calls[0].args]

    expected = ["actlize-fused", "acblaslt-scale-bias"]
    assert roles(text) == expected
    mutant = text.replace(
        'confirmed["acblaslt-scale-bias"]', 'confirmed["acblaslt-int32-only-DIAGNOSTIC"]'
    )
    assert roles(mutant) != expected  # A raw-GEMM win cannot silently become an end-to-end win.
