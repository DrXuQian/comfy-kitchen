#!/usr/bin/env python3
"""Device admission for PPU INT8 linear (all configs, not just default)."""

import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

import comfy_kitchen as ck
from comfy_kitchen.backends import ppu


def row_oracle(x):
    # The CUDA quantizer uses absmax * float32(1/127), not division by
    # 127 in a different precision. Its explicit input-dtype rounding is ABI.
    scale = (
        x.abs().amax(-1, keepdim=True).float() * torch.tensor(1 / 127, dtype=torch.float32)
    ).clamp_min(1e-30)
    scaled = x / scale.to(x.dtype)
    quantized = scaled.float().round().nan_to_num(nan=-128).clamp(-128, 127).to(torch.int8)
    return quantized, scale


def oracle(a, w, xs, ws, bias, dtype):
    # Integer matmul is independent of both PPU MMA and our epilogue layout.
    acc = a.cpu().to(torch.int64) @ w.cpu().to(torch.int64).T
    out = acc.float() * xs.cpu().float().reshape(-1, 1)
    out = out * ws.cpu().float().reshape(1, -1)
    if bias is not None:
        out = out + bias.cpu().to(dtype).float().reshape(1, -1)
    return out.to(dtype)


def assert_bits(got, want):
    got, want = got.cpu().contiguous(), want.cpu().contiguous()
    bad = (got.view(torch.uint8) != want.view(torch.uint8)).sum().item()
    if bad:
        raise AssertionError(
            f"raw byte mismatch: {bad}; max_abs={(got.float()-want.float()).abs().max().item()}"
        )


def check():
    if not ppu.runtime.is_ppu_device("cuda:0"):
        raise RuntimeError("device validation requires PPU; NOT RUN is not PASS")
    torch.manual_seed(819)
    device = torch.device("cuda:0")
    cases = 0
    # K=96 and 160 deliberately cross different configs' K-tile boundaries.
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for m, n, k in ((1, 1, 32), (17, 35, 96), (65, 129, 160), (129, 257, 256)):
            ac = torch.randint(-8, 9, (m, k), dtype=torch.int8)
            wc = torch.randint(-8, 9, (n, k), dtype=torch.int8)
            xs = (torch.arange(m) % 7 + 1).float()[:, None] / 16
            ws = (torch.arange(n) % 11 + 1).float() / 32
            bias = ((torch.arange(n) % 9 - 4).float() / 4).to(dtype)
            a, w = ac.to(device), wc.to(device)
            for scalar, biased in ((False, False), (False, True), (True, True)):
                scales = ws[:1] if scalar else ws
                b = bias if biased else None
                expected = oracle(ac, wc, xs, scales, b, dtype)
                for config, name in enumerate(ppu.configurations()):
                    out = ppu.int8_gemm(
                        a,
                        w,
                        xs.to(device),
                        scales.to(device),
                        None if b is None else b.to(device),
                        dtype,
                        config=config,
                    )
                    assert_bits(out, expected)
                    cases += 1
            print(
                f"[PPU INT8] core shape={m}x{n}x{k} dtype={dtype} all-configs RAW-BIT/PASS",
                flush=True,
            )
    # Pipeline reuse, distinct scales and bias, and a nondefault stream. Inputs
    # are allocated/used on this stream, avoiding reliance on default ordering.
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.randint(-15, 16, (2, 17, 256)).to(device=device, dtype=torch.bfloat16) / 8
        w = torch.randint(-8, 9, (129, 256), dtype=torch.int8).to(device)
        ws = ((torch.arange(129) % 13 + 1).float() / 16).to(device)
        bias = torch.arange(129).to(device=device, dtype=torch.bfloat16) / 16
        q, xs = ppu.quantize_int8_rowwise(x)
    stream.synchronize()
    q_ref, xs_ref = row_oracle(x.cpu())
    assert_bits(q, q_ref)
    assert_bits(xs, xs_ref)
    with torch.cuda.stream(stream), ck.use_backend("ppu"):
        results = [ck.int8_linear(x, w, ws, bias, out_dtype=torch.bfloat16) for _ in range(8)]
    stream.synchronize()
    want = oracle(
        q_ref.reshape(-1, 256), w, xs_ref.reshape(-1, 1), ws, bias, torch.bfloat16
    ).reshape(2, 17, 129)
    for result in results:
        assert_bits(result, want)
    print("[PPU INT8] public-dispatch + nondefault-stream + reuse=8/8 RAW-BIT/PASS", flush=True)
    # Explicit stochastic seed is an input, not a boolean or a fresh RNG seed.
    q1, s1 = ppu.quantize_int8_rowwise(x, stochastic_rounding=123)
    q2, s2 = ppu.quantize_int8_rowwise(x, stochastic_rounding=123)
    assert_bits(q1, q2)
    assert_bits(s1, s2)
    # Independent Hadamard anchor for ConvRot256. Integer small magnitudes make
    # the intermediate butterfly and bf16 boundaries exact (no loose tolerance).
    # Regular H4 is not the Sylvester H2 convention! Define the mathematical
    # matrix here, independently of the device radix-4 butterfly code.
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32
    )
    h = h4
    for _ in range(3):
        h = torch.kron(h, h4)
    for rows_count in (3, 17):  # warp path and ConvRot64 prefill path
        xx = torch.randint(-2, 3, (rows_count, 512)).to(torch.bfloat16)
        rotated = ((xx.float().reshape(-1, 256) @ h) / 16).reshape_as(xx).to(xx.dtype)
        qr, sr = row_oracle(rotated)
        qg, sg = ppu._quantize(xx.to(device), convrot=True)
        assert_bits(qg, qr)
        assert_bits(sg, sr)
        expected = oracle(
            qr, torch.ones(5, 512, dtype=torch.int8), sr, torch.ones(5), None, xx.dtype
        )
        actual = ppu.int8_linear(
            xx.to(device),
            torch.ones(5, 512, dtype=torch.int8, device=device),
            torch.ones(5, device=device),
            convrot=True,
        )
        assert_bits(actual, expected)
    # Negative controls: the fixture MUST observe per-row/per-column scaling.
    a = torch.arange(96).reshape(3, 32).remainder(9).to(torch.int8)
    w0 = torch.arange(224).reshape(7, 32).remainder(13).to(torch.int8)
    sx = torch.tensor([1.0, 2.0, 4.0])
    sw = torch.arange(1.0, 8.0)
    reference = oracle(a, w0, sx, sw, None, torch.float32)
    for label, sx_bad, sw_bad in (("row-scale", sx.roll(1), sw), ("column-scale", sx, sw.roll(1))):
        changed = ppu.int8_gemm(
            a.to(device),
            w0.to(device),
            sx_bad.to(device),
            sw_bad.to(device),
            out_dtype=torch.float32,
        )
        assert_bits(changed, oracle(a, w0, sx_bad, sw_bad, None, torch.float32))
        if torch.equal(changed.cpu(), reference):
            raise AssertionError(f"{label} was silently ignored")
        print(f"[PPU INT8 negative] {label} EXPECTED-RED/PASS")
    digest = hashlib.sha256(results[0].cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
    print(f"[PPU INT8] PASS core_cases={cases} convrot=INDEPENDENT/PASS fingerprint={digest}")


if __name__ == "__main__":
    with torch.inference_mode():
        check()
