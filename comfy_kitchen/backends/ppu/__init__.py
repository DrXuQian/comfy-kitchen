# SPDX-License-Identifier: Apache-2.0
"""PPU INT8 linear: upstream row quantizer + actlize AIU + fused epilogue.

Unsupported operations remain with the existing registry. Native errors are
never caught and retried using an unrelated backend.
"""

import ctypes as C
import math
import os
import sys

import torch

from comfy_kitchen.backends._activations import apply_input_act, apply_residual
from comfy_kitchen.constraints import FunctionConstraints, ValidationResult
from comfy_kitchen.registry import registry

from . import runtime
from .runtime import configurations, resources  # noqa: F401


def _require_ppu(tensor):
    if not runtime.is_ppu_device(tensor.device):
        raise ValueError("PPU backend requires a PPU tensor, not just a CUDA tensor")


def _same_device(reference, **tensors):
    for name, value in tensors.items():
        if value is not None and value.device != reference.device:
            raise ValueError(f"{name} must be on {reference.device}, got {value.device}")


def _row_contract(x, convrot=False, group_size=256):
    if x.dtype not in runtime.DTYPES or x.ndim < 1:
        raise ValueError("row quantization needs fp32/fp16/bf16 [..., K]")
    k = x.shape[-1]
    if not 0 < k <= 131040 or math.prod(x.shape[:-1]) > 2**31 - 1:
        raise ValueError("unsupported row quantization extent")
    if convrot and (group_size != 256 or k % 256):
        raise ValueError("PPU fused ConvRot supports group_size=256 and K divisible by 256")


def _quantize(x, *, convrot=False, group_size=256, stochastic_rounding=0):
    _require_ppu(x)
    _row_contract(x, convrot, group_size)
    original = x.shape
    x = x.reshape(-1, x.shape[-1]).contiguous()
    with torch.cuda.device(x.device):
        q = torch.empty_like(x, dtype=torch.int8)
        scales = torch.empty((x.shape[0], 1), dtype=torch.float32, device=x.device)
        if x.shape[0]:
            seed = int(stochastic_rounding or 0)
            stochastic = seed > 0
            runtime.check(
                runtime.library().comfy_ppu_quantize_int8(
                    x.data_ptr(),
                    q.data_ptr(),
                    scales.data_ptr(),
                    *x.shape,
                    runtime.DTYPES[x.dtype],
                    int(convrot),
                    group_size,
                    int(stochastic),
                    seed if stochastic else 0,
                    torch.cuda.current_stream(x.device).cuda_stream,
                )
            )
        return q.reshape(original), scales.reshape(*original[:-1], 1)


def quantize_int8_rowwise(x, stochastic_rounding=0):
    return _quantize(x, stochastic_rounding=stochastic_rounding)


def _gemm_contract(a, weight, x_scale, weight_scale, bias, out_dtype):
    if a.ndim != 2 or weight.ndim != 2 or a.dtype != torch.int8 or weight.dtype != torch.int8:
        raise ValueError("GEMM requires int8 A[M,K] and W[N,K]")
    m, k = a.shape
    n, wk = weight.shape
    if k != wk or k <= 0 or k % 32 or k > 131040:
        raise ValueError("K must agree, be divisible by 32 and <=131040 (INT32 accumulation bound)")
    if m > 2**31 - 1 or n > 65535 * 128:
        raise ValueError("GEMM exceeds supported launch extents")
    if x_scale.numel() != m or weight_scale.numel() not in (1, n):
        raise ValueError("expected one scale per A row and scalar/per-output-channel weight scale")
    if bias is not None and bias.numel() != n:
        raise ValueError("bias must have N elements")
    if out_dtype not in runtime.DTYPES:
        raise ValueError("output must be fp32/fp16/bf16")
    _same_device(a, weight=weight, x_scale=x_scale, weight_scale=weight_scale, bias=bias)


def int8_gemm(
    a, weight, x_scale, weight_scale, bias=None, out_dtype=torch.bfloat16, *, config=None
):
    """Prequantized core. Config IDs are listed by configurations(); no hidden sweep."""
    _require_ppu(a)
    _gemm_contract(a, weight, x_scale, weight_scale, bias, out_dtype)
    if config is None:
        config = int(os.environ.get("COMFY_KITCHEN_PPU_INT8_CONFIG", "-1"))
    with torch.cuda.device(a.device):
        a, weight = a.contiguous(), weight.contiguous()
        # A contiguous storage-offset view may still be misaligned for AIU.
        if a.data_ptr() % 16:
            a = a.clone()
        if weight.data_ptr() % 16:
            weight = weight.clone()
        xs = x_scale.to(torch.float32).reshape(-1).contiguous()
        ws = weight_scale.to(torch.float32).reshape(-1).contiguous()
        b = None if bias is None else bias.to(out_dtype).reshape(-1).contiguous()
        out = torch.empty((a.shape[0], weight.shape[0]), dtype=out_dtype, device=a.device)
        args = runtime.GemmArgs(
            a.shape[0],
            weight.shape[0],
            a.shape[1],
            a.data_ptr(),
            weight.data_ptr(),
            xs.data_ptr(),
            ws.data_ptr(),
            None if b is None else b.data_ptr(),
            out.data_ptr(),
            runtime.DTYPES[out_dtype],
            int(ws.numel() == 1),
            config,
        )
        runtime.check(
            runtime.library().comfy_ppu_int8_gemm(
                C.byref(args), torch.cuda.current_stream(a.device).cuda_stream
            )
        )
        return out


def int8_linear(
    x,
    weight,
    weight_scale,
    bias=None,
    out_dtype=None,
    convrot=False,
    convrot_groupsize=256,
    input_act=None,
    input_act_weight=None,
    input_act_eps=0.0,
    residual=None,
    residual_scale=None,
):
    _require_ppu(x)
    _same_device(
        x,
        weight=weight,
        weight_scale=weight_scale,
        bias=bias,
        input_act_weight=input_act_weight,
        residual=residual,
        residual_scale=residual_scale,
    )
    if residual is not None and residual_scale is None:
        raise ValueError("residual requires residual_scale")
    # Keep upstream activation/residual semantics; these are not claimed fused.
    x = apply_input_act(x, input_act, input_act_weight, input_act_eps)
    _row_contract(x, convrot, convrot_groupsize)
    if weight.ndim != 2 or x.shape[-1] != weight.shape[-1]:
        raise ValueError("INT8 linear input/weight K must agree")
    if weight.dtype != torch.int8 or x.shape[-1] % 32:
        raise ValueError("INT8 linear requires int8 weight and K divisible by 32")
    if weight_scale.numel() not in (1, weight.shape[0]) or (
        bias is not None and bias.numel() != weight.shape[0]
    ):
        raise ValueError("weight scale must be scalar/per-channel; bias must have N elements")
    shape = (*x.shape[:-1], weight.shape[0])
    q, scale = _quantize(x.reshape(-1, x.shape[-1]), convrot=convrot, group_size=convrot_groupsize)
    out = int8_gemm(q, weight, scale, weight_scale, bias, out_dtype or x.dtype).reshape(shape)
    return apply_residual(out, residual, residual_scale)


def _admit(kwargs):
    x = kwargs.get("x")
    if not isinstance(x, torch.Tensor) or not runtime.is_ppu_device(x.device):
        return ValidationResult.fail("x", "requires measured PPU device identity")
    # Admit the backend, not a guessed NVIDIA compute capability. Function-level
    # validators reject malformed arguments before passing pointers to C++.
    return ValidationResult.ok()


def _register():
    if not torch.cuda.is_available() or not any(
        runtime.is_ppu_device(f"cuda:{i}") for i in range(torch.cuda.device_count())
    ):
        registry.mark_unavailable("ppu", "PPU device not available")
        return
    try:
        runtime.library()
    except (OSError, RuntimeError) as error:
        registry.mark_unavailable("ppu", str(error))
        return
    constraint = FunctionConstraints(default_devices=frozenset({"cuda"}), call_rules=(_admit,))
    registry.register(
        "ppu",
        sys.modules[__name__],
        {name: constraint for name in ("int8_linear", "quantize_int8_rowwise")},
    )


_register()
