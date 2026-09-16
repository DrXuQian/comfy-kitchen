# PPU INT8 backend

This fork adds **INT8 linear only**, without replacing the NVIDIA/HIP backends or
requiring model weight repacking. It is not a port of every comfy-kitchen op.

## Architecture and supported contract

| Stage | Implementation | Storage / arithmetic |
|---|---|---|
| Activation quantization | Original `cuda/ops/int8_linear.cu`, compiled by HGGC | FP32/FP16/BF16 input, INT8 row data and FP32 row scales |
| ConvRot256 | Original regular-Hadamard warp / ConvRot64 kernels | Same rotation convention as existing offline weights |
| GEMM | actlize `MainloopPPUAiu`, native `PPU0010_16x16x32_S32S8S8S32_TN` | A `[M,K]`, W `[N,K]` contiguous, INT32 accumulation |
| Epilogue | `ppu/int8_epilogue.hpp` using the MMA's real `partition_C` | `cast_out((float(acc)*xs[row])*ws[col]+bias[col])` |
| Optional activation / residual | Existing PyTorch helpers | Supported, **not claimed fused** in this version |

Weight scale may be scalar or per-output-channel; bias is converted to output
dtype before addition, as in the upstream CUDA EVT. FP32/FP16/BF16 outputs;
arbitrary M/N tails; K must be positive, divisible by 32, and <= 131040 to
exclude INT32 overflow even for all `-128` inputs. ConvRot requires group 256
and K divisible by 256; very large rows may exceed device shared memory and are
explicitly rejected by the SDK, not silently changed. No FP8/FP4/attention port,
no split-K or persistent scheduling, no M=1-specialized GEMV in this first version.

Six independently selectable tile/warp/stage configurations live in
`ppu/int8_configs.inc`. Defaults (ID 0 below M128, ID 1 otherwise) are **starting
points, not measured PPU winners**. The sweep prints every config's actual
threads/shared memory/occupancy and both prequantized core and quant+core timing.

`third_party/actlize` tracks `ppu-w4a16-dev` and is pinned by the gitlink.
The PPU build explicitly includes two translation units, not a directory glob.
Only two upstream shared headers gain a target-guarded native runtime include;
NVIDIA sees the unchanged include branches. The native library uses a versioned
C ABI and does not link ATen, Python, nanobind or libtorch. Thus it has no
torch-2.8-vs-2.9 C++ ABI coupling, but **does** require a compatible PPU SDK/runtime.

## Build and admission on box

```bash
git clone https://github.com/DrXuQian/comfy-kitchen.git
cd comfy-kitchen
git submodule update --init third_party/actlize
PPU_SDK=/usr/local/PPU_SDK bash tools/run_ppu_int8_box.sh
```

This compiles locally on that box, checks all 6 configs x 3 output dtypes, tails,
independent INT64 GEMM reference, row/column scales, bias, public dispatch,
nondefault stream, eight repeats, fixed stochastic seed, and both ConvRot paths.
No downloading/building NVIDIA CUTLASS or FlashAttention is involved. Output goes
under `/workspace/`; the script does not install or overwrite an existing package.

To install for ComfyUI after device admission:

```bash
PPU_SDK=/usr/local/PPU_SDK COMFY_KITCHEN_BUILD_PPU=1 \
  python -m pip install --no-build-isolation --no-deps .
```

Use ComfyUI's Python environment. Keep the PPU SDK runtime in `LD_LIBRARY_PATH`.
The backend selects only devices whose measured name contains `PPU`, not NVIDIA
devices with a coincidentally compatible capability number. A missing library is
listed as unavailable with a reason. Force `ck.use_backend("ppu")` when checking
integration so a fallback cannot masquerade as PPU admission.

## Sweep / profile

```bash
M=4096 N=4096 K=4096 CONVROT=1 SWEEP=1 \
  PPU_SDK=/usr/local/PPU_SDK bash tools/run_ppu_int8_box.sh
```

Set M/N/K from the **full GEMM call**, not from the attention sequence length
alone: the truncated `DefaultGemmWithVisitor<signed char,...>` profiler name does
not establish all three dimensions. Kernels run sequentially on one stream.
The script uses warm/reused weights, aggregate events (including launch idle),
records raw samples + SHA + binary hash + device/runtime, and refuses to call
overlapping best/runner-up timing envelopes a resolved winner. This initial
six-config space is finite and explicit, **not an exhaustive optimization space**.
Do not compare core-only numbers to a quantization-inclusive baseline.

After finding a candidate, set `COMFY_KITCHEN_PPU_INT8_CONFIG=<ID>` in the
ComfyUI process to select it. This is an explicit diagnostic/selection override,
not an auto-tuning cache or a performance guarantee for other shapes.

For ACU, use the same environment/library and run e.g.:

```bash
mkdir -p /workspace/comfy-kitchen-ppu-acu
/sim/eec/shared/junfu.qx/asight/bin/acu -f \
  -o /workspace/comfy-kitchen-ppu-acu/int8 --set full \
  python tools/ppu_int8_sweep.py --m 4096 --n 4096 --k 4096 \
  --convrot --warmup 1 --samples 1 --iterations 1 \
  --out /workspace/comfy-kitchen-ppu-acu
```

ACU time is instrumented; use the unprofiled sweep for latency. INT8 throughput
is reported in TOPS. No hard-coded fp16 peak is used to manufacture an INT8 MFU;
`--peak-tops` is optional and must reflect the actual device/precision.

## Local evidence and its limits

```bash
python -m pytest tests/test_ppu_contract.py tests/test_constraints.py tests/test_backends.py -q
PPU_SDK=/path/to/sdk python tools/check_ppu_local.py
PPU_SDK=/path/to/sdk python tools/build_ppu.py
```

The CPU proof instantiates the **same** config types, lowering, MMA ownership and
epilogue as production, with an independent row-major scalar anchor. It covers
4,144,488 output checks across 6 configs, 3 types, tails, scalar/vector scales and
optional bias. Duplicate-owner and wrong-scale controls must fail normally.
Only unused device-constant registration is stubbed for that CPU executable;
launch/runtime/math functions are not stubbed. The production shared library
links the real SDK with `-z defs`. Missing SDK returns SKIP (77) for the host-type
proof, not PASS. A real compiler error is FAIL.

Local compilation and layout proofs **do not establish device MMA correctness,
PPU speedup, high MFU or equivalence to the user's installed original binary**.
These remain box verdicts. The box script is fail-closed on numeric failures.
The underlying integer MMA/AIU and runtime stream ordering are device boundaries.

### SDK 2.1.1 runtime loader correction

The SDK ships `libhggcrt1.so -> libhggcrt.13.0.so`, but its
`libhggc_wrapper.so` still explicitly dlopens `libhggcrt.12.0.so` during module
registration. The initial build accidentally linked the wrapper first. This
caused the reported load failure before any correctness test launched.

The build now links the same SDK's **native runtime directly**, and checks ELF
dependencies to reject wrapper linkage. The C-only loader uses local deep binding
so an already-global wrapper cannot interpose its stale lookup. No runtime
symlinks, SDK file changes, old-library download, or CUDA facade changes are
needed. Pull and rebuild; the original wrapper-linked binary is not corrected
by a Python-only update.

The exact old binary reproduces `cannot open ... libhggcrt.12.0.so` locally.
Relinking the same two objects to the native runtime passes real SDK module
registration / ABI checks, including with the old wrapper preloaded globally.
This check launches no device work. On this host it uses an isolated compatible
glibc/libstdc++ loader; no fake driver or launch-skip environment is involved.

Initial local validation (2026-09-16, SDK 2.1.1):

- Native shared library and the `setup.py build_ext` installation path both
  compiled and linked; generated ISA contains `v.mma.i32.i8.i8.m16n16k32`
  and `vmem.aiu.ld.tsm...b8`.
- The above actual-type CPU proof passed all 4,144,488 checks; both planted
  defects failed with the intended diagnostics.
- PPU contracts + upstream constraints/registry/input-activation/residual tests:
  **62 PASS / 76 SKIP / 0 FAIL**. Skips require a GPU/native GPU backend absent
  on this local host. They are not device passes.
- PPU device smoke/performance: **NOT RUN**. The local host has no usable PPU;
  SDK shared-runtime loading also needs the SDK's supported system runtime.
