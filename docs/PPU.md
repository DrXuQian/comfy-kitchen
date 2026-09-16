# PPU INT8 backend

## Multi-shape heuristic experiment (reuse the expanded binary)

The 4096-cube expanded sweep admitted `64x256x128_w64x32_s2` on the user's
72-CU PPU: core **208.960 us / 657.729 TOPS / 65.77%** of the declared
1000 INT8 TOPS peak; quant+core **289.366 us**. Both fresh-confirmation winners
were resolved in that 285-row space. This does **not** establish a global
fallback for all shapes. Product defaults remain unchanged pending this experiment.

Run eight controlled shapes without recompiling the library:

```bash
git pull --ff-only
COMFY_KITCHEN_PPU_LIBRARY=/workspace/comfy-kitchen-ppu-e39b8c70-20260916T055334Z-63522/_native.so \
  bash tools/run_ppu_int8_shape_sweep_box.sh
```

The SDK defaults to `/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK`; override `PPU_SDK`
if needed. Set `COMFY_KITCHEN_PPU_LIBRARY` to the exact expanded library that
passed the box sweep; its adjacent `_native.so.json` manifest is required.
No build, backend installation, or implicit fallback occurs in this runner.

| M | N | K | Question |
|---:|---:|---:|---|
| 4096 | 4096 | 4096 | Repeat the admitted cube control |
| 16384 | 4096 | 4096 | Increase M only |
| 73774 | 4096 | 4096 | Large M, including a real tile tail |
| 73774 | 8192 | 4096 | Increase output width N only |
| 73774 | 4096 | 8192 | Increase reduction depth K only |
| 73774 | 8192 | 8192 | Both N and K large |
| 73774 | 16384 | 4096 | Wide-output/expansion probe |
| 73774 | 4096 | 16384 | Long-K/contraction probe |

These are **synthetic heuristic probes, not a verified MiniMax H3 layer list**.
73774 is the previously reported sequence length; the actual linear-layer
N/K (and whether its flattened M is that sequence length) still need call-shape
evidence. Attention head count/dimension alone cannot establish all linear shapes.
Custom shapes can be supplied without editing code or recompiling:

```bash
COMFY_KITCHEN_PPU_LIBRARY=/workspace/<expanded-run>/_native.so \
  bash tools/run_ppu_int8_shape_sweep_box.sh --shapes 16384,4096,4096 73774,8192,8192
```

Each shape screens **all 285 configs**, both roles, with 3 samples x 3 launches.
Fresh interleaved confirmation uses 7 samples x 20 launches for the top eight
per role **plus** current product fallback config1, old six-row winner config5,
and the 4096-cube winner (selected by coordinates, not by fragile ID94).
Controls do not prune the competition. All raw outputs are byte-compared against
config0 on device outside timing; an independent sampled CPU-int64 oracle remains.
This avoids copying gigabyte outputs back to CPU after every timing batch.
Timing still uses the public API, warm/reused allocations, with launch idle included.
Overlapping envelopes remain UNRESOLVED, including overlap with an unconfirmed row.
Lower-cost screening is not itself a statistically established winner.

The runner is sequential, retains per-shape logs/raw JSON, and writes `summary.md`
plus **`shape-summary.json` (upload this single file)** containing all samples and
identities. Missing candidates, changed binary/device/runtime, numerical failures,
or missing controls fail the suite; resource SKIPs retain their reasons.
No result is installed into the selector automatically.

Local validation for this runner: **24 orchestration/comparison contracts pass**,
including missing/duplicate candidates, wrong binary/shape/device, missing
confirmation controls, and a changed last output byte or signed zero. The wider
CPU suite reports **101 PASS / 13 backend/environment SKIP / 0 FAIL**. Shell
syntax, dry-run, real existing 285-row library/manifest binding, and formatting
also pass. These are not new PPU numerical or timing measurements.

Use the result to test, not assume, a shape heuristic: compare fallback and cube
winner regrets separately for core and quant+core, inspect tile/warp/stage and
occupancy changes, and retain unresolved ties. A future heuristic needs held-out
shapes before claiming generalization; unknown shapes keep a documented safe
fallback rather than running an undisclosed full sweep on ComfyUI's first frame.

## Expanded INT8 sweep (opt-in, product defaults unchanged)

```bash
git pull --ff-only
PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK JOBS=8 \
  bash tools/run_ppu_int8_extended_sweep_box.sh
```

Defaults: `M=N=K=4096`, BF16 output, ConvRot, 1000 dense INT8 TOPS denominator.
The bounded Cartesian search uses `TileM/TileN/TileK={64,128,256}`,
`WarpM/WarpN={32,64}`, `Stages={2,3,4}`: **324** raw rows, **285** compile
candidates, **39** excluded. All exclusions and their coordinates appear in
`build/generated/int8_config_census.json` and in the output JSON. Reasons are
CTA >1024 threads or mainloop >256 KiB shared storage (both reasons retained
when applicable), not guessed performance. This is an expanded search, **not
every possible legal PPU configuration**; e.g. WarpN128 is outside this pass.

The original six rows retain IDs 0..5. Normal builds still compile only those
six and retain the same default selector. The expanded build uses the same
`Int8Config`/mainloop/epilogue with sharded host dispatch: 32 TUs, three output
dtypes, full link, no per-config mainloop fork. Actual kernel types assert the
enumerated CTA thread and shared-byte counts. Device memory/register residency
is checked separately: resource-inadmissible rows print SKIP with the actual
limits, unexplained API errors and numerical mismatches FAIL. Host ownership
and scale/bias negative tests can cover the full table using
`tools/check_ppu_local.py --extended`.

Every device-admitted config gets correctness checks and both core and
quant+core measurements. These are the existing **public-API aggregate-event**
measurements, not the Lt comparison's prepared-call timing. Then each role's
top eight plus legacy config5 are measured again in shuffled interleaved rounds.
`[PPU INT8 winner]` reports median time, TOPS/utilization, config ID, runner-up
gap and speedup versus config5 in the same confirmation. Overlapping envelopes
remain UNRESOLVED (including an unconfirmed candidate overlapping the winner).
`[PPU INT8 denominator]` explicitly accounts for all 285 candidates. No winner
is automatically installed as the production default.

Local validation: all 34 compilation units and the real shared-library link
passed with SDK 2.1.1; loading reports all 285 names/IDs exactly. The full-table
host proof passed **202,111,692** output checks across three dtypes; duplicate
owner and incorrect scale negative controls failed as intended. The original
six-row build/link and **4,144,488** host checks also passed. Original CPU contracts:
**77 PASS / 4 environment/device SKIP**. The subsequently returned 4096-cube
device sweep at `e39b8c7` measured all 285 rows, both roles, with zero resource
SKIPs; its fresh-confirmation winner is recorded above. The new large-M shapes
remain unmeasured until the multi-shape box run returns.

## Vendor Lt comparison (benchmark only)

On PPU, the vendor Lt library is SDK `libacblasLt.so` / `acblasLtMatmul`,
not NVIDIA's `libcublasLt.so`. Run the same 4096x4096x4096 BF16-output workload:

```bash
git pull --ff-only
PPU_SDK=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK \
  M=4096 N=4096 K=4096 DTYPE=bf16 CONVROT=1 PEAK_TOPS=1000 \
  bash tools/run_ppu_blaslt_box.sh
```

To reuse an already validated `_native.so`, additionally set `BUILD=0` and
`COMFY_KITCHEN_PPU_LIBRARY=/workspace/<prior-PASS-run>/_native.so`. The small Lt
benchmark bridge is still built separately. Shipping kernels/configs are unchanged.

The script requests 32 heuristics (`HEURISTICS`, maximum 256), allows 64 MiB
workspace (`WORKSPACE_MIB`), and also measures the library's default algorithm.
This is **not an exhaustive search of every vendor configuration**. A vendor
NOT_SUPPORTED is reported as SKIP; no valid candidate gives rc=2, never PASS.
Numerical/runtime failures remain FAIL, not SKIP. No fallback to FP16 GEMM.

Three main rows are kept separate:

- `acblaslt-int32-only-DIAGNOSTIC`: INT8 x INT8 -> INT32; **not** equivalent to our output contract.
- `acblaslt-scale-bias`: that GEMM plus one fused elementwise scale/bias/cast kernel.
- `actlize-fused`: existing INT8 GEMM with scale/bias/cast fused into its epilogue.

Only the last two participate in the winner comparison. The separate scale kernel
is a benchmark implementation, **not a claim that no vendor fused path exists**.
All three use identical input bytes, warm allocations, the same current stream,
and preallocated outputs. Activation quantization/ConvRot is untimed. The TN view
is zero-copy: column-major `Y^T[N,M] = W^T[K,N]^T * A^T[K,M]`; no hidden repack.
Before timing, an independent full CPU-int64 non-square test checks descriptor
orientation and epilogue; the target shape checks all output bits against our
admitted config0 and sampled INT32 accumulators against CPU-int64. Negative
controls transpose the output and omit row scales; both must be detected.

Fresh interleaved confirmation compares each backend's measured winner. Overlap
of sample envelopes means UNRESOLVED. The output includes raw samples, latency,
TOPS and utilization using the **user-declared 1000 dense INT8 TOPS** denominator,
not ACU SOL. An epilogue-only timing and an original-public-API control are separate
diagnostics; do not add separate medians to derive the complete path's time.
The full JSON and log are saved under `/workspace/comfy-kitchen-blaslt-...`, with
repo SHA, loaded binary hashes (including the actual Lt library), device/runtime,
input fingerprint and returned algorithm identities. Local compile/tests do not
claim a PPU device or performance result.

Local validation: SDK 2.1.1 real device compilation + `-z defs` multi-library link,
native bridge loading and vendor version query (1400), plus CPU contracts and
negative controls. This SDK's Lt requires `GLIBCXX_3.4.32`; on this development
host loading was checked with an existing isolated newer glibc/libstdc++ runtime,
not by replacing system libraries. The box needs a compatible host C++ runtime.

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
