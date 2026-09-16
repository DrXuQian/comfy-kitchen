# comfy-kitchen PPU wheel

Source main: `6d8adcb` (selector implementation `eca3955`). This independent
artifact branch stores the approximately 1 MB wheel as an ordinary Git blob;
**Git LFS is not required**. The source branch contains no wheel payload.

The SDK-built product library has seven configs and the native shape-family
selector. The separately built 285-config sweep library keeps its original IDs.

```bash
git clone --single-branch --branch ppu-wheels \
  https://github.com/DrXuQian/comfy-kitchen.git /workspace/ppu-wheel-comfy
bash /workspace/ppu-wheel-comfy/install.sh
export LD_LIBRARY_PATH=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK/lib:${LD_LIBRARY_PATH:-}
unset COMFY_KITCHEN_PPU_LIBRARY
(cd /workspace && python -c 'import comfy_kitchen; from comfy_kitchen.backends import ppu; print(comfy_kitchen.__file__); print(ppu.select_config(73774,21504,5376)); print(ppu.configurations())')
```

Change the SDK path if needed. Keep the existing PPU Torch (`--no-deps` is
intentional). Test outside the source checkout so old in-place files cannot
shadow the wheel. There is no box compilation. The native C ABI does not
exchange ATen/C++ objects; the handoff environment is Python 3.12 / Torch 2.9.

Large BF16 prefill uses `64x256x128_w64x32_s2` below K8192 and
`128x256x128_w64x64_s2` at/above K8192. Small/compact shapes and other dtypes keep
their conservative choices. Explicit coordinates/config IDs override auto.
The complete policy lives in the source `int8_selector.hpp`.

The eight-shape device experiment supports a better fallback, not a unique
global optimum: all 16 unique-winner comparisons remain UNRESOLVED. Real H3's
four main N/K pairs were not in that synthetic sweep. `--suite minimax-h3` on
the source branch runs those pairs using the existing expanded binary.

`install.sh` verifies the wheel digest before installation. `release.json` and
the embedded `.so.json` bind wheel/native/source/actlize hashes and SDK identity.
Local compile/link, ownership, policy and installation checks passed; a device
run of this newly assembled wheel is not claimed. SDK runtime libraries are
required and are not bundled. GEMM math, quantization and epilogue are unchanged.
