// SPDX-License-Identifier: Apache-2.0
// Reuse the actual upstream quantizer, including its input-dtype rounding and
// regular Hadamard/ConvRot convention. No second copy of the quantization math.
#define COMFY_KITCHEN_PPU 1
#include "../cuda/ops/int8_linear.cu"
#include "api.h"

extern "C" void comfy_ppu_set_error(const char*);
extern "C" int comfy_ppu_quantize_int8(const void* x, int8_t* q, float* scales,
    int64_t m, int64_t k, int dtype, int convrot, int group_size,
    int stochastic, uint64_t seed, void* stream) {
  try {
    if (!x || !q || !scales || m < 0 || m > INT32_MAX || k <= 0 || k > 131040)
      throw std::invalid_argument("invalid row quantization operands/extents");
    auto s = reinterpret_cast<cudaStream_t>(stream);
    if (convrot) {
      if (m > 8 && k <= 16384) {
        launch_quantize_int8_rowwise_convrot64_kernel(
            x, q, scales, m, k, group_size, dtype, stochastic != 0,
            comfy::kActNone, seed, nullptr, 0.0f, s);
      } else {
        launch_quantize_int8_rowwise_convrot_kernel(
            x, q, scales, m, k, group_size, dtype, stochastic != 0, seed, s);
      }
    } else {
      launch_quantize_int8_rowwise_kernel(x, q, scales, m, k, dtype, stochastic != 0, seed, s);
    }
    return 0;
  } catch (const std::exception& e) { comfy_ppu_set_error(e.what()); return 1; }
}
