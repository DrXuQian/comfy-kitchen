// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <cstddef>

// Torch-independent ABI. All sizes/strides are elements, pointers are device
// pointers, and stream is the caller's current PPU/CUDA-compatible stream.
struct ComfyPpuInt8Args {
  int64_t m, n, k;
  const int8_t* a;                 // contiguous [M,K]
  const int8_t* b;                 // contiguous [N,K]
  const float* x_scale;            // [M]
  const float* w_scale;            // [1] or [N]
  const void* bias;                // optional [N], output dtype
  void* output;                    // contiguous [M,N]
  int32_t dtype;                  // 0=float32, 1=float16, 2=bfloat16
  int32_t scalar_weight_scale;
  int32_t config;                 // -1=default, otherwise explicit table ID
};
static_assert(sizeof(ComfyPpuInt8Args) == 88, "Python/C++ INT8 ABI size drift");
static_assert(offsetof(ComfyPpuInt8Args, output) == 64, "INT8 ABI pointer offset drift");
static_assert(offsetof(ComfyPpuInt8Args, dtype) == 72, "INT8 ABI dtype offset drift");

extern "C" {
int comfy_ppu_abi_version();
const char* comfy_ppu_last_error();
int comfy_ppu_int8_gemm(const ComfyPpuInt8Args*, void* stream);
int comfy_ppu_int8_config_count();
const char* comfy_ppu_int8_config_name(int config);
// Host-only policy query; returns -1 with last_error on invalid arguments.
int comfy_ppu_int8_select_config(int64_t m, int64_t n, int64_t k, int dtype);
int comfy_ppu_int8_resources(int config, int dtype, int* threads, int* smem, int* occupancy);
int comfy_ppu_quantize_int8(const void* x, int8_t* q, float* scales,
                          int64_t m, int64_t k, int dtype, int convrot,
                          int group_size, int stochastic, uint64_t seed,
                          void* stream);
}
