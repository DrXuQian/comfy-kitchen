// SPDX-License-Identifier: Apache-2.0
#include "int8_dispatch.hpp"

#include <limits>
#include <stdexcept>
#include <string>

#ifdef COMFY_PPU_INT8_SHARDED
#include "int8_shard_dispatch.inc"
#endif

namespace {
thread_local std::string last_error;

#ifndef COMFY_PPU_INT8_SHARDED
template<class Output>
int dispatch(int id, const ComfyPpuInt8Args* args, void* stream,
             int* threads = nullptr, int* smem = nullptr, int* occupancy = nullptr) {
  switch (id) {
#define COMFY_PPU_INT8_CONFIG(ID, TM, TN, TK, WM, WN, STAGES) \
    case ID: return comfy::ppu::run_int8_config<Output, TM, TN, TK, WM, WN, STAGES>( \
        args, stream, threads, smem, occupancy);
#include COMFY_PPU_INT8_CONFIGS
#undef COMFY_PPU_INT8_CONFIG
    default: throw std::invalid_argument("unknown PPU INT8 config");
  }
}
#endif

int dispatch_dtype(int dtype, int config, const ComfyPpuInt8Args* args, void* stream,
                   int* threads = nullptr, int* smem = nullptr, int* occupancy = nullptr) {
#ifdef COMFY_PPU_INT8_SHARDED
  if (config < 0 || config >= comfy_ppu_int8_config_count())
    throw std::invalid_argument("unknown PPU INT8 config");
  return comfy_ppu_int8_shards[config / COMFY_PPU_INT8_SHARD_ROWS](
      dtype, config, args, stream, threads, smem, occupancy);
#else
  switch (dtype) {
    case 0: return dispatch<float>(config, args, stream, threads, smem, occupancy);
    case 1: return dispatch<cutlass::half_t>(config, args, stream, threads, smem, occupancy);
    case 2: return dispatch<cutlass::bfloat16_t>(config, args, stream, threads, smem, occupancy);
    default: throw std::invalid_argument("PPU INT8 output dtype must be fp32/fp16/bf16");
  }
#endif
}
}  // namespace

extern "C" int comfy_ppu_abi_version() { return 1; }
extern "C" const char* comfy_ppu_last_error() { return last_error.c_str(); }
extern "C" void comfy_ppu_set_error(const char* text) { last_error = text; }
extern "C" int comfy_ppu_int8_config_count() {
  int count = 0;
#define COMFY_PPU_INT8_CONFIG(...) ++count;
#include COMFY_PPU_INT8_CONFIGS
#undef COMFY_PPU_INT8_CONFIG
  return count;
}
extern "C" const char* comfy_ppu_int8_config_name(int id) {
  switch (id) {
#define COMFY_PPU_INT8_CONFIG(ID, TM, TN, TK, WM, WN, STAGES) \
    case ID: return #TM "x" #TN "x" #TK "_w" #WM "x" #WN "_s" #STAGES;
#include COMFY_PPU_INT8_CONFIGS
#undef COMFY_PPU_INT8_CONFIG
    default: return nullptr;
  }
}
extern "C" int comfy_ppu_int8_gemm(const ComfyPpuInt8Args* a, void* stream) {
  try {
    if (!a || a->m < 0 || a->n < 0 || a->k <= 0 || a->k % 32 ||
        a->m > INT32_MAX || a->n > INT32_MAX || a->k > 131040 ||
        (a->scalar_weight_scale != 0 && a->scalar_weight_scale != 1))
      throw std::invalid_argument("invalid INT8 extents: K must be a multiple of 32 <= 131040 (INT32 sum bound)");
    if (a->config < -1 || a->config >= comfy_ppu_int8_config_count())
      throw std::invalid_argument("unknown PPU INT8 config");
    if (a->dtype < 0 || a->dtype > 2) throw std::invalid_argument("invalid output dtype");
    if (a->m == 0 || a->n == 0) return 0;
    if (!a->a || !a->b || !a->x_scale || !a->w_scale || !a->output ||
        reinterpret_cast<uintptr_t>(a->a) % 16 || reinterpret_cast<uintptr_t>(a->b) % 16)
      throw std::invalid_argument("missing/unaligned INT8 operand");
    // Conservative starting point, NOT a measured PPU winner. Explicit IDs allow
    // device sweep without changing quantization, storage or operation order.
    const int id = a->config < 0 ? (a->m < 128 ? 0 : 1) : a->config;
    if (cute::ceil_div(a->n, int64_t(128)) > 65535)
      throw std::invalid_argument("N exceeds the supported grid.y extent");
    return dispatch_dtype(a->dtype, id, a, stream);
  } catch (const std::exception& e) { last_error = e.what(); return 1; }
}
extern "C" int comfy_ppu_int8_resources(int config, int dtype, int* threads, int* smem, int* occ) {
  try { return dispatch_dtype(dtype, config, nullptr, nullptr, threads, smem, occ); }
  catch (const std::exception& e) { last_error = e.what(); return 1; }
}
