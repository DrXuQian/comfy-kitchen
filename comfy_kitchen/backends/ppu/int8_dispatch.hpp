// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "int8_config.hpp"
#include <stdexcept>
#include <string>

namespace comfy::ppu {
// One launcher for both the small product table and sharded sweep TUs. Moving
// host dispatch into shards does not change the GEMM/mainloop/epilogue type.
template<class Output, int TM, int TN, int TK, int WM, int WN, int Stages>
int run_int8_config(const ComfyPpuInt8Args* a, void* stream,
                    int* threads, int* smem, int* occupancy) {
  using Config = Int8Config<Output, TM, TN, TK, WM, WN, Stages>;
  using Kernel = typename Config::Kernel;
  using Gemm = typename Config::Gemm;
  static_assert(Kernel::MaxThreadsPerBlock == 32 * (TM / WM) * (TN / WN),
                "enumerated CTA threads differ from the actual kernel");
  static_assert(Kernel::SharedStorageSize == (TM + TN) * TK * Stages,
                "enumerated shared bytes differ from the actual kernel");
  if (a) {
    if (cute::ceil_div(a->n, int64_t(TN)) > 65535)
      throw std::invalid_argument("N exceeds this config's grid.y extent");
    auto args = Config::arguments(*a);
    auto status = Gemm::can_implement(args);
    if (status != cutlass::Status::kSuccess)
      throw std::runtime_error("actlize cannot implement this INT8 shape/layout");
    Gemm op;
    status = op(args, nullptr, reinterpret_cast<hggcStream_t>(stream));
    if (status != cutlass::Status::kSuccess)
      throw std::runtime_error("actlize INT8 GEMM launch failed: " + std::to_string(int(status)));
  }
  if (threads) *threads = Kernel::MaxThreadsPerBlock;
  if (smem) *smem = Kernel::SharedStorageSize;
  if (occupancy) *occupancy = Gemm::maximum_active_blocks();
  return 0;
}
}  // namespace comfy::ppu
