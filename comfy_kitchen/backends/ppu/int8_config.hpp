// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ppu_include.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "int8_epilogue.hpp"

// Product builds keep the admitted six rows. The explicit sweep build supplies
// one generated table, shared by dispatch, names, counts and host ownership.
#ifndef COMFY_PPU_INT8_CONFIGS
#define COMFY_PPU_INT8_CONFIGS "int8_configs.inc"
#endif

namespace comfy::ppu {
// Single type authority shared with the host ownership/epilogue proof.
template<class Output, int TM, int TN, int TK, int WM, int WN, int Stages>
struct Int8Config {
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::PPU0010, cutlass::arch::OpClassTensorOp,
      int8_t, cutlass::layout::RowMajor, 16,
      int8_t, cutlass::layout::ColumnMajor, 16, int32_t,
      cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::Int<TK>>,
      cute::Shape<cute::Int<WM>, cute::Int<WN>, cute::Int<TK>>,
      cute::Int<Stages>, cutlass::gemm::KernelMultistage>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>, Mainloop, Int8Epilogue<Output>>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
  static typename Kernel::Arguments arguments(const ComfyPpuInt8Args& a) {
    return {cutlass::gemm::GemmUniversalMode::kGemm,
        {int(a.m), int(a.n), int(a.k), 1},
        {a.a, {a.k, cute::_1{}, int64_t(0)},
         a.b, {a.k, cute::_1{}, int64_t(0)}}, a};
  }
};
}  // namespace comfy::ppu
