// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "api.h"
#include "cute/tensor.hpp"
#include "cutlass/numeric_conversion.h"
#include "cutlass/ppu_host_adapter.hpp"

namespace comfy::ppu {

// Preserve the CUDA EVT's operation order, including rounding only after bias:
// cast_out((float(acc) * xs[row]) * ws[col] + bias[col]).
// The MMA's *actual* partition_C supplies ownership; no hand-maintained lane map.
template<class Output>
struct Int8Epilogue {
  using ElementC = Output;
  using ElementD = Output;
  using ElementOutput = Output;
  using ElementAccumulator = int32_t;
  using ElementCompute = float;
  struct ThreadEpilogueOp {
    static constexpr int kCount = 1;
  };
  using StrideC = cute::Stride<int64_t, cute::_1, int64_t>;
  using StrideD = StrideC;
  using GmemTiledCopyC = void;
  using GmemTiledCopyD = void;
  static constexpr int kOutputAlignment = 1;
  struct SharedStorage {};
  using TensorStorage = SharedStorage;
  using Arguments = ComfyPpuInt8Args;
  using Params = Arguments;
  Params params;

  template<class Shape>
  static Params to_underlying_arguments(const Shape&, const Arguments& args, void*) {
    return args;
  }
  template<class Shape>
  static size_t get_workspace_size(const Shape&, const Arguments&) { return 0; }
  template<class Shape>
  static cutlass::Status initialize_workspace(const Shape&, const Arguments&, void*,
      hggcStream_t, cutlass::HostAdapter* = nullptr) { return cutlass::Status::kSuccess; }
  template<class Shape>
  static bool can_implement(const Shape&, const Arguments&) { return true; }
  CUTLASS_HOST_DEVICE
  Int8Epilogue(const Params& args, const SharedStorage&) : params(args) {}
  CUTLASS_DEVICE bool is_source_needed() const { return false; }

  template<class Shape, class Tile, class Coord, class Engine, class Layout,
           class Mma, class Residue>
  CUTLASS_HOST_DEVICE void operator()(Shape, Tile tile, Coord block,
      const cute::Tensor<Engine, Layout>& accum, Mma mma, Residue residue,
      int thread, char*) const {
    using namespace cute;
    auto coordinates = make_identity_tensor(take<0, 2>(tile));
    auto owned = mma.get_thread_slice(thread).partition_C(coordinates);
    CUTE_STATIC_ASSERT_V(size(owned) == size(accum));
    auto* output = static_cast<Output*>(params.output);
    auto* bias = static_cast<const Output*>(params.bias);
    cutlass::NumericConverter<Output, float> convert;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(accum); ++i) {
      const int row = get<0>(owned(i));
      const int col = get<1>(owned(i));
      if (row < get<0>(residue) && col < get<1>(residue)) {
        const int64_t m = int64_t(get<0>(block)) * size<0>(tile) + row;
        const int64_t n = int64_t(get<1>(block)) * size<1>(tile) + col;
        float value = float(accum(i)) * params.x_scale[m];
        value = value * params.w_scale[params.scalar_weight_scale ? 0 : n];
        value = value + (bias ? float(bias[n]) : 0.0f);
        output[m * params.n + n] = convert(value);
      }
    }
  }
};

}  // namespace comfy::ppu
