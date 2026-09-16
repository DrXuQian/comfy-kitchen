// SPDX-License-Identifier: Apache-2.0
// Benchmark-only acBLASLt bridge. No production dispatch changes.
#include <hggc_runtime.h>
#include <hggc_fp16.h>
#include <hggc_bf16.h>
#include <acblasLt.h>
#include "../comfy_kitchen/backends/ppu/api.h"

#include <dlfcn.h>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
thread_local std::string error_text;
thread_local std::string info_text;

void check(acblasStatus_t status, const char* where) {
  if (status != ACBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string(where) + ": acBLASLt status=" + std::to_string(int(status)));
}

struct Plan {
  acblasLtHandle_t handle = nullptr;
  acblasLtMatmulDesc_t operation = nullptr;
  acblasLtMatrixLayout_t weight = nullptr, activation = nullptr, output = nullptr;
  acblasLtMatmulPreference_t preference = nullptr;
  std::vector<acblasLtMatmulHeuristicResult_t> algorithms;
  acblasStatus_t heuristic_status = ACBLAS_STATUS_SUCCESS;
  int requested = 0;
  int64_t m = 0, n = 0, k = 0;
  size_t workspace_bytes = 0;
  ~Plan() {
    if (preference) acblasLtMatmulPreferenceDestroy(preference);
    if (output) acblasLtMatrixLayoutDestroy(output);
    if (activation) acblasLtMatrixLayoutDestroy(activation);
    if (weight) acblasLtMatrixLayoutDestroy(weight);
    if (operation) acblasLtMatmulDescDestroy(operation);
    if (handle) acblasLtDestroy(handle);
  }
};

template<class Output>
__global__ void scale_bias(const int32_t* input, ComfyPpuInt8Args a) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < a.m * a.n) {
    const int64_t row = i / a.n, col = i % a.n;
    // Same order as Int8Epilogue; compiled with -fmad=false. In particular,
    // do not precombine the two scales or round before adding bias.
    float value = float(input[i]) * a.x_scale[row];
    value = value * a.w_scale[a.scalar_weight_scale ? 0 : col];
    value = value + (a.bias ? float(static_cast<const Output*>(a.bias)[col]) : 0.0f);
    static_cast<Output*>(a.output)[i] = Output(value);
  }
}
}  // namespace

extern "C" const char* comfy_lt_error() { return error_text.c_str(); }
extern "C" size_t comfy_lt_version() { return acblasLtGetVersion(); }
extern "C" const char* comfy_lt_library_path() {
  Dl_info info{};
  return dladdr(reinterpret_cast<void*>(&acblasLtMatmul), &info) ? info.dli_fname : nullptr;
}

extern "C" int comfy_lt_create(int64_t m, int64_t n, int64_t k,
                                size_t workspace_bytes, int requested, void** result) {
  if (result) *result = nullptr;
  try {
    if (!result || m <= 0 || n <= 0 || k <= 0 || k > 131040 ||
        m > INT32_MAX || n > INT32_MAX || requested < 1 || requested > 256)
      throw std::invalid_argument("invalid Lt benchmark plan extents/candidate count");
    auto p = std::make_unique<Plan>();
    p->m = m; p->n = n; p->k = k;
    p->workspace_bytes = workspace_bytes;
    p->requested = requested;
    check(acblasLtCreate(&p->handle), "create handle");
    check(acblasLtMatmulDescCreate(&p->operation, ACBLAS_COMPUTE_32I, HGGC_R_32I), "INT32 compute");
    const acblasOperation_t trans_a = ACBLAS_OP_T, trans_b = ACBLAS_OP_N;
    check(acblasLtMatmulDescSetAttribute(p->operation, ACBLASLT_MATMUL_DESC_TRANSA,
                                       &trans_a, sizeof(trans_a)), "trans A");
    check(acblasLtMatmulDescSetAttribute(p->operation, ACBLASLT_MATMUL_DESC_TRANSB,
                                       &trans_b, sizeof(trans_b)), "trans B");
    // Zero-copy TN column-major view:
    // W[N,K]_row == Wt[K,N]_col; A[M,K]_row == At[K,M]_col.
    // Lt computes Wt^T * At = Yt[N,M]_col == Y[M,N]_row.
    check(acblasLtMatrixLayoutCreate(&p->weight, HGGC_R_8I, k, n, k), "W layout");
    check(acblasLtMatrixLayoutCreate(&p->activation, HGGC_R_8I, k, m, k), "A layout");
    check(acblasLtMatrixLayoutCreate(&p->output, HGGC_R_32I, n, m, n), "Y layout");
    check(acblasLtMatmulPreferenceCreate(&p->preference), "preference");
    check(acblasLtMatmulPreferenceSetAttribute(p->preference,
        ACBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes, sizeof(workspace_bytes)), "workspace limit");
    p->algorithms.resize(requested);
    int returned = 0;
    p->heuristic_status = acblasLtMatmulAlgoGetHeuristic(p->handle, p->operation,
        p->weight, p->activation, p->output, p->output, p->preference,
        requested, p->algorithms.data(), &returned);
    if (p->heuristic_status != ACBLAS_STATUS_SUCCESS &&
        p->heuristic_status != ACBLAS_STATUS_NOT_SUPPORTED)
      check(p->heuristic_status, "heuristic query");
    if (p->heuristic_status != ACBLAS_STATUS_SUCCESS) returned = 0;
    if (returned < 0 || returned > requested) throw std::runtime_error("invalid heuristic count");
    p->algorithms.resize(returned);
    *result = p.release();
    return 0;
  } catch (const std::exception& e) { error_text = e.what(); return 1; }
}

extern "C" void comfy_lt_destroy(void* ptr) { delete static_cast<Plan*>(ptr); }
extern "C" int comfy_lt_count(void* ptr) {
  // Slot zero is the library's NULL-algorithm/default choice, not a fake heuristic.
  return ptr ? 1 + int(static_cast<Plan*>(ptr)->algorithms.size()) : 0;
}
extern "C" const char* comfy_lt_info(void* ptr, int slot) {
  if (!ptr || slot < 0 || slot >= comfy_lt_count(ptr)) return nullptr;
  auto& p = *static_cast<Plan*>(ptr);
  std::ostringstream s;
  s << "{\"slot\":" << slot << ",\"selection\":\""
    << (slot ? "heuristic" : "library-default") << "\",\"requested\":" << p.requested
    << ",\"returned\":" << p.algorithms.size()
    << ",\"heuristic_status\":" << int(p.heuristic_status);
  if (slot) {
    const auto& a = p.algorithms[slot - 1];
    s << ",\"state\":" << int(a.state) << ",\"workspace_bytes\":" << a.workspaceSize
      << ",\"algo_opaque_u64\":[";
    for (int i = 0; i < 8; ++i) s << (i ? "," : "") << a.algo.data[i];
    s << "]";
  }
  s << "}";
  info_text = s.str();
  return info_text.c_str();
}

// mode 0 = raw INT32 GEMM, 1 = GEMM + identical scale/bias/cast, 2 = epilogue only.
// Return 2 only for a vendor-reported unsupported algorithm; numerical errors
// and all other runtime failures remain FAIL in the caller.
extern "C" int comfy_lt_run(void* ptr, int slot, const ComfyPpuInt8Args* a,
    int32_t* accum, void* workspace, void* stream_ptr, int mode) {
  try {
    if (!ptr || !a || !accum || mode < 0 || mode > 2 || slot < 0 || slot >= comfy_lt_count(ptr))
      throw std::invalid_argument("invalid Lt run arguments");
    auto& p = *static_cast<Plan*>(ptr);
    if (a->m != p.m || a->n != p.n || a->k != p.k || a->dtype < 0 || a->dtype > 2 ||
        !a->a || !a->b || !a->output || !a->x_scale || !a->w_scale ||
        (p.workspace_bytes && !workspace)) throw std::invalid_argument("Lt plan/pointer mismatch");
    const auto stream = reinterpret_cast<hggcStream_t>(stream_ptr);
    if (mode != 2) {
      const acblasLtMatmulAlgo_t* algo = nullptr;
      if (slot) {
        const auto& choice = p.algorithms[slot - 1];
        if (choice.state == ACBLAS_STATUS_NOT_SUPPORTED || choice.workspaceSize > p.workspace_bytes) {
          error_text = "heuristic unsupported or exceeds declared workspace"; return 2;
        }
        check(choice.state, "heuristic state");
        algo = &choice.algo;
      }
      const int32_t alpha = 1, beta = 0;
      const auto status = acblasLtMatmul(p.handle, p.operation, &alpha,
          a->b, p.weight, a->a, p.activation, &beta, accum, p.output,
          accum, p.output, algo, workspace, p.workspace_bytes, stream);
      if (status == ACBLAS_STATUS_NOT_SUPPORTED) {
        error_text = "acblasLtMatmul: NOT_SUPPORTED (no dtype/layout fallback)"; return 2;
      }
      check(status, "acblasLtMatmul");
    }
    if (mode != 0) {
      const auto blocks = (a->m * a->n + 255) / 256;
      if (blocks > INT32_MAX) throw std::invalid_argument("epilogue grid overflow");
      switch (a->dtype) {
        case 0: scale_bias<float><<<blocks, 256, 0, stream>>>(accum, *a); break;
        case 1: scale_bias<__half><<<blocks, 256, 0, stream>>>(accum, *a); break;
        case 2: scale_bias<__ppu_bfloat16><<<blocks, 256, 0, stream>>>(accum, *a); break;
      }
      const auto status = hggcGetLastError();
      if (status != hggcSuccess) throw std::runtime_error(hggcGetErrorString(status));
    }
    return 0;
  } catch (const std::exception& e) { error_text = e.what(); return 1; }
}
