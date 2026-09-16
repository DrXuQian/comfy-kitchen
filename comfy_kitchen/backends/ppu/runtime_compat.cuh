// SPDX-License-Identifier: Apache-2.0
#pragma once
// Names only: compile upstream SIMT quantization against the native SDK. Do not
// mix CUDA_SDK and native HGGC runtime headers/types in the same translation unit.
#include <hggc_runtime.h>
#include <hggc_fp16.h>
#include <hggc_bf16.h>
#include <hggc_fp8.h>
using nv_bfloat16 = __ppu_bfloat16;
using nv_bfloat162 = __ppu_bfloat162;
using __nv_fp8_e4m3 = __hg_fp8_e4m3;
using __nv_fp8_e5m2 = __hg_fp8_e5m2;
#define cudaStream_t hggcStream_t
#define cudaError_t hggcError_t
#define cudaSuccess hggcSuccess
#define cudaFuncSetAttribute hggcFuncSetAttribute
#define cudaFuncAttributeMaxDynamicSharedMemorySize hggcFuncAttributeMaxDynamicSharedMemorySize
#define cudaGetLastError hggcGetLastError
#define cudaGetErrorString hggcGetErrorString
#define cudaMemcpyToSymbol hggcMemcpyToSymbol
#define cudaGetSymbolAddress hggcGetSymbolAddress
#define cudaDataType_t hggcDataType_t
#define CUDA_R_32F HGGC_R_32F
#define CUDA_R_16F HGGC_R_16F
#define CUDA_R_16BF HGGC_R_16BF
