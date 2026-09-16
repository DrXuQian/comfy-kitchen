// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

#ifndef COMFY_PPU_INT8_CONFIGS
#define COMFY_PPU_INT8_CONFIGS "int8_configs.inc"
#endif

namespace comfy::ppu {
struct Int8Geometry {
  int id, tm, tn, tk, wm, wn, stages;
};
inline constexpr Int8Geometry kInt8Geometries[] = {
#define COMFY_PPU_INT8_CONFIG(ID, TM, TN, TK, WM, WN, S) {ID, TM, TN, TK, WM, WN, S},
#include COMFY_PPU_INT8_CONFIGS
#undef COMFY_PPU_INT8_CONFIG
};

constexpr int int8_config_id(int tm, int tn, int tk, int wm, int wn, int stages) {
  for (auto c : kInt8Geometries)
    if (c.tm == tm && c.tn == tn && c.tk == tk && c.wm == wm &&
        c.wn == wn && c.stages == stages) return c.id;
  return -1;
}

// Resolve coordinates in the compiled table, never reuse a sweep-local ID.
inline constexpr int kInt8SmallM = int8_config_id(64, 128, 64, 32, 64, 3);
inline constexpr int kInt8Compact = int8_config_id(128, 128, 64, 64, 64, 3);
inline constexpr int kInt8Prefill = int8_config_id(64, 256, 128, 64, 32, 2);
inline constexpr int kInt8LongK = int8_config_id(128, 256, 128, 64, 64, 2);
static_assert(kInt8SmallM >= 0 && kInt8Compact >= 0 &&
              kInt8Prefill >= 0 && kInt8LongK >= 0,
              "PPU INT8 selector references a config absent from the compiled table");

// This is a bounded heuristic, not a lookup of eight measured shapes or a
// claim of a unique winner. Calibration: BF16 on PPU-ZW810, 72 CUs. Untested
// shapes inside the family use the same fallback; no implicit device sweep.
constexpr int default_int8_config(int64_t m, int64_t n, int64_t k, int dtype) {
  if (m < 128) return kInt8SmallM;
  if (dtype == 2 && m >= 4096 && n >= 4096 && k >= 4096)
    return k >= 8192 ? kInt8LongK : kInt8Prefill;
  // Small/compact problems and other output dtypes lack performance evidence
  // for the large-BF16-family policy, so retain the previous safe selection.
  return kInt8Compact;
}
}  // namespace comfy::ppu
