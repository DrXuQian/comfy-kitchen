// SPDX-License-Identifier: Apache-2.0
// Execute the shipping policy on CPU; no parallel Python implementation.
#include "int8_selector.hpp"
#include <cstring>
#include <iostream>

int main(int argc, char** argv) {
  using namespace comfy::ppu;
  bool old_default = argc > 1 && std::strcmp(argv[1], "--plant-old-default") == 0;
  int checked = 0;
  for (int64_t m : {0, 1, 127, 128, 4095, 4096, 16384, 73774, 77777})
    for (int64_t n : {0, 1, 255, 256, 4095, 4096, 7168, 8192, 16384})
      for (int64_t k : {32, 128, 4064, 4096, 6144, 8160, 8192, 12288, 16384})
        for (int dtype : {0, 1, 2}) {
          int selected = old_default ? (m < 128 ? 0 : 1) : default_int8_config(m, n, k, dtype);
          const Int8Geometry* got = nullptr;
          for (auto& g : kInt8Geometries) if (g.id == selected) got = &g;
          if (!got) return 2;
          // Check the selected geometry, not an ID shared with the selector.
          int tm=128, tn=128, tk=64, wm=64, wn=64, stages=3;
          if (m < 128) { tm=64; wm=32; }
          if (dtype == 2 && m >= 4096 && n >= 4096 && k >= 4096) {
            tn=256; tk=128; stages=2;
            if (k < 8192) { tm=64; wn=32; }
          }
          if (got->tm != tm || got->tn != tn || got->tk != tk ||
              got->wm != wm || got->wn != wn || got->stages != stages) {
            std::cerr << "selector mismatch " << m << 'x' << n << 'x' << k
                      << " dtype=" << dtype << " id=" << selected << '\n';
            return 1;
          }
          ++checked;
        }
  std::cout << "selector CPU PASS cases=" << checked
            << " balanced_id=" << kInt8Prefill << " long_k_id=" << kInt8LongK << '\n';
}
