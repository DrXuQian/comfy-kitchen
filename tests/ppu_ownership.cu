// SPDX-License-Identifier: Apache-2.0
// Runs on the CPU but instantiates the exact shipping PPU types and epilogue.
#include "comfy_kitchen/backends/ppu/int8_config.hpp"
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace cute;
static_assert(int(hggcDevAttrMaxThreadsPerBlock) == 1 &&
              int(hggcDevAttrMaxSharedMemoryPerBlockOptin) == 97,
              "Python device-limit attribute IDs differ from this SDK");
void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

template<class Output, int TM, int TN, int TK, int WM, int WN, int Stages>
void check(int id, bool plant_owner, bool plant_scale, uint64_t& cells) {
  using Config = comfy::ppu::Int8Config<Output, TM, TN, TK, WM, WN, Stages>;
  using Kernel = typename Config::Kernel;
  using Epi = comfy::ppu::Int8Epilogue<Output>;
  typename Kernel::TiledMma mma;
  auto tile = typename Kernel::TileShape{};
  auto coordinates = make_identity_tensor(take<0, 2>(tile));
  for (auto extents : {std::pair<int,int>{1,1}, {TM-1,TN-1}, {TM,TN}, {TM+1,TN+3}}) {
    int m = extents.first, n = extents.second;
    for (bool scalar : {false, true}) for (bool use_bias : {false,true}) {
      std::vector<Output> out(m*n, Output(-117.0f)), bias(n);
      std::vector<float> xs(m), ws(scalar ? 1 : n);
      std::vector<int> visits(m*n, 0);
      for (int r=0; r<m; ++r) xs[r] = float((r%7)+1) / 16;
      for (int c=0; c<int(ws.size()); ++c) ws[c] = float((c%11)+1) / 32;
      for (int c=0; c<n; ++c) bias[c] = Output(float((c%9)-4)/4);
      ComfyPpuInt8Args p{m,n,TK+32,nullptr,nullptr,xs.data(),ws.data(),
        use_bias ? bias.data() : nullptr,out.data(),0,int(scalar),id};
      auto args = Config::arguments(p);
      auto lowered = Kernel::to_underlying_arguments(args, nullptr);
      require(get<0>(lowered.problem_shape)==m && get<1>(lowered.problem_shape)==n &&
              get<2>(lowered.problem_shape)==TK+32 && get<3>(lowered.problem_shape)==1,
              "shape changed during lowering");
      require(get<0>(lowered.mainloop.dA)==TK+32 && get<0>(lowered.mainloop.dB)==TK+32,
              "A/B element pitch changed during lowering");
      auto grid = Kernel::get_grid_shape(lowered);
      require(grid.x==unsigned((m+TM-1)/TM) && grid.y==unsigned((n+TN-1)/TN) && grid.z==1,
              "launch grid changed");
      typename Epi::SharedStorage storage;
      Epi epi(p, storage);
      if (plant_scale) epi.params.scalar_weight_scale = 1;
      for (int bm=0; bm<(m+TM-1)/TM; ++bm) for (int bn=0; bn<(n+TN-1)/TN; ++bn) {
        for (int t=0; t<Kernel::MaxThreadsPerBlock; ++t) {
          int owner = plant_owner && t==1 ? 0 : t;
          auto owned = mma.get_thread_slice(owner).partition_C(coordinates);
          auto accum = partition_fragment_C(mma, take<0,2>(tile));
          for (int i=0; i<size(accum); ++i) {
            int r=bm*TM+get<0>(owned(i)), c=bn*TN+get<1>(owned(i));
            accum(i) = (r*17+c*3)%513-256;
            if (r<m && c<n) ++visits[r*n+c];
          }
          epi(lowered.problem_shape, tile, make_coord(bm,bn,_,0), accum, mma,
              make_tuple(m-bm*TM,n-bn*TN,-32),owner,nullptr);
        }
      }
      for (int r=0; r<m; ++r) for (int c=0; c<n; ++c) {
        require(visits[r*n+c]==1,"output owner missing/duplicate");
        // Independent row-major scalar anchor, not partition_C again.
        float v = float((r*17+c*3)%513-256) * xs[r];
        v = v * ws[scalar?0:c];
        v = v + (use_bias ? float(bias[c]) : 0.0f);
        Output expected(v);
        require(std::memcmp(&out[r*n+c],&expected,sizeof(Output))==0,"scale/bias/output mismatch");
        ++cells;
      }
    }
  }
}

int main(int argc, char** argv) {
  bool owner=argc>1 && std::string(argv[1])=="--plant-owner";
  bool scale=argc>1 && std::string(argv[1])=="--plant-scale";
  uint64_t cells=0;
  int configs=0;
  try {
#define COMFY_PPU_INT8_CONFIG(ID,TM,TN,TK,WM,WN,S) \
    ++configs; \
    check<float,TM,TN,TK,WM,WN,S>(ID,owner,scale,cells); \
    check<cutlass::half_t,TM,TN,TK,WM,WN,S>(ID,owner,scale,cells); \
    check<cutlass::bfloat16_t,TM,TN,TK,WM,WN,S>(ID,owner,scale,cells);
#include COMFY_PPU_INT8_CONFIGS
#undef COMFY_PPU_INT8_CONFIG
    std::cout << "[PPU host] " << configs << " configs x 3 dtypes; cells=" << cells
              << " actual partition_C+epilogue+Params PASS (NOT device MMA validation)\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[PPU host] FAIL: " << e.what() << '\n';
    return 1;
  }
}
