// SPDX-License-Identifier: Apache-2.0
// Real SDK load/registration check. No fake driver, launch bypass, or device work.
#include <dlfcn.h>
#include <cstdio>

int main(int argc, char** argv) {
  if (argc != 2 && argc != 3) return 2;
  if (argc == 3 && !dlopen(argv[2], RTLD_NOW | RTLD_GLOBAL)) {
    std::fprintf(stderr, "preload: %s\n", dlerror());
    return 1;
  }
  // Match runtime.py: this C-only extension binds its native HGGC dependency,
  // even if a torch-loaded global wrapper exports the same unversioned names.
  void* lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
  if (!lib) {
    std::fprintf(stderr, "load: %s\n", dlerror());
    return 1;
  }
  auto abi = reinterpret_cast<int (*)()>(dlsym(lib, "comfy_ppu_abi_version"));
  auto configs = reinterpret_cast<int (*)()>(dlsym(lib, "comfy_ppu_int8_config_count"));
  if (!abi || !configs || abi() != 1 || configs() != 6) return 1;
  std::puts("[PPU load] real SDK registration + ABI=1 configs=6 PASS; no device execution");
  dlclose(lib);
  return 0;
}
