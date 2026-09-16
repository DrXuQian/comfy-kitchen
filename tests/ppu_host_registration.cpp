// SPDX-License-Identifier: Apache-2.0
// CPU layout/epilogue test ONLY. hgcc emits registration for unused device
// constants even in a host-only test. No launch/runtime/math function is stubbed.
// The production shared library links the real SDK and never uses this file.
extern "C" {
void** __hggcRegisterFatBinary(void*) { static void* handle; return &handle; }
void __hggcUnregisterFatBinary(void**) {}
void __hggcRegisterVar(void**, ...) {}
}
