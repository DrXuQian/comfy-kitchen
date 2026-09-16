// ID, TileM, TileN, TileK, WarpM, WarpN, Stages
// Shared by the launcher, resource query and host ownership proof.
COMFY_PPU_INT8_CONFIG(0, 64, 128, 64, 32, 64, 3)
COMFY_PPU_INT8_CONFIG(1, 128, 128, 64, 64, 64, 3)
COMFY_PPU_INT8_CONFIG(2, 128, 256, 64, 64, 64, 3)
COMFY_PPU_INT8_CONFIG(3, 64, 128, 128, 32, 64, 3)
COMFY_PPU_INT8_CONFIG(4, 128, 128, 128, 64, 64, 3)
COMFY_PPU_INT8_CONFIG(5, 128, 256, 128, 64, 64, 2)
// Measured balanced prefill fallback. Product ID 6; expanded sweep ID 94.
COMFY_PPU_INT8_CONFIG(6, 64, 256, 128, 64, 32, 2)
