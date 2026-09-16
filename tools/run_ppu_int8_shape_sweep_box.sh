#!/usr/bin/env bash
# Reuse one admitted expanded library. GPU kernels run sequentially; no build.
if [[ ${BASH_SOURCE[0]} != "$0" ]]; then
  printf 'Run with bash, not source.\n' >&2
  return 2
fi
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PPU_SDK=${PPU_SDK:-/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK}
export LD_LIBRARY_PATH="$PPU_SDK/lib:${LD_LIBRARY_PATH:-}"
options=(--samples "${SAMPLES:-3}" --iterations "${ITERATIONS:-3}"
  --confirm-samples "${CONFIRM_SAMPLES:-7}" --confirm-iterations "${CONFIRM_ITERATIONS:-20}"
  --confirm-top "${CONFIRM_TOP:-8}" --peak-tops "${PEAK_TOPS:-1000}")
if [[ -n ${OUT:-} ]]; then options+=(--out "$OUT"); fi
python tools/ppu_int8_shape_suite.py "${options[@]}" "$@"
