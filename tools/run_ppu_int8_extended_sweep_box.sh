#!/usr/bin/env bash
# Expanded config competition only; no ACU and no vendor rerun.
if [[ ${BASH_SOURCE[0]} != "$0" ]]; then
  printf 'Run with bash, not source.\n' >&2
  return 2
fi
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PPU_SDK=${PPU_SDK:-/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK}
export BUILD=1 EXTENDED=1 SWEEP=1
export JOBS=${JOBS:-8} CONFIRM_TOP=${CONFIRM_TOP:-8} PEAK_TOPS=${PEAK_TOPS:-1000}
export M=${M:-4096} N=${N:-4096} K=${K:-4096} DTYPE=${DTYPE:-bf16}
export CONVROT=${CONVROT:-1} SAMPLES=${SAMPLES:-7} ITERATIONS=${ITERATIONS:-20}
bash tools/run_ppu_int8_box.sh
