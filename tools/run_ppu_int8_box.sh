#!/usr/bin/env bash
# Run as "bash tools/run_ppu_int8_box.sh" (not sourced). No parent-shell exit.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PPU_SDK=${PPU_SDK:-/usr/local/PPU_SDK}
export PPU_SDK
export LD_LIBRARY_PATH="$PPU_SDK/lib:${LD_LIBRARY_PATH:-}"
run_sha=$(git rev-parse --short=8 HEAD)
run_time=$(date -u +%Y%m%dT%H%M%SZ)
OUT=${OUT:-/workspace/comfy-kitchen-ppu-${run_sha}-${run_time}-$$}
mkdir -p "$OUT"
printf '[PPU INT8 runner] sha=%s out=%s\n' "$run_sha" "$OUT"
git rev-parse HEAD > "$OUT/sha.txt"
git status --short > "$OUT/git-status.txt"
if [[ ${BUILD:-1} == 1 ]]; then
  build_extra=(--jobs "${JOBS:-1}")
  if [[ ${EXTENDED:-0} == 1 ]]; then build_extra+=(--extended); fi
  python tools/build_ppu.py --sdk "$PPU_SDK" --build-dir "$OUT/build" \
    --output "$OUT/_native.so" "${build_extra[@]}" 2>&1 | tee "$OUT/build.log"
  export COMFY_KITCHEN_PPU_LIBRARY="$OUT/_native.so"
fi
python tools/ppu_int8_check.py 2>&1 | tee "$OUT/correctness.log"
if [[ ${SWEEP:-0} == 1 ]]; then
  extra=()
  if [[ ${CONVROT:-1} == 1 ]]; then extra+=(--convrot); fi
  if [[ -n ${PEAK_TOPS:-} ]]; then extra+=(--peak-tops "$PEAK_TOPS"); fi
  python tools/ppu_int8_sweep.py --m "${M:-4096}" --n "${N:-4096}" --k "${K:-4096}" \
    --dtype "${DTYPE:-bf16}" --samples "${SAMPLES:-7}" --iterations "${ITERATIONS:-20}" \
    --confirm-top "${CONFIRM_TOP:-0}" \
    --out "$OUT" "${extra[@]}" 2>&1 | tee "$OUT/sweep.log"
fi
printf '[PPU INT8 runner] PASS; artifacts: %s\n' "$OUT"
