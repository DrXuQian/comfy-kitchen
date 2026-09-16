#!/usr/bin/env bash
# Execute with bash, not source: errors never exit the user's interactive shell.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PPU_SDK=${PPU_SDK:-/usr/local/PPU_SDK}
export LD_LIBRARY_PATH="$PPU_SDK/lib:${LD_LIBRARY_PATH:-}"
run_sha=$(git rev-parse --short=8 HEAD)
run_time=$(date -u +%Y%m%dT%H%M%SZ)
OUT=${OUT:-/workspace/comfy-kitchen-blaslt-${run_sha}-${run_time}-$$}
mkdir -p "$OUT"
printf '[PPU Lt runner] sha=%s out=%s\n' "$run_sha" "$OUT"
git rev-parse HEAD > "$OUT/sha.txt"
git status --short > "$OUT/git-status.txt"
if [[ ${BUILD:-1} == 1 ]]; then
  python tools/build_ppu.py --sdk "$PPU_SDK" --build-dir "$OUT/build" \
    --output "$OUT/_native.so" 2>&1 | tee "$OUT/build.log"
  export COMFY_KITCHEN_PPU_LIBRARY="$OUT/_native.so"
fi
python tools/build_ppu_blaslt.py --sdk "$PPU_SDK" --out "$OUT/blaslt-build" \
  2>&1 | tee "$OUT/blaslt-build.log"
extra=()
if [[ ${CONVROT:-1} == 1 ]]; then extra+=(--convrot); fi
python tools/ppu_blaslt_compare.py --library "$OUT/blaslt-build/_blaslt_probe.so" \
  --m "${M:-4096}" --n "${N:-4096}" --k "${K:-4096}" --dtype "${DTYPE:-bf16}" \
  --samples "${SAMPLES:-7}" --iterations "${ITERATIONS:-20}" --heuristics "${HEURISTICS:-32}" \
  --workspace-mib "${WORKSPACE_MIB:-64}" --peak-tops "${PEAK_TOPS:-1000}" \
  --out "$OUT" "${extra[@]}" 2>&1 | tee "$OUT/compare.log"
printf '[PPU Lt runner] PASS; artifacts: %s\n' "$OUT"
