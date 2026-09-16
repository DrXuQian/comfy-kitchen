#!/usr/bin/env bash
set -euo pipefail
bundle_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python - "$bundle_dir" <<'PY'
import hashlib
import json
import pathlib
import subprocess
import sys

root = pathlib.Path(sys.argv[1])
release = json.loads((root / 'release.json').read_text())
filename = release['wheel']
if pathlib.Path(filename).name != filename:
    raise SystemExit('Invalid wheel filename in manifest')
wheel = root / filename
if not wheel.is_file() or hashlib.sha256(wheel.read_bytes()).hexdigest() != release['wheel_sha256']:
    raise SystemExit('Wheel missing or SHA256 mismatch; fetch the complete artifact branch.')
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--force-reinstall', str(wheel)], check=True)
print('Installed', release['version'], '; unset COMFY_KITCHEN_PPU_LIBRARY to use the wheel native library.')
PY
