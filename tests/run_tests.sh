#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd -- "${script_dir}/.." && pwd)

cd "${repository_root}"
python3 tests/st/scripts/check_coverage.py
python3 -m pytest -q tests/ut
