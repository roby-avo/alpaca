#!/usr/bin/env sh
set -eu

PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" -m unittest tests.test_end_to_end_small_sample -v
