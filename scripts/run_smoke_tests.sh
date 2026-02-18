#!/usr/bin/env sh
set -eu

PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" -m unittest discover -s tests -p 'test_*.py' -v
