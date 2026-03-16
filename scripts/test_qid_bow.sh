#!/usr/bin/env sh
set -eu

PYTHON_BIN="${PYTHON_BIN:-python3}"

exec "$PYTHON_BIN" -m src.test_qid_bow "$@"
