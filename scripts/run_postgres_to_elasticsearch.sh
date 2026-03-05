#!/usr/bin/env sh
set -eu

PYTHON_BIN="${PYTHON_BIN:-python3}"

exec "$PYTHON_BIN" -m src.index_postgres_to_elasticsearch "$@"
