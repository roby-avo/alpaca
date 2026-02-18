#!/usr/bin/env sh
set -eu

HOST="${ALPACA_API_HOST:-0.0.0.0}"
PORT="${ALPACA_API_PORT:-8000}"

exec uvicorn src.api:app --host "$HOST" --port "$PORT"
