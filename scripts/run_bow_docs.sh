#!/usr/bin/env sh
set -eu

DUMP_PATH="${ALPACA_DUMP_PATH:-}"
DB_PATH="${ALPACA_LABELS_DB_PATH:-./data/output/wikidata_labels.sqlite}"
OUTPUT_PATH="${ALPACA_BOW_OUTPUT_PATH:-./data/output/bow_docs.jsonl.gz}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ -z "${DUMP_PATH}" ]; then
  echo "ERROR: Set ALPACA_DUMP_PATH to the dump location (or pass --dump-path directly to python)." >&2
  echo "Example: ALPACA_DUMP_PATH=/absolute/path/latest-all.json.bz2 ./scripts/run_bow_docs.sh" >&2
  exit 1
fi

exec "$PYTHON_BIN" -m src.build_bow_docs \
  --dump-path "$DUMP_PATH" \
  --labels-db-path "$DB_PATH" \
  --output-path "$OUTPUT_PATH" \
  "$@"
