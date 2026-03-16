#!/usr/bin/env sh
set -eu

PYTHON_BIN="${PYTHON_BIN:-python3}"

SOURCE_DUMP_PATH="${ALPACA_DUMP_PATH:-}"
OUTPUT_PATH="${ALPACA_SMALL_DUMP_OUTPUT_PATH:-./data/input/small_dump.json.bz2}"
COUNT="${ALPACA_SMALL_DUMP_COUNT:-1000}"
IDS="${ALPACA_SMALL_DUMP_IDS:-}"
IDS_FILE="${ALPACA_SMALL_DUMP_IDS_FILE:-}"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/build_small_dump.sh --source-dump-path /path/latest-all.json.bz2 --count 1000
  ./scripts/build_small_dump.sh --source-dump-path /path/latest-all.json.bz2 --ids Q42,Q90,P31

Options:
  --source-dump-path PATH
  --output-path PATH
  --count N
  --ids CSV
  --ids-file PATH
  -h, --help
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --source-dump-path) SOURCE_DUMP_PATH="$2"; shift 2 ;;
    --output-path) OUTPUT_PATH="$2"; shift 2 ;;
    --count) COUNT="$2"; shift 2 ;;
    --ids) IDS="$2"; shift 2 ;;
    --ids-file) IDS_FILE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

mkdir -p "$(dirname -- "$OUTPUT_PATH")"

if [ -z "${SOURCE_DUMP_PATH}" ]; then
  echo "ERROR: --source-dump-path is required." >&2
  usage >&2
  exit 1
fi

set -- "$PYTHON_BIN" -m src.build_small_dump \
  --source-dump-path "$SOURCE_DUMP_PATH" \
  --output-path "$OUTPUT_PATH"
if [ -n "${IDS}" ]; then
  exec "$@" --ids "$IDS"
fi
if [ -n "${IDS_FILE}" ]; then
  exec "$@" --ids-file "$IDS_FILE"
fi
exec "$@" --count "$COUNT"
