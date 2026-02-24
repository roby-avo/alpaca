#!/usr/bin/env sh
set -eu

PYTHON_BIN="${PYTHON_BIN:-python3}"

SOURCE_DUMP_PATH="${ALPACA_DUMP_PATH:-}"
OUTPUT_PATH="${ALPACA_SMALL_DUMP_OUTPUT_PATH:-./data/input/small_dump.json.bz2}"
COUNT="${ALPACA_SMALL_DUMP_COUNT:-1000}"
IDS="${ALPACA_SMALL_DUMP_IDS:-}"
IDS_FILE="${ALPACA_SMALL_DUMP_IDS_FILE:-}"
LIVE_MODE=0
FORCE_REFRESH=0

usage() {
  cat <<'EOF'
Usage:
  # Build from an existing large local dump (supports count or explicit IDs)
  ./scripts/build_small_dump.sh --source-dump-path /path/latest-all.json.bz2 --count 1000
  ./scripts/build_small_dump.sh --source-dump-path /path/latest-all.json.bz2 --ids Q42,Q90,P31

  # Build from live Wikidata via Postgres sample cache
  ./scripts/build_small_dump.sh --live --ids Q42,Q90,Q64
  ./scripts/build_small_dump.sh --live --count 120
  ./scripts/build_small_dump.sh --live --ids-file ./sample_qids.txt --force-refresh

Options:
  --source-dump-path PATH
  --live
  --output-path PATH
  --count N
  --ids CSV
  --ids-file PATH
  --force-refresh   (live mode only)
  -h, --help
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --source-dump-path) SOURCE_DUMP_PATH="$2"; shift 2 ;;
    --live) LIVE_MODE=1; shift ;;
    --output-path) OUTPUT_PATH="$2"; shift 2 ;;
    --count) COUNT="$2"; shift 2 ;;
    --ids) IDS="$2"; shift 2 ;;
    --ids-file) IDS_FILE="$2"; shift 2 ;;
    --force-refresh) FORCE_REFRESH=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

mkdir -p "$(dirname -- "$OUTPUT_PATH")"

if [ "${LIVE_MODE}" -eq 0 ] && [ -n "${SOURCE_DUMP_PATH}" ]; then
  set -- "$PYTHON_BIN" -m src.build_small_dump \
    --source-dump-path "$SOURCE_DUMP_PATH" \
    --output-path "$OUTPUT_PATH"
  if [ -n "${IDS}" ]; then
    exec "$@" --ids "$IDS"
  fi
  exec "$@" --count "$COUNT"
fi

# Live mode (explicit --live or no source dump path) uses Postgres cache and supports
# either explicit IDs or deterministic --count (Q1..Q<count>).
if [ -z "${IDS}" ] && [ -z "${IDS_FILE}" ] && [ -z "${COUNT}" ]; then
  echo "ERROR: Live mode requires --ids, --ids-file, or --count." >&2
  exit 1
fi

SELECTORS=0
[ -n "${IDS}" ] && SELECTORS=$((SELECTORS + 1))
[ -n "${IDS_FILE}" ] && SELECTORS=$((SELECTORS + 1))
[ -n "${COUNT}" ] && SELECTORS=$((SELECTORS + 1))
if [ "${SELECTORS}" -ne 1 ]; then
  echo "ERROR: Use exactly one of --ids, --ids-file, or --count in live mode." >&2
  exit 1
fi

set -- "$PYTHON_BIN" -m src.wikidata_sample_postgres
if [ -n "${IDS}" ]; then
  set -- "$@" --ids "$IDS"
fi
if [ -n "${IDS_FILE}" ]; then
  set -- "$@" --ids-file "$IDS_FILE"
fi
if [ -n "${COUNT}" ]; then
  set -- "$@" --count "$COUNT"
fi
if [ "${FORCE_REFRESH}" -eq 1 ]; then
  set -- "$@" --force-refresh
fi
"$@"

set -- "$PYTHON_BIN" -m src.build_postgres_sample_dump \
  --output-path "$OUTPUT_PATH"
if [ -n "${IDS}" ]; then
  set -- "$@" --ids "$IDS"
fi
if [ -n "${IDS_FILE}" ]; then
  set -- "$@" --ids-file "$IDS_FILE"
fi
if [ -n "${COUNT}" ]; then
  set -- "$@" --count "$COUNT"
fi
exec "$@"
