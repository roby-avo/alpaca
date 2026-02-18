#!/usr/bin/env sh
set -eu

PYTHON_BIN="${PYTHON_BIN:-python3}"
SOURCE_DUMP_PATH="${ALPACA_DUMP_PATH:-}"
DEFAULT_OUTPUT_PATH="./data/input/live_sample_dump.json.bz2"
OUTPUT_PATH="${ALPACA_SMALL_DUMP_OUTPUT_PATH:-${DEFAULT_OUTPUT_PATH}}"
COUNT="${ALPACA_SMALL_DUMP_COUNT:-1000}"
IDS="${ALPACA_SMALL_DUMP_IDS:-}"

# Live-mode options (used when --live is set or --source-dump-path is omitted).
LIVE_MODE=0
CACHE_DIR="${ALPACA_SMALL_DUMP_CACHE_DIR:-./data/cache/wikidata_entities}"
MAX_AGE_HOURS="${ALPACA_SMALL_DUMP_MAX_AGE_HOURS:-72}"
CONCURRENCY="${ALPACA_SMALL_DUMP_CONCURRENCY:-12}"
TIMEOUT_SECONDS="${ALPACA_SMALL_DUMP_TIMEOUT_SECONDS:-20}"
SLEEP_SECONDS="${ALPACA_SMALL_DUMP_SLEEP_SECONDS:-0.05}"
FORCE_REFRESH=0

usage() {
  cat <<'EOF'
Usage:
  # Build from an existing large dump
  ./scripts/build_small_dump.sh --source-dump-path /absolute/path/latest-all.json.bz2 [--count 1000]
  ./scripts/build_small_dump.sh --source-dump-path /absolute/path/latest-all.json.bz2 --ids Q42,Q90,P31

  # Build from live Wikidata (no local dump required)
  ./scripts/build_small_dump.sh --live --count 1000
  ./scripts/build_small_dump.sh --live --ids Q42,Q90,P31 --force-refresh

Options:
  --source-dump-path PATH   Source dump path (.json/.json.gz/.json.bz2).
  --live                    Force live mode (fetch from Wikidata + build compact dump from cache).
  --output-path PATH        Output path for small dump (.json/.json.gz/.json.bz2).
                            Default: ./data/input/live_sample_dump.json.bz2
  --count N                 Number of Q* IDs when --ids is omitted (default: 1000).
  --ids CSV                 Optional explicit entity IDs (Q*), comma-separated.

  Live-mode options:
  --cache-dir PATH          Cache dir for live EntityData JSON (default: ./data/cache/wikidata_entities).
  --max-age-hours H         Cache freshness window (default: 72).
  --concurrency N           Live fetch concurrency (default: 12).
  --timeout-seconds S       HTTP timeout per request (default: 20).
  --sleep-seconds S         Delay after each successful fetch (default: 0.05).
  --force-refresh           Force re-fetch requested IDs from Wikidata.

  -h, --help                Show this help.

Mode selection:
  1) If --live is passed: live mode.
  2) Else if --source-dump-path is provided: source-dump mode.
  3) Else: live mode.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --source-dump-path)
      SOURCE_DUMP_PATH="$2"
      shift 2
      ;;
    --live)
      LIVE_MODE=1
      shift
      ;;
    --output-path)
      OUTPUT_PATH="$2"
      shift 2
      ;;
    --count)
      COUNT="$2"
      shift 2
      ;;
    --ids)
      IDS="$2"
      shift 2
      ;;
    --cache-dir)
      CACHE_DIR="$2"
      shift 2
      ;;
    --max-age-hours)
      MAX_AGE_HOURS="$2"
      shift 2
      ;;
    --concurrency)
      CONCURRENCY="$2"
      shift 2
      ;;
    --timeout-seconds)
      TIMEOUT_SECONDS="$2"
      shift 2
      ;;
    --sleep-seconds)
      SLEEP_SECONDS="$2"
      shift 2
      ;;
    --force-refresh)
      FORCE_REFRESH=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [ "${LIVE_MODE}" -eq 0 ] && [ -n "${SOURCE_DUMP_PATH}" ]; then
  if [ -n "${IDS}" ]; then
    exec "$PYTHON_BIN" -m src.build_small_dump \
      --source-dump-path "$SOURCE_DUMP_PATH" \
      --output-path "$OUTPUT_PATH" \
      --ids "$IDS"
  fi

  exec "$PYTHON_BIN" -m src.build_small_dump \
    --source-dump-path "$SOURCE_DUMP_PATH" \
    --output-path "$OUTPUT_PATH" \
    --count "$COUNT"
fi

# Live mode (explicit --live OR no --source-dump-path provided).
mkdir -p "$(dirname -- "$OUTPUT_PATH")" "${CACHE_DIR}"

echo "Building small dump from live Wikidata cache..."

echo "Step 1/2: fetch live entities into cache"
set -- -m src.wikidata_sample_cache \
  --cache-dir "$CACHE_DIR" \
  --max-age-hours "$MAX_AGE_HOURS" \
  --concurrency "$CONCURRENCY" \
  --timeout-seconds "$TIMEOUT_SECONDS" \
  --sleep-seconds "$SLEEP_SECONDS"

if [ -n "${IDS}" ]; then
  set -- "$@" --ids "$IDS"
else
  set -- "$@" --count "$COUNT"
fi

if [ "${FORCE_REFRESH}" -eq 1 ]; then
  set -- "$@" --force-refresh
fi

"$PYTHON_BIN" "$@"

echo "Step 2/2: build compact .json.bz2 dump from cache"
set -- -m src.build_cached_sample_dump \
  --cache-dir "$CACHE_DIR" \
  --output-path "$OUTPUT_PATH"

if [ -n "${IDS}" ]; then
  set -- "$@" --ids "$IDS"
else
  set -- "$@" --count "$COUNT"
fi

exec "$PYTHON_BIN" "$@"
