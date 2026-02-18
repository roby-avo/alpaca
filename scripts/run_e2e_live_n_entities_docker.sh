#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)

COUNT="${ALPACA_E2E_COUNT:-50}"
IDS="${ALPACA_E2E_IDS:-}"
CACHE_DIR="${ALPACA_E2E_CACHE_DIR:-${ROOT_DIR}/data/cache/wikidata_entities}"
OUTPUT_DUMP="${ALPACA_E2E_OUTPUT_DUMP:-${ROOT_DIR}/data/input/live_sample_dump.json.bz2}"

MAX_AGE_HOURS="${ALPACA_E2E_MAX_AGE_HOURS:-72}"
CONCURRENCY="${ALPACA_E2E_CONCURRENCY:-12}"
TIMEOUT_SECONDS="${ALPACA_E2E_TIMEOUT_SECONDS:-20}"
SLEEP_SECONDS="${ALPACA_E2E_SLEEP_SECONDS:-0.05}"
FORCE_REFRESH="${ALPACA_E2E_FORCE_REFRESH:-0}"

INDEX_ID="${ALPACA_E2E_INDEX_ID:-}"
SEARCH_QUERY="${ALPACA_E2E_SEARCH_QUERY:-apple}"
KEEP_SERVICES="${ALPACA_E2E_KEEP_SERVICES:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_e2e_live_n_entities_docker.sh --count N [--ids Q42,Q90] [--force-refresh] [--keep-services]

Options:
  --count N             Number of live QIDs to fetch when --ids is not provided (default: 50).
  --ids CSV             Optional explicit QIDs (example: Q42,Q90).
  --cache-dir PATH      Cache dir for live EntityData JSON (default: ./data/cache/wikidata_entities).
  --output-dump PATH    Output .json.bz2 sample dump built from cache.
  --max-age-hours H     Cache freshness window for live fetch (default: 72).
  --concurrency N       Live fetch concurrency (default: 12).
  --timeout-seconds S   HTTP timeout per live fetch request (default: 20).
  --sleep-seconds S     Delay after each successful live fetch (default: 0.05).
  --force-refresh       Force re-fetch all requested IDs from Wikidata.
  --index-id ID         Quickwit index id override.
  --search-query QUERY  API smoke query text after indexing (default: apple).
  --keep-services       Deprecated no-op (services are never stopped by E2E scripts).
  -h, --help            Show this help.
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: Missing required command: $1" >&2
    exit 1
  fi
}

while [ "$#" -gt 0 ]; do
  case "$1" in
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
    --output-dump)
      OUTPUT_DUMP="$2"
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
    --index-id)
      INDEX_ID="$2"
      shift 2
      ;;
    --search-query)
      SEARCH_QUERY="$2"
      shift 2
      ;;
    --keep-services)
      KEEP_SERVICES=1
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

require_cmd "${PYTHON_BIN}"
require_cmd docker
require_cmd curl

mkdir -p "${CACHE_DIR}" "$(dirname -- "${OUTPUT_DUMP}")"

set -- -m src.wikidata_sample_cache \
  --cache-dir "${CACHE_DIR}" \
  --max-age-hours "${MAX_AGE_HOURS}" \
  --concurrency "${CONCURRENCY}" \
  --timeout-seconds "${TIMEOUT_SECONDS}" \
  --sleep-seconds "${SLEEP_SECONDS}"

if [ -n "${IDS}" ]; then
  set -- "$@" --ids "${IDS}"
else
  set -- "$@" --count "${COUNT}"
fi

if [ "${FORCE_REFRESH}" = "1" ]; then
  set -- "$@" --force-refresh
fi

echo "Step 1/3: Fetch live Wikidata entity JSON into cache..."
"${PYTHON_BIN}" "$@"

echo "Step 2/3: Build compact sample dump from cached live JSON..."
set -- -m src.build_cached_sample_dump \
  --cache-dir "${CACHE_DIR}" \
  --output-path "${OUTPUT_DUMP}"

if [ -n "${IDS}" ]; then
  set -- "$@" --ids "${IDS}"
else
  set -- "$@" --count "${COUNT}"
fi

"${PYTHON_BIN}" "$@"

echo "Step 3/3: Run Docker E2E pipeline + Quickwit indexing + API smoke..."
set -- "${ROOT_DIR}/scripts/run_e2e_n_entities_docker.sh" \
  --dump-path "${OUTPUT_DUMP}" \
  --count "${COUNT}" \
  --search-query "${SEARCH_QUERY}"

if [ -n "${INDEX_ID}" ]; then
  set -- "$@" --index-id "${INDEX_ID}"
fi
if [ "${KEEP_SERVICES}" = "1" ]; then
  set -- "$@" --keep-services
fi

"$@"
