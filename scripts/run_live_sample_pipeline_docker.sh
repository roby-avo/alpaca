#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)

COUNT="${ALPACA_LIVE_SAMPLE_COUNT:-1000}"
IDS="${ALPACA_LIVE_SAMPLE_IDS:-}"
CACHE_DIR="${ALPACA_LIVE_SAMPLE_CACHE_DIR:-${ROOT_DIR}/data/cache/wikidata_entities}"
OUTPUT_DUMP="${ALPACA_LIVE_SAMPLE_DUMP_PATH:-${ROOT_DIR}/data/input/live_sample_dump.json.bz2}"

MAX_AGE_HOURS="${ALPACA_LIVE_SAMPLE_MAX_AGE_HOURS:-72}"
CONCURRENCY="${ALPACA_LIVE_SAMPLE_CONCURRENCY:-12}"
TIMEOUT_SECONDS="${ALPACA_LIVE_SAMPLE_TIMEOUT_SECONDS:-20}"
SLEEP_SECONDS="${ALPACA_LIVE_SAMPLE_SLEEP_SECONDS:-0.05}"
FORCE_REFRESH="${ALPACA_LIVE_SAMPLE_FORCE_REFRESH:-0}"

INDEX_ID="${ALPACA_INDEX_ID:-wikidata_entities_live_$(date +%s)}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_live_sample_pipeline_docker.sh --count 1000 [--force-refresh] [--index-id ID]
  ./scripts/run_live_sample_pipeline_docker.sh --ids Q42,Q90,Q30 [--force-refresh] [--index-id ID]

Options:
  --count N              Number of live entities to fetch if --ids is not provided (default: 1000).
  --ids CSV              Optional explicit QIDs to fetch (example: Q42,Q90,Q30).
  --index-id ID          Quickwit index id for this run.

  --cache-dir PATH       Cache dir for live JSON files (default: ./data/cache/wikidata_entities).
  --output-dump PATH     Output dump path (default: ./data/input/live_sample_dump.json.bz2).

  --max-age-hours H      Cache freshness window (default: 72).
  --concurrency N        Live fetch concurrency (default: 12).
  --timeout-seconds S    HTTP timeout per fetch request (default: 20).
  --sleep-seconds S      Delay after each successful fetch (default: 0.05).
  --force-refresh        Force re-fetch all requested IDs.

  -h, --help             Show this help.

Notes:
  - Start containers first: docker compose up -d quickwit api
  - Step progress is visible with tqdm in each heavy stage.
EOF
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
    --index-id)
      INDEX_ID="$2"
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

mkdir -p "${CACHE_DIR}" "$(dirname -- "${OUTPUT_DUMP}")"

echo "Step 1/3: Fetch live Wikidata entities (tqdm progress)"
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

"${PYTHON_BIN}" "$@"

echo "Step 2/3: Build compact dump from cached live entities (tqdm progress)"
set -- -m src.build_cached_sample_dump \
  --cache-dir "${CACHE_DIR}" \
  --output-path "${OUTPUT_DUMP}"

if [ -n "${IDS}" ]; then
  set -- "$@" --ids "${IDS}"
else
  set -- "$@" --count "${COUNT}"
fi

"${PYTHON_BIN}" "$@"

echo "Step 3/3: Run indexing pipeline on the generated sample dump"
set -- "${ROOT_DIR}/scripts/run_full_dump_pipeline_docker.sh" \
  --dump-path "${OUTPUT_DUMP}" \
  --index-id "${INDEX_ID}"

"$@"
