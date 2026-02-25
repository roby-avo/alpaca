#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)

IDS="${ALPACA_LIVE_SAMPLE_IDS:-}"
IDS_FILE="${ALPACA_LIVE_SAMPLE_IDS_FILE:-}"
COUNT="${ALPACA_LIVE_SAMPLE_COUNT:-}"
OUTPUT_DUMP="${ALPACA_LIVE_SAMPLE_DUMP_PATH:-${ROOT_DIR}/data/input/live_sample_dump.json.bz2}"
FORCE_REFRESH=0
LIVE_CONCURRENCY="${ALPACA_LIVE_SAMPLE_CONCURRENCY:-}"
LIVE_SLEEP_SECONDS="${ALPACA_LIVE_SAMPLE_SLEEP_SECONDS:-}"
LIVE_HTTP_MAX_RETRIES="${ALPACA_LIVE_SAMPLE_HTTP_MAX_RETRIES:-}"
LIVE_MAX_CONTEXT_SUPPORT_PREFETCH="${ALPACA_LIVE_SAMPLE_MAX_CONTEXT_SUPPORT_PREFETCH:-}"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_live_sample_pipeline_docker.sh --ids Q42,Q90,Q64
  ./scripts/run_live_sample_pipeline_docker.sh --ids-file ./sample_qids.txt
  ./scripts/run_live_sample_pipeline_docker.sh --count 120

Options:
  --ids CSV           Explicit QIDs for the live demo sample cache (required if --ids-file is omitted).
  --ids-file PATH     Text file with one QID per line (comments with # allowed).
  --count N           Deterministic count mode (probes upward from Q1 and skips missing IDs).
  --output-dump PATH  Output dump path (default: ./data/input/live_sample_dump.json.bz2).
  --force-refresh     Refetch QIDs even if already present in Postgres sample cache.
  --concurrency N     Live demo fetch concurrency for Wikidata requests (seed fetch; support fetch is internally capped).
  --sleep-seconds S   Delay after successful Wikidata fetches (support fetch has a minimum throttle).
  --http-max-retries N
                     Retries per item for transient HTTP/network errors (429/5xx).
  --max-context-support-prefetch N
                     Cap one-hop support entities fetched for context labels (0 disables support prefetch).
  -h, --help          Show this help.

Notes:
  - Uses Postgres table sample_entity_cache for live demo caching.
  - --count is deterministic: probes upward from Q1 and skips missing QIDs (404s).
  - Automatically prefetches one-hop related entity IDs from claim objects so context strings can be built properly.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --ids)
      IDS="$2"
      shift 2
      ;;
    --ids-file)
      IDS_FILE="$2"
      shift 2
      ;;
    --count)
      COUNT="$2"
      shift 2
      ;;
    --output-dump)
      OUTPUT_DUMP="$2"
      shift 2
      ;;
    --force-refresh)
      FORCE_REFRESH=1
      shift
      ;;
    --concurrency)
      LIVE_CONCURRENCY="$2"
      shift 2
      ;;
    --sleep-seconds)
      LIVE_SLEEP_SECONDS="$2"
      shift 2
      ;;
    --http-max-retries)
      LIVE_HTTP_MAX_RETRIES="$2"
      shift 2
      ;;
    --max-context-support-prefetch)
      LIVE_MAX_CONTEXT_SUPPORT_PREFETCH="$2"
      shift 2
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

SELECTORS=0
[ -n "${IDS}" ] && SELECTORS=$((SELECTORS + 1))
[ -n "${IDS_FILE}" ] && SELECTORS=$((SELECTORS + 1))
[ -n "${COUNT}" ] && SELECTORS=$((SELECTORS + 1))
if [ "${SELECTORS}" -ne 1 ]; then
  echo "ERROR: Provide exactly one of --ids, --ids-file, or --count." >&2
  usage >&2
  exit 1
fi

mkdir -p "$(dirname -- "${OUTPUT_DUMP}")"

echo "Step 1/4: Start local services (Postgres + Adminer + API)"
docker compose up -d postgres adminer api

echo "Step 2/4: Fetch live Wikidata seed entities (+ one-hop context support entities) into Postgres sample cache"
set -- docker compose exec -T api python -m src.wikidata_sample_postgres
if [ -n "${IDS}" ]; then
  set -- "$@" --ids "${IDS}"
fi
if [ -n "${IDS_FILE}" ]; then
  set -- "$@" --ids-file "${IDS_FILE}"
fi
if [ -n "${COUNT}" ]; then
  set -- "$@" --count "${COUNT}"
fi
if [ "${FORCE_REFRESH}" -eq 1 ]; then
  set -- "$@" --force-refresh
fi
if [ -n "${LIVE_CONCURRENCY}" ]; then
  set -- "$@" --concurrency "${LIVE_CONCURRENCY}"
fi
if [ -n "${LIVE_SLEEP_SECONDS}" ]; then
  set -- "$@" --sleep-seconds "${LIVE_SLEEP_SECONDS}"
fi
if [ -n "${LIVE_HTTP_MAX_RETRIES}" ]; then
  set -- "$@" --http-max-retries "${LIVE_HTTP_MAX_RETRIES}"
fi
if [ -n "${LIVE_MAX_CONTEXT_SUPPORT_PREFETCH}" ]; then
  set -- "$@" --max-context-support-prefetch "${LIVE_MAX_CONTEXT_SUPPORT_PREFETCH}"
fi
"$@"

echo "Step 3/4: Build compact sample dump from Postgres sample cache"
set -- docker compose exec -T api python -m src.build_postgres_sample_dump \
  --output-path "${OUTPUT_DUMP}"
if [ -n "${IDS}" ]; then
  set -- "$@" --ids "${IDS}"
fi
if [ -n "${IDS_FILE}" ]; then
  set -- "$@" --ids-file "${IDS_FILE}"
fi
if [ -n "${COUNT}" ]; then
  set -- "$@" --count "${COUNT}"
fi
"$@"

echo "Step 4/4: Run Postgres pipeline on the generated sample dump"
set -- docker compose exec -T api python -m src.run_pipeline \
  --dump-path "${OUTPUT_DUMP}"
"$@"

echo "Done."
echo "API:       http://localhost:${ALPACA_API_PORT:-8000}"
echo "Adminer:   http://localhost:${ALPACA_ADMINER_PORT:-8080}"
