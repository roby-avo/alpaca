#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
COMPOSE_FILE="${ROOT_DIR}/docker-compose.yml"
# shellcheck source=./lib/docker_checks.sh
. "${SCRIPT_DIR}/lib/docker_checks.sh"

COUNT="${ALPACA_E2E_COUNT:-100}"
DUMP_PATH="${ALPACA_E2E_DUMP_PATH:-}"
OUTPUT_DIR="${ALPACA_OUTPUT_DIR:-${ROOT_DIR}/data/output}"
INDEX_ID="${ALPACA_E2E_INDEX_ID:-}"
SEARCH_QUERY="${ALPACA_E2E_SEARCH_QUERY:-apple}"
KEEP_SERVICES="${ALPACA_E2E_KEEP_SERVICES:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

QUICKWIT_URL_INNER="${ALPACA_E2E_QUICKWIT_URL_INNER:-http://quickwit:7280}"
QUICKWIT_URL_OUTER="${ALPACA_E2E_QUICKWIT_URL_OUTER:-http://localhost:7280}"
API_PORT="${ALPACA_API_PORT:-8000}"
API_BASE_URL="${ALPACA_E2E_API_BASE_URL:-http://localhost:${API_PORT}}"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_e2e_n_entities_docker.sh --dump-path /absolute/path/latest-all.json.bz2 --count N [--index-id ID] [--search-query QUERY] [--keep-services]

Options:
  --dump-path PATH      Host path to dump (.json/.json.gz/.json.bz2). Required.
  --count N             Number of parsed entities for the sample run (default: 100).
  --index-id ID         Quickwit index id override (default: auto-generated).
  --search-query QUERY  API smoke query text (default: apple).
  --keep-services       Deprecated no-op (services are never stopped by this script).
  -h, --help            Show this help.
EOF
}

docker_compose() {
  docker compose -f "${COMPOSE_FILE}" "$@"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dump-path)
      DUMP_PATH="$2"
      shift 2
      ;;
    --count)
      COUNT="$2"
      shift 2
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

case "${COUNT}" in
  ''|*[!0-9]*)
    echo "ERROR: --count must be a positive integer." >&2
    exit 1
    ;;
esac
if [ "${COUNT}" -le 0 ]; then
  echo "ERROR: --count must be > 0." >&2
  exit 1
fi

if [ -z "${DUMP_PATH}" ]; then
  echo "ERROR: --dump-path is required (or set ALPACA_E2E_DUMP_PATH)." >&2
  echo "Use an explicit dump location for deterministic runs." >&2
  exit 1
fi

if [ ! -f "${DUMP_PATH}" ]; then
  echo "ERROR: Dump file not found at '${DUMP_PATH}'." >&2
  exit 1
fi

if [ -z "${INDEX_ID}" ]; then
  INDEX_ID="wikidata_entities_sample_n${COUNT}_$(date +%s)"
fi

INPUT_DIR=$(CDPATH= cd -- "$(dirname -- "${DUMP_PATH}")" && pwd)
DUMP_NAME=$(basename -- "${DUMP_PATH}")
OUTPUT_DIR_ABS=$(CDPATH= cd -- "${OUTPUT_DIR}" 2>/dev/null && pwd || true)
if [ -z "${OUTPUT_DIR_ABS}" ]; then
  mkdir -p "${OUTPUT_DIR}"
  OUTPUT_DIR_ABS=$(CDPATH= cd -- "${OUTPUT_DIR}" && pwd)
fi

DUMP_CONTAINER_PATH="/mnt/input/${DUMP_NAME}"

require_docker_ready
require_cmd curl
require_cmd "${PYTHON_BIN}"

START_CMD="ALPACA_INPUT_DIR=\"${INPUT_DIR}\" ALPACA_OUTPUT_DIR=\"${OUTPUT_DIR_ABS}\" docker compose -f \"${COMPOSE_FILE}\" up -d quickwit api"
require_compose_service_running "${COMPOSE_FILE}" "quickwit" "${START_CMD}"
require_compose_service_running "${COMPOSE_FILE}" "api" "${START_CMD}"

if [ "${KEEP_SERVICES}" = "1" ]; then
  echo "Note: --keep-services is a no-op; services are never stopped by this script."
fi

echo "Running Docker E2E sample pipeline:"
echo "  dump_path=${DUMP_PATH}"
echo "  count=${COUNT}"
echo "  index_id=${INDEX_ID}"
echo "  input_dir=${INPUT_DIR}"
echo "  output_dir=${OUTPUT_DIR_ABS}"

ALPACA_INPUT_DIR="${INPUT_DIR}" \
ALPACA_OUTPUT_DIR="${OUTPUT_DIR_ABS}" \
ALPACA_DUMP_PATH="${DUMP_CONTAINER_PATH}" \
docker_compose run --rm api python -m src.run_pipeline \
  --quickwit-url "${QUICKWIT_URL_INNER}" \
  --index-id "${INDEX_ID}" \
  --limit "${COUNT}" \
  --quickwit-final-commit auto

# Reconfigure the long-running API service to query the new index.
ALPACA_INPUT_DIR="${INPUT_DIR}" \
ALPACA_OUTPUT_DIR="${OUTPUT_DIR_ABS}" \
ALPACA_QUICKWIT_INDEX_ID="${INDEX_ID}" \
docker_compose up -d api

attempt=0
until curl -fsS "${API_BASE_URL}/healthz" >/dev/null 2>&1; do
  attempt=$((attempt + 1))
  if [ "${attempt}" -ge 60 ]; then
    echo "ERROR: API health check failed at ${API_BASE_URL}/healthz" >&2
    exit 1
  fi
  sleep 1
done

split_payload=""
indexed_docs=0
attempt=0
while [ "${attempt}" -lt 60 ]; do
  current_payload=$(curl -fsS "${QUICKWIT_URL_OUTER}/api/v1/indexes/${INDEX_ID}/splits")
  split_payload="${current_payload}"
  current_docs=$("${PYTHON_BIN}" - "${current_payload}" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
splits = payload.get("splits") if isinstance(payload, dict) else []
if not isinstance(splits, list):
    print(0)
    raise SystemExit(0)

total = 0
for split in splits:
    if not isinstance(split, dict):
        continue
    num_docs = split.get("num_docs")
    if isinstance(num_docs, int):
        total += num_docs
print(total)
PY
)
  indexed_docs="${current_docs}"
  if [ "${indexed_docs}" -gt 0 ]; then
    break
  fi
  attempt=$((attempt + 1))
  sleep 1
done

if [ "${indexed_docs}" -le 0 ]; then
  echo "ERROR: Quickwit index '${INDEX_ID}' has 0 indexed docs. Payload: ${split_payload}" >&2
  exit 1
fi

search_request=$("${PYTHON_BIN}" - "${SEARCH_QUERY}" <<'PY'
import json
import sys

query_text = sys.argv[1]
print(json.dumps({"query": query_text, "limit": 10}, ensure_ascii=False))
PY
)

search_response=""
attempt=0
while [ "${attempt}" -lt 60 ]; do
  if ! current_response=$(curl -sS -X POST "${API_BASE_URL}/v1/entities/search" \
    -H "Content-Type: application/json" \
    -d "${search_request}" 2>/dev/null); then
    attempt=$((attempt + 1))
    sleep 1
    continue
  fi

  search_response="${current_response}"
  if "${PYTHON_BIN}" - "${search_response}" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
if not isinstance(payload, dict):
    raise SystemExit(1)
if "hits" not in payload:
    raise SystemExit(1)
PY
  then
    break
  fi

  attempt=$((attempt + 1))
  sleep 1
done

if [ -z "${search_response}" ]; then
  echo "ERROR: API search endpoint did not return a valid payload." >&2
  exit 1
fi
if [ "${attempt}" -ge 60 ]; then
  echo "ERROR: API search endpoint did not become ready within timeout. Last payload: ${search_response}" >&2
  exit 1
fi

returned_count=$("${PYTHON_BIN}" - "${search_response}" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
returned = payload.get("returned")
if not isinstance(returned, int):
    returned = 0
print(returned)
PY
)

echo "E2E sample pipeline completed."
echo "  quickwit_index=${INDEX_ID}"
echo "  indexed_docs=${indexed_docs}"
echo "  api_search_query=${SEARCH_QUERY}"
echo "  api_returned=${returned_count}"
echo "  api_base_url=${API_BASE_URL}"
echo "  quickwit_ui=${QUICKWIT_URL_OUTER}/ui/"
