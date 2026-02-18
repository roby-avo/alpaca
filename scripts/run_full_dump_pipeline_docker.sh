#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
COMPOSE_FILE="${ROOT_DIR}/docker-compose.yml"
# shellcheck source=./lib/docker_checks.sh
. "${SCRIPT_DIR}/lib/docker_checks.sh"

DUMP_PATH="${ALPACA_DUMP_PATH:-}"
OUTPUT_DIR="${ALPACA_OUTPUT_DIR:-${ROOT_DIR}/data/output}"
INDEX_ID="${ALPACA_INDEX_ID:-wikidata_entities_full_$(date +%s)}"

QUICKWIT_URL_INNER="${ALPACA_QUICKWIT_URL_INNER:-http://quickwit:7280}"
QUICKWIT_URL_OUTER="${ALPACA_QUICKWIT_URL_OUTER:-http://localhost:7280}"
API_PORT="${ALPACA_API_PORT:-8000}"
API_BASE_URL="${ALPACA_API_BASE_URL:-http://localhost:${API_PORT}}"

# Memory-safe defaults for large dumps on ~30GB RAM VMs.
LABELS_BATCH_SIZE="${ALPACA_LABELS_BATCH_SIZE:-2000}"
BOW_BATCH_SIZE="${ALPACA_BOW_BATCH_SIZE:-2000}"
QUICKWIT_CHUNK_BYTES="${ALPACA_QUICKWIT_CHUNK_BYTES:-4000000}"
QUICKWIT_FINAL_COMMIT="${ALPACA_QUICKWIT_FINAL_COMMIT:-wait_for}"
QUICKWIT_HTTP_TIMEOUT_SECONDS="${ALPACA_QUICKWIT_HTTP_TIMEOUT_SECONDS:-300}"
LANGUAGES="${ALPACA_LANGUAGES:-en}"
MAX_ALIASES_PER_LANGUAGE="${ALPACA_MAX_ALIASES_PER_LANGUAGE:-8}"
MAX_BOW_TOKENS="${ALPACA_MAX_BOW_TOKENS:-128}"
MAX_CONTEXT_OBJECT_IDS="${ALPACA_MAX_CONTEXT_OBJECT_IDS:-32}"
MAX_CONTEXT_CHARS="${ALPACA_MAX_CONTEXT_CHARS:-640}"
MAX_DOC_BYTES="${ALPACA_MAX_DOC_BYTES:-4096}"
CONTEXT_LABEL_CACHE_SIZE="${ALPACA_CONTEXT_LABEL_CACHE_SIZE:-200000}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

docker_compose() {
  docker compose -f "${COMPOSE_FILE}" "$@"
}

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_full_dump_pipeline_docker.sh --dump-path /absolute/path/latest-all.json.bz2 [--index-id ID]

Options:
  --dump-path PATH            Full dump path on host (.json/.json.gz/.json.bz2). Required.
  --index-id ID               Quickwit index id (default: wikidata_entities_full_<timestamp>).
  --output-dir PATH           Host output dir for labels DB + bow docs (default: ./data/output).

  --labels-batch-size N       SQLite insert batch size (default: 2000).
  --bow-batch-size N          BOW write batch size (default: 2000).
  --quickwit-chunk-bytes N    Ingest chunk size in bytes (default: 4000000).
  --quickwit-final-commit M   auto|wait_for|force (default: wait_for).
  --quickwit-http-timeout-seconds N  HTTP timeout per Quickwit request (default: 300).

  --quickwit-url-inner URL    Quickwit URL from container network (default: http://quickwit:7280).
  --quickwit-url-outer URL    Quickwit URL from host for verification (default: http://localhost:7280).
  --api-base-url URL          API URL from host for health check (default: http://localhost:8000).

  -h, --help                  Show this help.

Notes:
  - Start containers first: docker compose up -d quickwit api
  - This script is streaming and memory-safe by default for large dumps.
  - Tuning can be done via env vars:
      ALPACA_LANGUAGES, ALPACA_MAX_ALIASES_PER_LANGUAGE, ALPACA_MAX_BOW_TOKENS,
      ALPACA_MAX_CONTEXT_OBJECT_IDS, ALPACA_MAX_CONTEXT_CHARS, ALPACA_MAX_DOC_BYTES,
      ALPACA_CONTEXT_LABEL_CACHE_SIZE
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dump-path)
      DUMP_PATH="$2"
      shift 2
      ;;
    --index-id)
      INDEX_ID="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --labels-batch-size)
      LABELS_BATCH_SIZE="$2"
      shift 2
      ;;
    --bow-batch-size)
      BOW_BATCH_SIZE="$2"
      shift 2
      ;;
    --quickwit-chunk-bytes)
      QUICKWIT_CHUNK_BYTES="$2"
      shift 2
      ;;
    --quickwit-final-commit)
      QUICKWIT_FINAL_COMMIT="$2"
      shift 2
      ;;
    --quickwit-http-timeout-seconds)
      QUICKWIT_HTTP_TIMEOUT_SECONDS="$2"
      shift 2
      ;;
    --quickwit-url-inner)
      QUICKWIT_URL_INNER="$2"
      shift 2
      ;;
    --quickwit-url-outer)
      QUICKWIT_URL_OUTER="$2"
      shift 2
      ;;
    --api-base-url)
      API_BASE_URL="$2"
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

if [ -z "${DUMP_PATH}" ]; then
  echo "ERROR: --dump-path is required." >&2
  usage >&2
  exit 1
fi
if [ ! -f "${DUMP_PATH}" ]; then
  echo "ERROR: Dump not found at '${DUMP_PATH}'." >&2
  exit 1
fi

INPUT_DIR=$(CDPATH= cd -- "$(dirname -- "${DUMP_PATH}")" && pwd)
DUMP_NAME=$(basename -- "${DUMP_PATH}")
DUMP_CONTAINER_PATH="/mnt/input/${DUMP_NAME}"

OUTPUT_DIR_ABS=$(CDPATH= cd -- "${OUTPUT_DIR}" 2>/dev/null && pwd || true)
if [ -z "${OUTPUT_DIR_ABS}" ]; then
  mkdir -p "${OUTPUT_DIR}"
  OUTPUT_DIR_ABS=$(CDPATH= cd -- "${OUTPUT_DIR}" && pwd)
fi

LABELS_DB_PATH="/workspace/data/output/wikidata_labels.sqlite"
BOW_OUTPUT_PATH="/workspace/data/output/bow_docs.jsonl.gz"

require_docker_ready
require_cmd curl
require_cmd "${PYTHON_BIN}"

START_CMD="ALPACA_INPUT_DIR=\"${INPUT_DIR}\" ALPACA_OUTPUT_DIR=\"${OUTPUT_DIR_ABS}\" docker compose -f \"${COMPOSE_FILE}\" up -d quickwit api"
require_compose_service_running "${COMPOSE_FILE}" "quickwit" "${START_CMD}"
require_compose_service_running "${COMPOSE_FILE}" "api" "${START_CMD}"

echo "Pipeline configuration:"
echo "  dump_path=${DUMP_PATH}"
echo "  index_id=${INDEX_ID}"
echo "  output_dir=${OUTPUT_DIR_ABS}"
echo "  labels_batch_size=${LABELS_BATCH_SIZE}"
echo "  bow_batch_size=${BOW_BATCH_SIZE}"
echo "  quickwit_chunk_bytes=${QUICKWIT_CHUNK_BYTES}"
echo "  quickwit_final_commit=${QUICKWIT_FINAL_COMMIT}"
echo "  quickwit_http_timeout_seconds=${QUICKWIT_HTTP_TIMEOUT_SECONDS}"
echo "  languages=${LANGUAGES}"
echo "  max_aliases_per_language=${MAX_ALIASES_PER_LANGUAGE}"
echo "  max_bow_tokens=${MAX_BOW_TOKENS}"
echo "  max_context_object_ids=${MAX_CONTEXT_OBJECT_IDS}"
echo "  max_context_chars=${MAX_CONTEXT_CHARS}"
echo "  max_doc_bytes=${MAX_DOC_BYTES}"
echo "  context_label_cache_size=${CONTEXT_LABEL_CACHE_SIZE}"

echo "Step 1/5: Build labels DB (streaming, tqdm progress)"
ALPACA_INPUT_DIR="${INPUT_DIR}" \
ALPACA_OUTPUT_DIR="${OUTPUT_DIR_ABS}" \
docker_compose run --rm api python -m src.build_labels_db \
  --dump-path "${DUMP_CONTAINER_PATH}" \
  --db-path "${LABELS_DB_PATH}" \
  --batch-size "${LABELS_BATCH_SIZE}" \
  --languages "${LANGUAGES}" \
  --max-aliases-per-language "${MAX_ALIASES_PER_LANGUAGE}"

echo "Step 2/5: Build BOW docs (streaming, tqdm progress)"
ALPACA_INPUT_DIR="${INPUT_DIR}" \
ALPACA_OUTPUT_DIR="${OUTPUT_DIR_ABS}" \
docker_compose run --rm api python -m src.build_bow_docs \
  --dump-path "${DUMP_CONTAINER_PATH}" \
  --labels-db-path "${LABELS_DB_PATH}" \
  --output-path "${BOW_OUTPUT_PATH}" \
  --batch-size "${BOW_BATCH_SIZE}" \
  --max-aliases-per-language "${MAX_ALIASES_PER_LANGUAGE}" \
  --max-bow-tokens "${MAX_BOW_TOKENS}" \
  --max-context-object-ids "${MAX_CONTEXT_OBJECT_IDS}" \
  --max-context-chars "${MAX_CONTEXT_CHARS}" \
  --max-doc-bytes "${MAX_DOC_BYTES}" \
  --context-label-cache-size "${CONTEXT_LABEL_CACHE_SIZE}"

echo "Step 3/5: Ingest into Quickwit (tqdm doc progress)"
ALPACA_INPUT_DIR="${INPUT_DIR}" \
ALPACA_OUTPUT_DIR="${OUTPUT_DIR_ABS}" \
docker_compose run --rm api python -m src.build_quickwit_index \
  --quickwit-url "${QUICKWIT_URL_INNER}" \
  --index-id "${INDEX_ID}" \
  --bow-docs-path "${BOW_OUTPUT_PATH}" \
  --chunk-bytes "${QUICKWIT_CHUNK_BYTES}" \
  --http-timeout-seconds "${QUICKWIT_HTTP_TIMEOUT_SECONDS}" \
  --final-commit "${QUICKWIT_FINAL_COMMIT}"

echo "Step 4/5: Rebind API to the new index"
ALPACA_INPUT_DIR="${INPUT_DIR}" \
ALPACA_OUTPUT_DIR="${OUTPUT_DIR_ABS}" \
ALPACA_QUICKWIT_INDEX_ID="${INDEX_ID}" \
docker_compose up -d api

echo "Step 5/5: Verify indexed docs and API health"
attempt=0
until curl -fsS "${API_BASE_URL}/healthz" >/dev/null 2>&1; do
  attempt=$((attempt + 1))
  if [ "${attempt}" -ge 120 ]; then
    echo "ERROR: API health check failed at ${API_BASE_URL}/healthz" >&2
    exit 1
  fi
  sleep 1
done

split_payload=""
indexed_docs=0
attempt=0
while [ "${attempt}" -lt 120 ]; do
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

echo "Full dump pipeline completed successfully."
echo "  quickwit_index=${INDEX_ID}"
echo "  indexed_docs=${indexed_docs}"
echo "  api_base_url=${API_BASE_URL}"
echo "  quickwit_ui=${QUICKWIT_URL_OUTER}/ui/"
