#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
COMPOSE_FILE="${ROOT_DIR}/docker-compose.yml"
# shellcheck source=./lib/docker_checks.sh
. "${SCRIPT_DIR}/lib/docker_checks.sh"

INPUT_DIR="${ALPACA_INPUT_DIR:-${ROOT_DIR}/data/input}"
OUTPUT_DIR="${ALPACA_OUTPUT_DIR:-${ROOT_DIR}/data/output}"

DUMP_NAME="${ALPACA_E2E_DUMP_NAME:-docker_e2e_tiny_dump.json.bz2}"
NER_TYPES_NAME="${ALPACA_E2E_NER_TYPES_NAME:-docker_e2e_ner_types.jsonl}"
INDEX_ID="${ALPACA_E2E_INDEX_ID:-wikidata_entities_e2e_$(date +%s)}"

DUMP_HOST_PATH="${INPUT_DIR}/${DUMP_NAME}"
NER_TYPES_HOST_PATH="${INPUT_DIR}/${NER_TYPES_NAME}"
DUMP_CONTAINER_PATH="/mnt/input/${DUMP_NAME}"
NER_TYPES_CONTAINER_PATH="/mnt/input/${NER_TYPES_NAME}"

QUICKWIT_URL_INNER="${ALPACA_E2E_QUICKWIT_URL_INNER:-http://quickwit:7280}"
API_PORT="${ALPACA_API_PORT:-8000}"
API_BASE_URL="${ALPACA_E2E_API_BASE_URL:-http://localhost:${API_PORT}}"

KEEP_SERVICES="${ALPACA_E2E_KEEP_SERVICES:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

docker_compose() {
  docker compose -f "${COMPOSE_FILE}" "$@"
}

require_docker_ready
require_cmd curl
require_cmd "${PYTHON_BIN}"

mkdir -p "${INPUT_DIR}" "${OUTPUT_DIR}"

START_CMD="ALPACA_INPUT_DIR=\"${INPUT_DIR}\" ALPACA_OUTPUT_DIR=\"${OUTPUT_DIR}\" docker compose -f \"${COMPOSE_FILE}\" up -d quickwit api"
require_compose_service_running "${COMPOSE_FILE}" "quickwit" "${START_CMD}"
require_compose_service_running "${COMPOSE_FILE}" "api" "${START_CMD}"

if [ "${KEEP_SERVICES}" = "1" ]; then
  echo "Note: ALPACA_E2E_KEEP_SERVICES is a no-op; services are never stopped by this script."
fi

"${PYTHON_BIN}" - "${DUMP_HOST_PATH}" "${NER_TYPES_HOST_PATH}" <<'PY'
import bz2
import json
import sys
from pathlib import Path

dump_path = Path(sys.argv[1])
ner_types_path = Path(sys.argv[2])

entities = [
    {
        "id": "Q312",
        "labels": {"en": {"language": "en", "value": "Apple Inc."}},
        "aliases": {"en": [{"language": "en", "value": "Apple"}]},
        "descriptions": {
            "en": {"language": "en", "value": "American technology company based in Cupertino."}
        },
    },
    {
        "id": "Q89",
        "labels": {"en": {"language": "en", "value": "apple"}},
        "aliases": {"en": [{"language": "en", "value": "fruit"}]},
        "descriptions": {
            "en": {"language": "en", "value": "Edible fruit produced by an apple tree."}
        },
    },
    {
        "id": "L1",
        "labels": {"en": {"language": "en", "value": "ignored lexeme"}},
        "descriptions": {"en": {"language": "en", "value": "should be ignored"}},
    },
]

ner_types = [
    {"id": "Q312", "coarse_types": ["ORGANIZATION"], "fine_types": ["COMPANY"]},
    {"id": "Q89", "coarse_types": ["CONCEPT"], "fine_types": ["FOOD"]},
]

dump_path.parent.mkdir(parents=True, exist_ok=True)
ner_types_path.parent.mkdir(parents=True, exist_ok=True)

with bz2.open(dump_path, mode="wt", encoding="utf-8") as handle:
    handle.write("[\n")
    for idx, entity in enumerate(entities):
        line = json.dumps(entity, ensure_ascii=False)
        suffix = "," if idx < len(entities) - 1 else ""
        handle.write(f"{line}{suffix}\n")
    handle.write("]\n")

ner_types_path.write_text(
    "\n".join(json.dumps(item, ensure_ascii=False) for item in ner_types) + "\n",
    encoding="utf-8",
)
PY

echo "Prepared sample files:"
echo "  dump: ${DUMP_HOST_PATH}"
echo "  ner_types: ${NER_TYPES_HOST_PATH}"
echo "  index_id: ${INDEX_ID}"

ALPACA_INPUT_DIR="${INPUT_DIR}" \
ALPACA_OUTPUT_DIR="${OUTPUT_DIR}" \
ALPACA_DUMP_PATH="${DUMP_CONTAINER_PATH}" \
ALPACA_NER_TYPES_PATH="${NER_TYPES_CONTAINER_PATH}" \
docker_compose run --rm api python -m src.run_pipeline \
  --quickwit-url "${QUICKWIT_URL_INNER}" \
  --index-id "${INDEX_ID}" \
  --quickwit-final-commit auto

# Reconfigure the long-running API service to point at the freshly built index.
ALPACA_INPUT_DIR="${INPUT_DIR}" \
ALPACA_OUTPUT_DIR="${OUTPUT_DIR}" \
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

unfiltered_response=""
attempt=0
while [ "${attempt}" -lt 60 ]; do
  if ! current_response=$(curl -sS -X POST "${API_BASE_URL}/v1/entities/search" \
    -H "Content-Type: application/json" \
    -d '{"query":"apple","context":"technology cupertino","limit":5}' 2>/dev/null); then
    attempt=$((attempt + 1))
    sleep 1
    continue
  fi
  unfiltered_response="${current_response}"

  if "${PYTHON_BIN}" - "${current_response}" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
hits = payload.get("hits") or []
if len(hits) < 2:
    raise SystemExit(1)
PY
  then
    break
  fi

  attempt=$((attempt + 1))
  sleep 1
done

if [ "${attempt}" -ge 60 ]; then
  echo "ERROR: Docs were not searchable within timeout. Last payload: ${unfiltered_response}" >&2
  exit 1
fi

filtered_response=""
attempt=0
while [ "${attempt}" -lt 60 ]; do
  if ! current_filtered=$(curl -sS -X POST "${API_BASE_URL}/v1/entities/search" \
    -H "Content-Type: application/json" \
    -d '{"query":"apple","context":"technology cupertino","ner_coarse_types":["ORGANIZATION"],"ner_fine_types":["COMPANY"],"limit":5}' 2>/dev/null); then
    attempt=$((attempt + 1))
    sleep 1
    continue
  fi

  filtered_response="${current_filtered}"
  if "${PYTHON_BIN}" - "${filtered_response}" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
hits = payload.get("hits") or []
if len(hits) < 1:
    raise SystemExit(1)
PY
  then
    break
  fi

  attempt=$((attempt + 1))
  sleep 1
done

if [ -z "${filtered_response}" ]; then
  echo "ERROR: Filtered search did not return a valid payload." >&2
  exit 1
fi
if [ "${attempt}" -ge 60 ]; then
  echo "ERROR: Filtered search did not become ready within timeout. Last payload: ${filtered_response}" >&2
  exit 1
fi

"${PYTHON_BIN}" - "${unfiltered_response}" "${filtered_response}" <<'PY'
import json
import sys

unfiltered = json.loads(sys.argv[1])
filtered = json.loads(sys.argv[2])

unfiltered_hits = unfiltered.get("hits") or []
if len(unfiltered_hits) < 2:
    raise SystemExit(
        f"Expected at least 2 unfiltered hits, got {len(unfiltered_hits)}. Payload: {unfiltered}"
    )
if unfiltered_hits[0].get("id") != "Q312":
    raise SystemExit(f"Expected top unfiltered hit to be Q312. Payload: {unfiltered}")

if int(filtered.get("returned", 0)) != 1:
    raise SystemExit(f"Expected exactly 1 filtered hit. Payload: {filtered}")
filtered_hits = filtered.get("hits") or []
if not filtered_hits or filtered_hits[0].get("id") != "Q312":
    raise SystemExit(f"Expected filtered hit Q312. Payload: {filtered}")

print("Search assertions passed.")
PY

echo "Docker E2E small-sample pipeline completed successfully."
echo "API base URL: ${API_BASE_URL}"
echo "Index ID: ${INDEX_ID}"
