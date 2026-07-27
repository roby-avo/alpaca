#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)

ALPACA_FULL_DUMP_PATH="${ALPACA_FULL_DUMP_PATH:-/home/roby-avo/latest-all.json.bz2}"
ALPACA_EXPECTED_ENTITY_TOTAL="${ALPACA_EXPECTED_ENTITY_TOTAL:-110000000}"
ALPACA_FULL_MIN_FREE_GB="${ALPACA_FULL_MIN_FREE_GB:-80}"
ALPACA_FULL_WORKERS="${ALPACA_FULL_WORKERS:-8}"
ALPACA_FULL_PG_BATCH_SIZE="${ALPACA_FULL_PG_BATCH_SIZE:-5000}"
ALPACA_FULL_NER_BATCH_SIZE="${ALPACA_FULL_NER_BATCH_SIZE:-20000}"
ALPACA_FULL_ES_INDEX="${ALPACA_FULL_ES_INDEX:-alpaca-wikidata}"

if [ ! -f "${ALPACA_FULL_DUMP_PATH}" ]; then
  echo "ERROR: Wikidata dump not found: ${ALPACA_FULL_DUMP_PATH}" >&2
  exit 1
fi

cd "${ROOT_DIR}"
mkdir -p data/output data/postgres data/elasticsearch

AVAILABLE_KB=$(df -Pk "${ROOT_DIR}" | awk 'NR == 2 {print $4}')
REQUIRED_KB=$((ALPACA_FULL_MIN_FREE_GB * 1024 * 1024))
if [ "${AVAILABLE_KB}" -lt "${REQUIRED_KB}" ]; then
  echo "ERROR: only $((AVAILABLE_KB / 1024 / 1024)) GiB free; at least ${ALPACA_FULL_MIN_FREE_GB} GiB is required." >&2
  exit 1
fi

if docker info >/dev/null 2>&1; then
  DOCKER_PREFIX=""
elif sudo -n docker info >/dev/null 2>&1; then
  DOCKER_PREFIX="sudo"
else
  echo "ERROR: Docker is not accessible directly or through passwordless sudo." >&2
  exit 1
fi

# Elasticsearch runs as uid 1000 in the official image.
if sudo -n true >/dev/null 2>&1; then
  sudo chown -R 1000:0 data/elasticsearch
else
  chmod 0777 data/elasticsearch
fi

docker_compose() {
  if [ -n "${DOCKER_PREFIX}" ]; then
    sudo -E docker compose "$@"
  else
    docker compose "$@"
  fi
}

export ALPACA_INPUT_DIR
ALPACA_INPUT_DIR=$(dirname "${ALPACA_FULL_DUMP_PATH}")
export ALPACA_DUMP_PATH="/mnt/input/$(basename "${ALPACA_FULL_DUMP_PATH}")"
export ALPACA_POSTGRES_DATA_DIR="${ROOT_DIR}/data/postgres"
export ALPACA_ELASTIC_DATA_DIR="${ROOT_DIR}/data/elasticsearch"
export ALPACA_OUTPUT_DIR="${ROOT_DIR}/data/output"
export ALPACA_POSTGRES_SHM_SIZE="${ALPACA_POSTGRES_SHM_SIZE:-8GB}"
export ALPACA_POSTGRES_SHARED_BUFFERS="${ALPACA_POSTGRES_SHARED_BUFFERS:-8GB}"
export ALPACA_POSTGRES_MAINTENANCE_WORK_MEM="${ALPACA_POSTGRES_MAINTENANCE_WORK_MEM:-1GB}"
export ALPACA_POSTGRES_MAX_WAL_SIZE="${ALPACA_POSTGRES_MAX_WAL_SIZE:-8GB}"
export ALPACA_ELASTIC_HEAP_SIZE="${ALPACA_ELASTIC_HEAP_SIZE:-16g}"

POSTGRES_DSN="postgresql://postgres@postgres:5432/alpaca"
ELASTICSEARCH_URL="http://elasticsearch:9200"
INGEST_MARKER="${ROOT_DIR}/data/output/.full-wikidata-ingest-complete"
NER_MARKER="${ROOT_DIR}/data/output/.full-wikidata-ner-complete"
ES_STARTED_MARKER="${ROOT_DIR}/data/output/.full-wikidata-es-started"
ES_COMPLETE_MARKER="${ROOT_DIR}/data/output/.full-wikidata-es-complete"

echo "Full Wikidata pipeline configuration:"
echo "  dump=${ALPACA_FULL_DUMP_PATH}"
echo "  expected_entities~${ALPACA_EXPECTED_ENTITY_TOTAL}"
echo "  workers=${ALPACA_FULL_WORKERS}"
echo "  elasticsearch_index=${ALPACA_FULL_ES_INDEX}"
echo "  free_space=$((AVAILABLE_KB / 1024 / 1024)) GiB"
echo "  postgres_data=${ALPACA_POSTGRES_DATA_DIR}"
echo "  elasticsearch_data=${ALPACA_ELASTIC_DATA_DIR}"

echo "Building the pinned Alpaca runtime and starting Postgres..."
docker_compose build api
docker_compose up -d --wait postgres

echo "Preparing the wikidata-ner-classifier hierarchy schema..."
docker_compose run --rm -T --no-deps \
  -e ALPACA_POSTGRES_DSN="${POSTGRES_DSN}" \
  api \
  python -m src.classify_postgres_entities \
    --postgres-dsn "${POSTGRES_DSN}" \
    --prepare-only

if [ ! -f "${INGEST_MARKER}" ]; then
  echo "Phase 1/3: stream the compressed dump into lean Postgres storage"
  docker_compose run --rm -T --no-deps \
    -e ALPACA_POSTGRES_DSN="${POSTGRES_DSN}" \
    -e ALPACA_DUMP_PATH="${ALPACA_DUMP_PATH}" \
    api \
    python -m src.build_postgres_entities \
      --dump-path "${ALPACA_DUMP_PATH}" \
      --postgres-dsn "${POSTGRES_DSN}" \
      --expected-entity-total "${ALPACA_EXPECTED_ENTITY_TOTAL}" \
      --batch-size "${ALPACA_FULL_PG_BATCH_SIZE}" \
      --workers "${ALPACA_FULL_WORKERS}" \
      --languages en,mul \
      --storage-languages en,mul \
      --max-aliases-per-language 4 \
      --disable-ner-classifier \
      --disable-entity-triples \
      --include-unlabeled
  touch "${INGEST_MARKER}"
else
  echo "Phase 1/3: already complete (${INGEST_MARKER})"
fi

if [ ! -f "${NER_MARKER}" ]; then
  echo "Phase 2/3: classify every Q-item from resolved P31/P106/P279 labels"
  docker_compose run --rm -T --no-deps \
    -e ALPACA_POSTGRES_DSN="${POSTGRES_DSN}" \
    api \
    python -m src.classify_postgres_entities \
      --postgres-dsn "${POSTGRES_DSN}" \
      --batch-size "${ALPACA_FULL_NER_BATCH_SIZE}" \
      --workers "${ALPACA_FULL_WORKERS}"
  touch "${NER_MARKER}"
else
  echo "Phase 2/3: already complete (${NER_MARKER})"
fi

if [ ! -f "${ES_COMPLETE_MARKER}" ]; then
  echo "Phase 3/3: bulk index Postgres + NER results into Elasticsearch"
  docker_compose up -d --wait elasticsearch
  RECREATE_ARG=""
  if [ ! -f "${ES_STARTED_MARKER}" ]; then
    RECREATE_ARG="--recreate-index"
    touch "${ES_STARTED_MARKER}"
  fi
  # Word splitting is intentional for the optional single flag.
  # shellcheck disable=SC2086
  docker_compose run --rm -T --no-deps \
    -e ALPACA_POSTGRES_DSN="${POSTGRES_DSN}" \
    -e ALPACA_ELASTICSEARCH_URL="${ELASTICSEARCH_URL}" \
    api \
    python -m src.index_postgres_to_elasticsearch \
      --postgres-dsn "${POSTGRES_DSN}" \
      --elasticsearch-url "${ELASTICSEARCH_URL}" \
      --index-name "${ALPACA_FULL_ES_INDEX}" \
      --batch-size 10000 \
      --bulk-actions 2000 \
      --workers 4 \
      --max-inflight 8 \
      --max-indexed-labels 4 \
      --max-indexed-aliases 8 \
      --max-context-chars 0 \
      ${RECREATE_ARG}
  touch "${ES_COMPLETE_MARKER}"
else
  echo "Phase 3/3: already complete (${ES_COMPLETE_MARKER})"
fi

echo "Full Wikidata pipeline completed successfully."
