#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ALPACA_ENV_FILE:-${ROOT_DIR}/.env}"
TYPE_QID="${ALPACA_NER_REFRESH_TYPE_QID:-Q5}"
WORKERS="${ALPACA_NER_REFRESH_WORKERS:-8}"
CLASSIFY_BATCH_SIZE="${ALPACA_NER_REFRESH_BATCH_SIZE:-20000}"
ES_BATCH_SIZE="${ALPACA_NER_REFRESH_ES_BATCH_SIZE:-20000}"
ES_BULK_ACTIONS="${ALPACA_NER_REFRESH_ES_BULK_ACTIONS:-2000}"
ES_WORKERS="${ALPACA_NER_REFRESH_ES_WORKERS:-4}"
ES_INDEX="${ALPACA_FULL_ES_INDEX:-alpaca-wikidata}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing environment file: ${ENV_FILE}" >&2
  exit 1
fi

compose=(
  docker compose
  --env-file "${ENV_FILE}"
  -f "${ROOT_DIR}/docker-compose.yml"
)
if [[ -f "${ROOT_DIR}/compose.cloudflare.yml" ]] && docker network inspect cloudflare >/dev/null 2>&1; then
  compose+=(-f "${ROOT_DIR}/compose.cloudflare.yml")
fi

cd "${ROOT_DIR}"

echo "Building Alpaca with wikidata-ner-classifier==0.8.0..."
"${compose[@]}" build api

echo "Phase 1/2: refresh ${TYPE_QID} classifications in PostgreSQL"
"${compose[@]}" run --rm --no-deps api \
  python -m src.classify_postgres_entities \
  --postgres-dsn postgresql://postgres@postgres:5432/alpaca \
  --required-type-qid "${TYPE_QID}" \
  --batch-size "${CLASSIFY_BATCH_SIZE}" \
  --workers "${WORKERS}"

echo "Phase 2/2: update only NER fields in Elasticsearch"
"${compose[@]}" run --rm --no-deps api \
  python -m src.update_elasticsearch_ner \
  --postgres-dsn postgresql://postgres@postgres:5432/alpaca \
  --elasticsearch-url http://elasticsearch:9200 \
  --index-name "${ES_INDEX}" \
  --required-type-qid "${TYPE_QID}" \
  --batch-size "${ES_BATCH_SIZE}" \
  --bulk-actions "${ES_BULK_ACTIONS}" \
  --workers "${ES_WORKERS}"

echo "Refreshing the long-running Alpaca API container with the new library..."
"${compose[@]}" up -d --no-deps api

echo "NER refresh completed successfully: classifier=0.8.0 type=${TYPE_QID} index=${ES_INDEX}"
