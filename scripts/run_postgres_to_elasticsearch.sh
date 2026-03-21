#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)

cd "${ROOT_DIR}"

echo "Starting required services (Postgres + Elasticsearch + API)"
docker compose up -d postgres elasticsearch api

POSTGRES_DSN="postgresql://postgres@postgres:5432/alpaca"
ELASTICSEARCH_URL="http://elasticsearch:9200"

if [ -t 0 ] && [ -t 1 ]; then
  exec docker compose exec \
    -e ALPACA_POSTGRES_DSN="${POSTGRES_DSN}" \
    -e ALPACA_ELASTICSEARCH_URL="${ELASTICSEARCH_URL}" \
    api \
    python -m src.index_postgres_to_elasticsearch "$@"
fi

exec docker compose exec \
  -T \
  -e ALPACA_POSTGRES_DSN="${POSTGRES_DSN}" \
  -e ALPACA_ELASTICSEARCH_URL="${ELASTICSEARCH_URL}" \
  api \
  python -m src.index_postgres_to_elasticsearch "$@"
