#!/usr/bin/env sh
set -eu

QUICKWIT_URL="${ALPACA_QUICKWIT_URL:-http://localhost:7280}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PREFIX=""
DRY_RUN=0
CONFIRMED=0

usage() {
  cat <<'EOF'
Usage:
  ./scripts/delete_quickwit_indexes.sh --yes [--quickwit-url URL] [--prefix PREFIX]
  ./scripts/delete_quickwit_indexes.sh --dry-run [--quickwit-url URL] [--prefix PREFIX]

Options:
  --yes               Required confirmation for destructive deletion.
  --dry-run           Show indexes that would be deleted, without deleting.
  --quickwit-url URL  Quickwit base URL (default: http://localhost:7280 or ALPACA_QUICKWIT_URL).
  --prefix PREFIX     Delete only indexes with IDs starting with this prefix.
  -h, --help          Show this help.

Examples:
  ./scripts/delete_quickwit_indexes.sh --yes
  ./scripts/delete_quickwit_indexes.sh --yes --prefix wikidata_entities
  ./scripts/delete_quickwit_indexes.sh --dry-run --prefix wikidata_entities_e2e_
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
    --yes)
      CONFIRMED=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --quickwit-url)
      QUICKWIT_URL="$2"
      shift 2
      ;;
    --prefix)
      PREFIX="$2"
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

require_cmd curl
require_cmd "$PYTHON_BIN"

if [ "${DRY_RUN}" -ne 1 ] && [ "${CONFIRMED}" -ne 1 ]; then
  echo "ERROR: Pass --yes to confirm index deletion (or use --dry-run)." >&2
  exit 1
fi

indexes_payload=$(curl -fsS "${QUICKWIT_URL}/api/v1/indexes")

index_ids=$("$PYTHON_BIN" - "${indexes_payload}" "${PREFIX}" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
prefix = sys.argv[2]

ids = []
if isinstance(payload, list):
    for entry in payload:
        if not isinstance(entry, dict):
            continue

        index_id = None
        index_config = entry.get("index_config")
        if isinstance(index_config, dict):
            candidate = index_config.get("index_id")
            if isinstance(candidate, str) and candidate:
                index_id = candidate

        if index_id is None:
            for key in ("index_id", "id"):
                candidate = entry.get(key)
                if isinstance(candidate, str) and candidate:
                    index_id = candidate
                    break

        if not index_id:
            continue

        if prefix and not index_id.startswith(prefix):
            continue

        ids.append(index_id)

seen = set()
for index_id in ids:
    if index_id in seen:
        continue
    seen.add(index_id)
    print(index_id)
PY
)

if [ -z "${index_ids}" ]; then
  if [ -n "${PREFIX}" ]; then
    echo "No Quickwit indexes matched prefix '${PREFIX}'."
  else
    echo "No Quickwit indexes found."
  fi
  exit 0
fi

echo "Quickwit indexes selected for deletion (${QUICKWIT_URL}):"
printf '%s\n' "${index_ids}" | sed 's/^/  - /'

if [ "${DRY_RUN}" -eq 1 ]; then
  echo "Dry-run only. Nothing deleted."
  exit 0
fi

deleted=0
failed=0

for index_id in ${index_ids}; do
  encoded_id=$("$PYTHON_BIN" - "${index_id}" <<'PY'
import sys
from urllib.parse import quote

print(quote(sys.argv[1], safe=""))
PY
)

  response=$(curl -sS -X DELETE -w '\nHTTP_STATUS:%{http_code}\n' "${QUICKWIT_URL}/api/v1/indexes/${encoded_id}")
  status=$(printf '%s\n' "${response}" | sed -n 's/^HTTP_STATUS://p')
  body=$(printf '%s\n' "${response}" | sed '/^HTTP_STATUS:/d')

  case "${status}" in
    2??)
      deleted=$((deleted + 1))
      echo "Deleted index: ${index_id}"
      ;;
    *)
      failed=$((failed + 1))
      echo "ERROR: Failed to delete index '${index_id}' (HTTP ${status})." >&2
      if [ -n "${body}" ]; then
        echo "  ${body}" >&2
      fi
      ;;
  esac
done

echo "Done. deleted=${deleted} failed=${failed}"

if [ "${failed}" -gt 0 ]; then
  exit 1
fi
