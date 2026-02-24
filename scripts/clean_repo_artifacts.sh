#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)

CONFIRMED=0

usage() {
  cat <<'EOF'
Usage:
  ./scripts/clean_repo_artifacts.sh --yes

Options:
  --yes       Required confirmation for destructive cleanup.
  -h, --help  Show this help.

Removes generated artifacts under:
  data/input/* (except .gitkeep)
  data/output/* (except .gitkeep)
  Python/tool caches (__pycache__, .pytest_cache, .mypy_cache, .ruff_cache)
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --yes)
      CONFIRMED=1
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

if [ "${CONFIRMED}" -ne 1 ]; then
  echo "ERROR: Pass --yes to confirm cleanup." >&2
  exit 1
fi

clean_dir_keep_gitkeep() {
  target_dir="$1"
  if [ ! -d "${target_dir}" ]; then
    return
  fi
  find "${target_dir}" -mindepth 1 ! -name '.gitkeep' -exec rm -rf {} +
}

clean_dir_keep_gitkeep "${ROOT_DIR}/data/input"
clean_dir_keep_gitkeep "${ROOT_DIR}/data/output"

find "${ROOT_DIR}" -type d \( -name '__pycache__' -o -name '.pytest_cache' -o -name '.mypy_cache' -o -name '.ruff_cache' \) -prune -exec rm -rf {} +

echo "Cleanup completed."
