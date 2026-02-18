#!/usr/bin/env sh

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: Missing required command: $1" >&2
    exit 1
  fi
}

require_docker_ready() {
  require_cmd docker

  if ! docker info >/dev/null 2>&1; then
    echo "ERROR: Docker daemon is not running." >&2
    echo "Start Docker Desktop (or the Docker service) and retry." >&2
    exit 1
  fi

  if ! docker compose version >/dev/null 2>&1; then
    echo "ERROR: 'docker compose' is not available." >&2
    exit 1
  fi
}

compose_service_running() {
  compose_file="$1"
  service="$2"
  docker compose -f "$compose_file" ps --status running --services 2>/dev/null \
    | grep -Fx "$service" >/dev/null 2>&1
}

require_compose_service_running() {
  compose_file="$1"
  service="$2"
  help_cmd="$3"

  if compose_service_running "$compose_file" "$service"; then
    return
  fi

  echo "ERROR: Required Docker service '$service' is not running." >&2
  echo "Run containers first, then retry:" >&2
  echo "  $help_cmd" >&2
  exit 1
}
