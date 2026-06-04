#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <label> <command> [args...]" >&2
  exit 2
fi

label="$1"
shift

heartbeat_seconds="${GB10_BUILD_HEARTBEAT_SECONDS:-300}"
if [[ ! "$heartbeat_seconds" =~ ^[1-9][0-9]*$ ]]; then
  echo "GB10_BUILD_HEARTBEAT_SECONDS must be a positive integer." >&2
  exit 2
fi

timestamp() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

print_resource_snapshot() {
  echo "GB10 ${label} resource snapshot at $(timestamp)"
  df -h || true
  if [ "${GB10_BUILD_HEARTBEAT_DOCKER_DF:-0}" = "1" ] \
    && command -v docker >/dev/null 2>&1; then
    docker system df || true
  fi
}

child_pid=""
heartbeat_pid=""

cleanup() {
  status="$?"
  if [ -n "${heartbeat_pid:-}" ]; then
    kill "$heartbeat_pid" 2>/dev/null || true
  fi
  if [ -n "${child_pid:-}" ]; then
    kill "$child_pid" 2>/dev/null || true
  fi
  exit "$status"
}

trap cleanup INT TERM

echo "GB10 ${label} starting at $(timestamp)"
print_resource_snapshot

"$@" &
child_pid="$!"

(
  while true; do
    sleep "$heartbeat_seconds"
    if kill -0 "$child_pid" 2>/dev/null; then
      echo "GB10 ${label} still running at $(timestamp)"
      print_resource_snapshot
    else
      exit 0
    fi
  done
) &
heartbeat_pid="$!"

set +e
wait "$child_pid"
status="$?"
set -e

kill "$heartbeat_pid" 2>/dev/null || true
wait "$heartbeat_pid" 2>/dev/null || true
trap - INT TERM

if [ "$status" -eq 0 ]; then
  echo "GB10 ${label} finished at $(timestamp)"
else
  echo "GB10 ${label} failed with exit code ${status} at $(timestamp)" >&2
  print_resource_snapshot
fi

exit "$status"
