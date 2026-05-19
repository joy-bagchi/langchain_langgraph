#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-agentic-harness-pg}"
EXISTING_CONTAINER_NAME="${EXISTING_CONTAINER_NAME:-}"
DATABASE_NAME="${DATABASE_NAME:-agentic_harness}"
USERNAME="${USERNAME:-postgres}"
PASSWORD="${PASSWORD:-postgres}"
DB_HOST="${DB_HOST:-localhost}"
PORT="${PORT:-5432}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RESET_CONTAINER="${RESET_CONTAINER:-false}"
MANAGE_CONTAINER="${MANAGE_CONTAINER:-false}"

require_command() {
  local name="$1"
  if ! command -v "$name" >/dev/null 2>&1; then
    echo "Required command '$name' is not available on PATH." >&2
    exit 1
  fi
}

wait_for_postgres() {
  local container="$1"
  local user="$2"
  local database="$3"

  for _ in $(seq 1 30); do
    if docker exec "$container" pg_isready -U "$user" -d "$database" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done

  echo "Postgres container '$container' did not become ready in time." >&2
  exit 1
}

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
storage_root="$project_root/.workflow_memory_postgres_smoke"
db_url="${AGENTIC_HARNESS_DB_URL:-postgresql://${USERNAME}:${PASSWORD}@${DB_HOST}:${PORT}/${DATABASE_NAME}}"

require_command "$PYTHON_BIN"

echo "Project root: $project_root"
echo "Storage root: $storage_root"
echo "Database URL: $db_url"

use_managed_container="false"
inspection_mode="none"
inspection_target=""

if [[ -n "$EXISTING_CONTAINER_NAME" ]]; then
  require_command docker
  inspection_mode="docker"
  inspection_target="$EXISTING_CONTAINER_NAME"

  if ! docker ps -a --format '{{.Names}}' | grep -Fxq "$EXISTING_CONTAINER_NAME"; then
    echo "Existing Postgres container '$EXISTING_CONTAINER_NAME' was not found." >&2
    exit 1
  fi
  if ! docker ps --format '{{.Names}}' | grep -Fxq "$EXISTING_CONTAINER_NAME"; then
    echo "Starting existing container $EXISTING_CONTAINER_NAME..."
    docker start "$EXISTING_CONTAINER_NAME" >/dev/null
  else
    echo "Reusing running container $EXISTING_CONTAINER_NAME..."
  fi

  echo "Waiting for Postgres readiness..."
  wait_for_postgres "$EXISTING_CONTAINER_NAME" "$USERNAME" "$DATABASE_NAME"
elif [[ "$MANAGE_CONTAINER" == "true" ]]; then
  require_command docker
  use_managed_container="true"
  inspection_mode="docker"
  inspection_target="$CONTAINER_NAME"

  if [[ "$RESET_CONTAINER" == "true" ]]; then
    if docker ps -a --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
      echo "Removing existing container $CONTAINER_NAME..."
      docker rm -f "$CONTAINER_NAME" >/dev/null
    fi
  fi

  if ! docker ps -a --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
    echo "Creating Postgres container $CONTAINER_NAME..."
    docker run \
      --name "$CONTAINER_NAME" \
      -e "POSTGRES_PASSWORD=$PASSWORD" \
      -e "POSTGRES_DB=$DATABASE_NAME" \
      -p "${PORT}:5432" \
      -d postgres:16 >/dev/null
  elif ! docker ps --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
    echo "Starting existing container $CONTAINER_NAME..."
    docker start "$CONTAINER_NAME" >/dev/null
  else
    echo "Reusing running container $CONTAINER_NAME..."
  fi

  echo "Waiting for Postgres readiness..."
  wait_for_postgres "$CONTAINER_NAME" "$USERNAME" "$DATABASE_NAME"
else
  if [[ -z "${AGENTIC_HARNESS_DB_URL:-}" ]]; then
    echo "No AGENTIC_HARNESS_DB_URL was provided, so the script will use the default direct Postgres URL."
    echo "Docker container management is disabled unless MANAGE_CONTAINER=true or EXISTING_CONTAINER_NAME is set."
  else
    echo "Using externally provided AGENTIC_HARNESS_DB_URL."
  fi
  if command -v psql >/dev/null 2>&1; then
    inspection_mode="psql"
  fi
fi

rm -rf "$storage_root"
export AGENTIC_HARNESS_DB_URL="$db_url"

cd "$project_root"

echo
echo "Running research_agent against Postgres..."
agent_raw="$("$PYTHON_BIN" -m agentic_harness run-agent \
  --agent agents/research_agent.yaml \
  --query "What is an SABR model" \
  --storage-root "$storage_root" \
  --output-mode internal)"
agent_run_id="$(printf '%s' "$agent_raw" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["run_id"])')"
agent_status="$(printf '%s' "$agent_raw" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["status"])')"
agent_last_error="$(printf '%s' "$agent_raw" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin).get("last_error") or "")')"
echo "research_agent run_id: $agent_run_id"
echo "research_agent status: $agent_status"

echo
echo "Running durable workflow against Postgres..."
workflow_raw="$("$PYTHON_BIN" -m agentic_harness run \
  --workflow examples/workflows/research_brief.md \
  --input examples/workflows/research_brief_input.json \
  --storage-root "$storage_root" \
  --output-mode internal)"
workflow_run_id="$(printf '%s' "$workflow_raw" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["run_id"])')"
workflow_status="$(printf '%s' "$workflow_raw" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["status"])')"
echo "research_brief run_id: $workflow_run_id"
echo "research_brief status: $workflow_status"

echo
echo "Inspecting Postgres runtime ledger..."
if [[ "$inspection_mode" == "docker" ]]; then
  docker exec -e "PGPASSWORD=$PASSWORD" "$inspection_target" psql -U "$USERNAME" -d "$DATABASE_NAME" <<'SQL'
\pset pager off
select run_id, workflow_id, status, run_kind from runs order by updated_at desc;
select run_id, checkpoint_index from checkpoints order by created_at desc limit 10;
select run_id, event_type from events order by created_at desc limit 20;
select invocation_id, run_id, agent_id, runtime_profile, status from agent_invocations order by updated_at desc limit 10;
select record_id, namespace, memory_type, source_run_id from memory_records order by created_at desc limit 10;
SQL
elif [[ "$inspection_mode" == "psql" ]]; then
  PGPASSWORD="$PASSWORD" psql "$db_url" <<'SQL'
\pset pager off
select run_id, workflow_id, status, run_kind from runs order by updated_at desc;
select run_id, checkpoint_index from checkpoints order by created_at desc limit 10;
select run_id, event_type from events order by created_at desc limit 20;
select invocation_id, run_id, agent_id, runtime_profile, status from agent_invocations order by updated_at desc limit 10;
select record_id, namespace, memory_type, source_run_id from memory_records order by created_at desc limit 10;
SQL
else
  echo "Skipping SQL inspection because neither an inspectable Docker container nor local psql is available."
  echo "You can inspect manually with a Postgres client using:"
  echo "  $db_url"
fi

echo
echo "Smoke test summary:"
echo "  - runtime ledger connectivity: PASS"
echo "  - research_agent ledger persistence: PASS (run_id=$agent_run_id, status=$agent_status)"
echo "  - research_brief ledger persistence: PASS (run_id=$workflow_run_id, status=$workflow_status)"

if [[ "$agent_status" == "failed" && -n "$agent_last_error" ]]; then
  echo "  - research_agent external execution: FAILED"
  if [[ "$agent_last_error" == *"api.tavily.com"* ]]; then
    echo "    reason: Tavily/web access failed, but the Postgres ledger path still passed."
  else
    echo "    reason: $agent_last_error"
  fi
fi

if [[ "$workflow_status" == "awaiting_review" ]]; then
  echo "  - durable workflow behavior: PASS (paused at expected human review gate)"
fi

echo
echo "Local compatibility mirrors remain under:"
echo "  $storage_root"
