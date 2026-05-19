#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="${PROJECT_NAME:-agentic_harness}"
AGENT_PATH="${AGENT_PATH:-agents/research_analyst.yaml}"
INPUT_PATH="${INPUT_PATH:-examples/workflows/research_brief_input.json}"
STORAGE_ROOT="${STORAGE_ROOT:-.workflow_memory_langsmith_smoke}"
RECENT_WINDOW_SECONDS="${RECENT_WINDOW_SECONDS:-180}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

require_command() {
  local name="$1"
  if ! command -v "$name" >/dev/null 2>&1; then
    echo "Required command '$name' is not available on PATH." >&2
    exit 1
  fi
}

require_command "$PYTHON_BIN"

if [[ -z "${LANGSMITH_API_KEY:-}" ]]; then
  echo "LANGSMITH_API_KEY is not set." >&2
  exit 1
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$STORAGE_ROOT" = /* ]]; then
  resolved_storage_root="$STORAGE_ROOT"
else
  resolved_storage_root="$project_root/$STORAGE_ROOT"
fi

started_at="$("$PYTHON_BIN" -c 'from datetime import datetime, timezone; print(datetime.now(timezone.utc).isoformat())')"

echo "Project root: $project_root"
echo "LangSmith project: $PROJECT_NAME"
echo "Storage root: $resolved_storage_root"
if [[ -n "${AGENTIC_HARNESS_DB_URL:-}" ]]; then
  echo "Runtime ledger DB: $AGENTIC_HARNESS_DB_URL"
else
  echo "Runtime ledger DB: default local SQLite under storage root"
fi

rm -rf "$resolved_storage_root"

cd "$project_root"

export LANGSMITH_TRACING=true
export LANGSMITH_PROJECT="$PROJECT_NAME"

echo
echo "Running LangSmith-traced agent smoke test..."
run_raw="$("$PYTHON_BIN" -m agentic_harness run-agent \
  --agent "$AGENT_PATH" \
  --input "$INPUT_PATH" \
  --storage-root "$resolved_storage_root" \
  --langsmith-tracing \
  --langsmith-project "$PROJECT_NAME" \
  --output-mode internal)"

run_status="$(printf '%s' "$run_raw" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["status"])')"
run_id="$(printf '%s' "$run_raw" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["run_id"])')"

verification_raw="$("$PYTHON_BIN" - <<PY
from datetime import datetime, timedelta, timezone
import json
from langsmith import Client

project_name = ${PROJECT_NAME@Q}
window_seconds = int(${RECENT_WINDOW_SECONDS@Q})
started_at = datetime.fromisoformat(${started_at@Q})
threshold = started_at - timedelta(seconds=window_seconds)

client = Client()
runs = list(client.list_runs(project_name=project_name, limit=50))

def to_utc(value):
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)

recent = []
for run in runs:
    start_time = to_utc(getattr(run, "start_time", None))
    if start_time is None or start_time >= threshold:
        recent.append(
            {
                "id": str(getattr(run, "id", "")),
                "name": getattr(run, "name", ""),
                "run_type": getattr(run, "run_type", ""),
                "status": getattr(run, "status", ""),
                "start_time": start_time.isoformat() if start_time else None,
            }
        )

print(json.dumps({"recent_count": len(recent), "recent_runs": recent[:10]}, indent=2))
PY
)"

recent_count="$(printf '%s' "$verification_raw" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["recent_count"])')"

echo
echo "LangSmith smoke summary:"
echo "  - agent run status: $run_status"
echo "  - harness run_id: $run_id"
echo "  - LangSmith project: $PROJECT_NAME"
echo "  - recent LangSmith runs found: $recent_count"

if [[ "$recent_count" -le 0 ]]; then
  echo "LangSmith did not return any recent runs for project '$PROJECT_NAME'." >&2
  exit 1
fi

echo
echo "Recent LangSmith runs:"
printf '%s' "$verification_raw" | "$PYTHON_BIN" -c 'import json,sys; data=json.load(sys.stdin); [print(f"  - {run['\''id'\'']} | {run['\''name'\'']} | {run['\''run_type'\'']} | {run['\''status'\'']}") for run in data["recent_runs"]]'
