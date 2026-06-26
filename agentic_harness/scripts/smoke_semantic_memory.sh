#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

require_command() {
  local name="$1"
  if ! command -v "$name" >/dev/null 2>&1; then
    echo "Required command '$name' is not available on PATH." >&2
    exit 1
  fi
}

require_command "$PYTHON_BIN"

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Project root: $project_root"
if [[ -n "${AGENTIC_HARNESS_DB_URL:-}" ]]; then
  echo "Runtime ledger DB: $AGENTIC_HARNESS_DB_URL"
else
  echo "Runtime ledger DB: postgresql://postgres:postgres@localhost:5432/agentic_harness"
fi

cd "$project_root"
"$PYTHON_BIN" scripts/smoke_semantic_memory.py
