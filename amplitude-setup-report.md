<wizard-report>

# Amplitude Setup Report

## Summary

Amplitude analytics has been integrated into the `agentic_harness` Python CLI. Seven workflow lifecycle events are now tracked across all four CLI commands (`run`, `run-agent`, `run-dag`, `resume-dag`).

## What Was Done

### SDK Installed
- `amplitude-analytics>=1.1.0` installed via `pip install amplitude-analytics`
- Added to `agentic_harness/pyproject.toml` dependencies

### New File: `agentic_harness/agentic_harness/analytics.py`
Thin wrapper around the Amplitude Python SDK. Lazily initializes the client from `AMPLITUDE_API_KEY` and no-ops silently if the key is absent. Call `track(event_name, properties)` anywhere in the codebase; call `shutdown()` before process exit to flush the queue.

### Modified: `agentic_harness/agentic_harness/__main__.py`
- Added `import agentic_harness.analytics as _amp`
- Added `_amp.track(...)` calls before/after each CLI command's run function
- Added `_amp.shutdown()` after all command branches to flush events before exit

### Environment Variables (`.env`)
- `AMPLITUDE_API_KEY` — present
- `AMPLITUDE_SERVER_URL` — present (`https://api2.amplitude.com`)

## Events Tracked

| Event | CLI Command | Key Properties |
|---|---|---|
| `Workflow Run Started` | `run` | `workflow_id`, `run_id`, `has_llm_provider` |
| `Workflow Run Completed` | `run`, `resume` | `workflow_id`, `run_id`, `status`, `step_count` |
| `Agent Workflow Started` | `run-agent` | `run_id`, `agent_path` |
| `Agent Workflow Completed` | `run-agent` | `agent_id`, `run_id`, `status`, `step_count` |
| `DAG Workflow Started` | `run-dag` | `run_id`, `auto_approve_gates` |
| `DAG Workflow Completed` | `run-dag`, `resume-dag` | `workflow_id`, `run_id`, `status`, `completed_stage_count` |
| `Human Gate Decided` | `resume-dag` | `run_id`, `node_id`, `decision` |

## Existing Analytics Patterns

No pre-existing analytics instrumentation was found in the codebase. This is the first integration.

## Build Verification

Runtime import verification (`python -c "import ..."`) was blocked by shell policy. The code changes are syntactically correct and the module structure follows the same import patterns used throughout the package. To verify manually after setup:

```bash
cd agentic_harness
python -c "import agentic_harness.analytics; print('ok')"
```

## Notes

- The `user_id` is set to the constant string `"cli"` since the harness has no user authentication layer. Replace with a real user/machine identifier if needed.
- Events will be silently dropped (no exception raised) if `AMPLITUDE_API_KEY` is not set — the harness continues to function normally.
- The `.env` file at the project root is not automatically loaded by the SDK. If you rely on `python-dotenv`, call `load_dotenv()` before any `track()` call, or set the env var in your shell/CI environment directly.

</wizard-report>
