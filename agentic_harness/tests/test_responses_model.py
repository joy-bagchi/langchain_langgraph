"""Offline contract tests for the Responses model adapter."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from agentic_harness.agentic_os.tool_service import (
    RegisteredToolService,
    ToolDefinition,
    ToolExecutionResponse,
)
from agentic_harness.contracts import WorkflowStep
from agentic_harness.llm import (
    LLMConfig,
    ResponsesExecutionError,
    build_model_callable,
    resolve_llm_config,
)
from agentic_harness import build_platform_services, inspect_run, start_workflow


class FakeResponses:
    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        return SimpleNamespace(
            id=f"resp_{len(self.requests)}", status="completed",
            output=self.outputs.pop(0), created_at=1790733600,
            usage={"input_tokens": 10, "output_tokens": 5},
        )


def model_with(*outputs, rounds=4):
    responses = FakeResponses(*outputs)
    model = build_model_callable(
        LLMConfig(provider="openai", model="gpt-6-astra", api="responses",
                  reasoning_effort="low", max_tool_rounds=rounds),
        client=SimpleNamespace(responses=responses),
    )
    return model, responses


def step():
    return WorkflowStep(step_id="answer", title="Answer", step_type="prompt", prompt="Answer")


def state(allowed=None):
    return {"run_id": "run-1", "workflow_id": "workflow-1", "allowed_tools": allowed or [],
            "step_history": []}


def tool_service(status="succeeded"):
    calls = []

    def handler(request):
        calls.append(request)
        return ToolExecutionResponse(status=status, output={"answer": 42},
                                     metadata={"reason": "provider failed"} if status != "succeeded" else {})

    return RegisteredToolService(
        definitions=[ToolDefinition(tool_id="lookup", name="Lookup", description="Look up a value",
                                    input_schema={"type": "object", "properties": {"query": {"type": "string"}}})],
        handlers={"lookup": handler},
    ), calls


def test_text_response_preserves_reasoning_and_replays_next_step():
    model, api = model_with([{"type": "reasoning", "id": "rs_1", "summary": []},
                             {"type": "message", "content": [{"type": "output_text", "text": "Hello"}]}],
                            [{"type": "message", "content": [{"type": "output_text", "text": "Again"}]}])
    first = model.execute_prompt("Hi", step(), state(), None)
    assert first.output == "Hello"
    assert "temperature" not in api.requests[0]
    assert api.requests[0]["reasoning"] == {"effort": "low"}
    assert first.metadata["responses"]["transcript"][1]["type"] == "reasoning"
    next_state = state()
    next_state["step_history"] = [{"metadata": first.metadata}]
    model.execute_prompt("Continue", step(), next_state, None)
    assert [item["type"] for item in api.requests[1]["input"][1:3]] == ["reasoning", "message"]
    assert api.requests[1]["input"][-1] == {"role": "user", "content": "Continue"}


def test_one_tool_round_trip_uses_registered_service_and_matching_call_id():
    call = {"type": "function_call", "id": "fc_1", "call_id": "call_1",
            "name": "lookup", "arguments": '{"query":"x"}'}
    model, api = model_with([{"type": "reasoning", "id": "rs_1", "summary": []}, call],
                            [{"type": "message", "content": [{"type": "output_text", "text": "42"}]}])
    tools, calls = tool_service()
    result = model.execute_prompt("Find x", step(), state(["lookup"]), tools)
    assert result.output == "42"
    assert calls[0].tool_id == "lookup"
    assert calls[0].arguments == {"query": "x"}
    assert calls[0].metadata["call_id"] == "call_1"
    assert api.requests[1]["input"][-1]["call_id"] == "call_1"
    assert api.requests[1]["input"][-1]["type"] == "function_call_output"
    assert result.metadata["responses"]["tool_rounds"] == 1


def test_unauthorized_tool_request_fails_without_executing():
    model, api = model_with([{"type": "function_call", "call_id": "call_1",
                             "name": "lookup", "arguments": "{}"}])
    tools, calls = tool_service()
    with pytest.raises(ResponsesExecutionError, match="not allowed") as error:
        model.execute_prompt("Find x", step(), state(), tools)
    assert calls == []
    assert "tools" not in api.requests[0]
    assert error.value.metadata["responses"]["transcript"][-1]["type"] == "function_call"


def test_failed_tool_records_call_result_and_stops():
    model, api = model_with([{"type": "function_call", "call_id": "call_1",
                             "name": "lookup", "arguments": "{}"}])
    tools, calls = tool_service(status="error")
    with pytest.raises(ResponsesExecutionError, match="provider failed") as error:
        model.execute_prompt("Find x", step(), state(["lookup"]), tools)
    assert len(calls) == 1
    transcript = error.value.metadata["responses"]["transcript"]
    assert transcript[-1]["type"] == "function_call_output"
    assert transcript[-1]["call_id"] == "call_1"
    assert len(api.requests) == 1


def test_malformed_arguments_and_round_limit_fail_clearly():
    model, _ = model_with([{"type": "function_call", "call_id": "call_1",
                           "name": "lookup", "arguments": "{"}])
    tools, calls = tool_service()
    with pytest.raises(ResponsesExecutionError, match="Malformed arguments"):
        model.execute_prompt("Find x", step(), state(["lookup"]), tools)
    assert calls == []
    model, _ = model_with([{"type": "function_call", "call_id": "call_1",
                           "name": "lookup", "arguments": "{}"}], rounds=0)
    with pytest.raises(ResponsesExecutionError, match="exceeded 0 tool rounds"):
        model.execute_prompt("Find x", step(), state(["lookup"]), tools)


def test_multiple_calls_keep_their_own_call_ids():
    model, api = model_with([
        {"type": "function_call", "call_id": "call_a", "name": "lookup", "arguments": '{"query":"a"}'},
        {"type": "function_call", "call_id": "call_b", "name": "lookup", "arguments": '{"query":"b"}'},
    ], [{"type": "message", "content": [{"type": "output_text", "text": "Done"}]}])
    tools, calls = tool_service()
    result = model.execute_prompt("Find both", step(), state(["lookup"]), tools)
    assert [call.arguments["query"] for call in calls] == ["a", "b"]
    assert [item["call_id"] for item in api.requests[1]["input"]
            if item.get("type") == "function_call_output"] == ["call_a", "call_b"]
    assert result.metadata["responses"]["tool_calls"] == 2
    assert result.metadata["responses"]["turns"][0]["usage"]["input_tokens"] == 10


def test_incomplete_and_refused_responses_fail_explicitly():
    model, _ = model_with([])
    model.client.responses.create = lambda **kwargs: SimpleNamespace(
        id="resp_1", status="incomplete", created_at=1790733600, usage=None, output=[])
    with pytest.raises(ResponsesExecutionError, match="status incomplete"):
        model.execute_prompt("Hi", step(), state(), None)
    model, _ = model_with([{"type": "message", "content":
                            [{"type": "refusal", "refusal": "Cannot help"}]}])
    with pytest.raises(ResponsesExecutionError, match="refused"):
        model.execute_prompt("Hi", step(), state(), None)


def test_explicit_mode_leaves_legacy_default_intact():
    assert resolve_llm_config(provider="openai", model="gpt-4o-mini").api == "chat"
    config = resolve_llm_config(provider="openai", model="gpt-6-astra", api="responses",
                                reasoning_effort="low")
    assert (config.api, config.reasoning_effort) == ("responses", "low")
    with pytest.raises(ValueError, match="supports low"):
        resolve_llm_config(provider="openai", model="gpt-6-astra", api="responses",
                           reasoning_effort="none")


def test_runtime_ledger_retains_typed_items(tmp_path):
    workflow = tmp_path / "response.md"
    workflow.write_text("""---
workflow_id: response_test
title: Response Test
entry_step: answer
memory_namespace: response_test
---

## Step: answer
```yaml
type: prompt
id: answer
output_key: answer
memory:
  enabled: false
```

```prompt
Answer this.
```
""", encoding="utf-8")
    model, _ = model_with([{"type": "reasoning", "id": "rs_1", "summary": []},
                           {"type": "message", "content": [{"type": "output_text", "text": "Done"}]}])
    storage = tmp_path / "ledger"
    result = start_workflow(workflow, {}, storage_root=storage, model_callable=model,
                            langsmith_tracing=False)
    saved = inspect_run(result["run_id"], storage_root=storage)
    assert saved["status"] == "completed"
    assert saved["step_outputs"]["answer"] == "Done"
    transcript = saved["step_history"][-1]["metadata"]["responses"]["transcript"]
    assert [item["type"] for item in transcript[1:]] == ["reasoning", "message"]


def test_completed_run_replay_reads_ledger_without_model_or_search(tmp_path):
    workflow = tmp_path / "response.md"
    workflow.write_text("""---
workflow_id: replay_test
title: Replay Test
entry_step: answer
memory_namespace: replay_test
---

## Step: answer
```yaml
type: prompt
id: answer
output_key: answer
memory:
  enabled: false
```

```prompt
Find a source.
```
""", encoding="utf-8")
    model, api = model_with(
        [{"type": "reasoning", "id": "rs_1", "encrypted_content": "encrypted"},
         {"type": "function_call", "call_id": "call_search", "name": "web_search",
          "arguments": '{"query":"example"}'}],
        [{"type": "message", "content": [{"type": "output_text", "text": "Found it"}]}],
    )

    class SearchStub:
        calls = 0

        def search(self, **kwargs):
            self.calls += 1
            return {"results": [{"url": "https://example.org", "title": "Example",
                                 "published_date": "2026-09-29"}]}

    search = SearchStub()
    storage = tmp_path / "ledger"
    services = build_platform_services(storage_root=storage, model_callable=model,
                                       web_search_client=search, langsmith_tracing=False)
    result = start_workflow(workflow, {}, storage_root=storage, services=services,
                            initial_state_overrides={"allowed_tools": ["web_search"]})
    recorded = inspect_run(result["run_id"], storage_root=storage)
    replayed = inspect_run(result["run_id"], storage_root=storage)
    assert recorded == replayed
    assert len(api.requests) == 2 and search.calls == 1
    metadata = recorded["step_history"][-1]["metadata"]["responses"]
    assert metadata["transcript"][1]["encrypted_content"] == "encrypted"
    tool_result = next(item for item in metadata["transcript"]
                       if item.get("type") == "function_call_output")
    assert tool_result["call_id"] == "call_search"
    assert "https://example.org" in tool_result["output"]
    assert metadata["turns"][0]["status"] == "completed"
