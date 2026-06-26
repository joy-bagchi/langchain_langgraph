# Tutor Context for Future Sessions

## Agreed Working Relationship

You want to learn LangChain and LangGraph by implementing the coursework yourself.

My role is to act as your instructor and reviewer, not to pre-build lessons unless you explicitly ask.

## How We Will Work

1. You implement each day’s deliverable from the syllabus.
2. You share the file(s) or ask for a review.
3. I provide:
   - concept check (what is understood vs missing),
   - code review (bugs, risks, design quality),
   - actionable improvements (priority ordered),
   - optional stretch exercise.

## Review Template I Should Use

- Correctness
- Framework usage
- Code quality
- Testing
- Next step

## Current Starting Point

- Syllabus and module alignment are in `README.md`.
- Python scaffold files are already created.
- Project config and smoke test are in `pyproject.toml` and `tests/test_smoke_imports.py`.

## First Planned Review

When ready, review `src/01_hello_chat.py` (Week 1 Day 1).

## Session Log - 2026-03-03

- Completed and reviewed `src/01_hello_chat.py` (Week 1 Day 1).
- Confirmed Day 1 objective: basic model call + message primitives + response metadata.
- Completed and reviewed `src/02_prompt_templates.py` (Week 1 Day 2).
- Updated Day 2 code to use explicit runtime validation (no `assert` checks).
- Started and reviewed `src/03_output_parsing.py` (Week 1 Day 3).
- Clarified LCEL wiring:
  - Use one chat prompt + `with_structured_output(ModelResponse)`.
  - Put system instructions in `ChatPromptTemplate.from_messages(...)`.
- Clarified typing warning:
  - IDE may infer `BaseModel`; use explicit annotation/cast to `ModelResponse`.
- Clarified structured output behavior:
  - `with_structured_output(ModelResponse)` drives model output into schema fields.

## Session Log - 2026-03-04

- Completed final polish and review for `src/03_output_parsing.py` (Week 1 Day 3):
  - removed unused imports,
  - constrained `confidence` to `[0, 1]`,
  - clarified prompt instructions for structured fields,
  - kept parse-error handling path.
- Completed and reviewed `src/04_chains.py` (Week 1 Day 4):
  - composed reusable runnable chain (`prompt -> model -> parser`),
  - added `build_chain(...)` and `run_lesson_example(...)`,
  - resolved indentation/syntax issue and passed `py_compile`.
- Added and completed bonus exercise `src/04b_streaming_exercise.py`:
  - streamed model output live to console,
  - captured streamed chunks into a final string,
  - reviewed for style and cleanup notes.
- Confirmed next syllabus target is Week 1 Day 5 integration lab.

## Where To Resume Next Session

- Start Week 1 Day 5 integration lab:
  - implement `src/apps/prompt_lab.py` as a CLI-style prompt app,
  - reuse prompt templates + chain composition patterns from Days 1-4,
  - add smoke test in `tests/test_week1_prompt_lab.py`.
- Suggested Day 5 sequence:
  1. define app inputs and validation,
  2. build reusable chain invocation path,
  3. implement CLI runner,
  4. add and run smoke test.

## Session Log - 2026-03-07

- Completed and reviewed Week 1 Day 5 integration lab:
  - implemented `src/apps/prompt_lab.py` end-to-end:
    - mode-specific prompts (`explain`, `quiz`),
    - reusable chain composition (`prompt -> model -> parser`),
    - topic validation + single-run invocation path,
    - CLI entrypoint behavior with argument parsing.
  - expanded `tests/test_week1_prompt_lab.py` smoke coverage:
    - parser defaults and mode validation,
    - prompt mode behavior,
    - `run_once(...)` invocation payload contract via monkeypatch,
    - empty-topic validation path.
- Validation:
  - `python -m pytest langchain_langgraph_tutorial/tests/test_week1_prompt_lab.py`
  - Result: `7 passed`.

## Where To Resume Next Session

- Start Week 2 Day 1 (Module 1: Simple Graph):
  - implement `src/graph/01_state_and_nodes.py`,
  - define typed shared graph state,
  - build and run a 2-node linear graph,
  - add quick behavior checks before moving to conditional routing.

## Session Log - 2026-03-08

- Reviewed Week 2 Day 1 status:
  - confirmed `src/graph/01_state_and_nodes.py` is implemented and stable.
- Completed and reviewed Week 2 Day 2 (`src/graph/02_conditional_routing.py`):
  - added typed shared state with reducer-backed fields via `Annotated[..., add]`,
  - implemented conditional routing (`math` vs `general`) using `add_conditional_edges`,
  - added deterministic branch nodes and finalization node,
  - added reusable `build_graph()` and `run_query(...)` helpers.
- Added focused Week 2 tests in `tests/test_week2_state_and_nodes.py`:
  - Day 1 node behavior checks,
  - Day 2 conditional routing checks for both paths,
  - reducer/diagnostic trace validation.
- Validation:
  - `python -m pytest langchain_langgraph_tutorial/tests/test_week2_state_and_nodes.py`
  - Result: `3 passed`.

## Where To Resume Next Session

- Start Week 2 Day 3 (iteration loops + termination safeguards):
  - implement `src/graph/03_revision_loop.py`,
  - add max-iteration guard to prevent infinite loops,
  - capture per-iteration diagnostics,
  - write run artifact to `outputs/week4_loops.json`.
