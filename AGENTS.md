# Repository Guidelines

## Project Structure & Module Organization
- `agentic_rag/` hosts the runnable LangChain + FAISS pipeline; start from `agentic_rag.py`.
- `building_workflows_with_langgraph/` and `building_intelligent_rag_systems/` collect LangGraph tutorials and policy demos alongside assets such as `companyPolicies.txt` or `working_with_short_context_window/`.
- `coursera/` contains course labs; notebooks depend on helpers (`utils.py`, `news_data_dedup.csv`) and graders in `Introduction to RAG/`.
- `crewai/`, `deep_agents/`, and root scripts like `react_agent.py` are standalone agent experiments.
- Shared datasets (`classification_dataset.csv`, `regression-dataset.csv`) live in the root; exploratory workspaces sit under `sandbox/` and `Advanced-Deep-Learning-with-Keras/`.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` creates an isolated environment.
- `pip install -U langchain langchain-openai langgraph bm25s crewai python-dotenv` installs the common stack; add `pip install -r Advanced-Deep-Learning-with-Keras/requirements.txt` when touching that workspace.
- `python agentic_rag/agentic_rag.py`, `python crewai/agents.py`, and `python react_agent.py` provide quick smoke tests for the primary flows.
- Use notebooks for exploratory runs and keep large downloads out of version control.

## Coding Style & Naming Conventions
- Follow PEP 8 with four-space indentation, explicit type hints, and docstrings where logic is non-obvious.
- Modules stay lowercase_with_underscores; class names use CamelCase; configuration constants remain UPPER_CASE near the top of each file.
- Reuse shared helpers instead of copying prompts or retrieval logic between labs.

## Testing Guidelines
- Add `pytest` modules alongside new production code and run them with `python -m pytest path/to/tests`.
- Course labs rely on inline graders (e.g., `coursera/Introduction to RAG/unittests.py`); call the provided `test_*` helpers from notebooks before committing.
- After editing data or retriever code, run the associated script end-to-end to confirm FAISS indexes, CrewAI tasks, and LangGraph workflows still execute.

## Commit & Pull Request Guidelines
- Keep commit subjects short and imperative (`Add LangGraph workflow for multi-agent prompt chaining`) and scope changes to a single concern.
- In commit bodies or PR descriptions, mention touched modules, dataset updates, required keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.), and smoke tests executed.
- Attach screenshots when notebook outputs or agent transcripts change materially.

## Environment & Secrets
- Store provider tokens in an untracked `.env` and load them with `python-dotenv` or shell exports; never hard-code keys.
- Document new environment variables or external services in your PR so teammates can reproduce results quickly.
