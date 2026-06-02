import json
import os
import shutil
import sys
import uuid
from pathlib import Path

import psycopg
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic_harness.runtime import run_agent_workflow


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _database_url() -> str:
    return os.environ.get(
        "AGENTIC_HARNESS_DB_URL",
        "postgresql://postgres:postgres@localhost:5432/agentic_harness",
    )


def _prepare_temp_agent(workdir: Path, namespace: str) -> Path:
    root = _project_root()
    source_agent = root / "agents" / "research_analyst.yaml"
    source_workflow = root / "examples" / "workflows" / "research_brief.md"

    agent_data = yaml.safe_load(source_agent.read_text(encoding="utf-8"))
    workflow_text = source_workflow.read_text(encoding="utf-8")
    workflow_text = workflow_text.replace("memory_namespace: research_brief_memory", f"memory_namespace: {namespace}")

    temp_workflow = workdir / "research_brief_semantic.md"
    temp_workflow.write_text(workflow_text, encoding="utf-8")

    agent_data["agent_id"] = "research_analyst_semantic_smoke"
    agent_data["name"] = "Research Analyst Semantic Smoke"
    agent_data["workflow_path"] = "./research_brief_semantic.md"

    temp_agent = workdir / "research_analyst_semantic.yaml"
    temp_agent.write_text(yaml.safe_dump(agent_data, sort_keys=False), encoding="utf-8")
    return temp_agent


def _count_memory_rows(database_url: str, namespace: str) -> int:
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*)
                FROM memory_embeddings e
                JOIN memory_records r ON r.record_id = e.record_id
                WHERE r.namespace = %s
                """,
                (namespace,),
            )
            return int(cur.fetchone()[0])


def main() -> None:
    root = _project_root()
    storage_root = root / ".workflow_memory_semantic_smoke"
    temp_root = storage_root / "semantic_smoke_assets"
    namespace = f"semantic_smoke_{uuid.uuid4().hex[:12]}"
    database_url = _database_url()

    if storage_root.exists():
        shutil.rmtree(storage_root)
    temp_root.mkdir(parents=True, exist_ok=True)

    temp_agent = _prepare_temp_agent(temp_root, namespace)

    first_topic = "What is an SABR model?"
    second_topic = "Explain SABR model parameters and why the model is used in interest rate derivatives."

    first_result = run_agent_workflow(
        temp_agent,
        {"topic": first_topic},
        storage_root=storage_root,
        database_url=database_url,
    )
    second_result = run_agent_workflow(
        temp_agent,
        {"topic": second_topic},
        storage_root=storage_root,
        database_url=database_url,
    )

    first_summary = (first_result.get("active_context") or {}).get("memory_summary", "")
    second_summary = (second_result.get("active_context") or {}).get("memory_summary", "")
    embedding_row_count = _count_memory_rows(database_url, namespace)

    output = {
        "database_url": database_url,
        "memory_namespace": namespace,
        "agent_path": str(temp_agent),
        "first_run": {
            "run_id": first_result.get("run_id"),
            "status": first_result.get("status"),
            "pending_review_step": (first_result.get("pending_review") or {}).get("step_id"),
            "memory_summary": first_summary,
        },
        "second_run": {
            "run_id": second_result.get("run_id"),
            "status": second_result.get("status"),
            "pending_review_step": (second_result.get("pending_review") or {}).get("step_id"),
            "memory_summary": second_summary,
            "recalled_first_topic": first_topic in second_summary,
        },
        "pgvector_validation": {
            "embedding_row_count": embedding_row_count,
        },
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
