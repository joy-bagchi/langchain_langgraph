import json
import pathlib
import sys
import uuid

import psycopg

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from agentic_harness.contracts import MemoryQuery, MemoryRecord
from agentic_harness.agentic_os.memory_service import SemanticMemoryService


def main() -> None:
    database_url = "postgresql://postgres:postgres@localhost:5432/agentic_harness"
    conn = psycopg.connect(database_url)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
    ext = cur.fetchone()

    service = SemanticMemoryService(
        pathlib.Path(".workflow_memory_semantic_pgvector_verify"),
        database_url=database_url,
    )
    service.remember(
        MemoryRecord.create(
            namespace="semantic_pgvector_test",
            memory_type="semantic",
            content="SABR is a stochastic volatility model used in interest rate derivatives.",
            source_run_id=f"semantic-pgvector-run-{uuid.uuid4()}",
            source_step_id="verify_pgvector",
            metadata={"topic": "finance"},
        )
    )
    service.remember(
        MemoryRecord.create(
            namespace="semantic_pgvector_test",
            memory_type="semantic",
            content="LangGraph is used for stateful graph orchestration.",
            source_run_id=f"semantic-pgvector-run-{uuid.uuid4()}",
            source_step_id="verify_pgvector",
            metadata={"topic": "agents"},
        )
    )
    results = service.recall(
        MemoryQuery(
            namespace="semantic_pgvector_test",
            text="What stochastic volatility model is used in rates?",
            max_results=2,
            memory_types=["semantic"],
        )
    )
    cur.execute(
        """
        SELECT COUNT(*)
        FROM memory_embeddings e
        JOIN memory_records r ON r.record_id = e.record_id
        WHERE r.namespace = %s
        """,
        ("semantic_pgvector_test",),
    )
    count = cur.fetchone()[0]
    print(
        json.dumps(
            {
                "vector_extension_version": ext[0] if ext else None,
                "pgvector_ready": bool(getattr(service, "_pgvector_ready", False)),
                "result_count": len(results),
                "top_result": results[0].record.content if results else None,
                "top_score": results[0].score if results else None,
                "embedding_row_count": count,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
