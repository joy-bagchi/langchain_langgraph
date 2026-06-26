"""Memory service contract and default implementations."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from agentic_harness.contracts import MemoryQuery, MemoryRecord, MemorySearchResult
from agentic_harness.shared.services import ServiceDescriptor
from agentic_harness.stores import FilesystemMemoryStore, RuntimeLedger, _dict_matches_filters


@dataclass(slots=True)
class MemoryServiceSelection:
    """Configuration for choosing a memory service implementation."""

    service_type: str = "filesystem"
    storage_root: str | Path | None = None
    database_url: str | None = None


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    left_norm = math.sqrt(sum(item * item for item in left))
    right_norm = math.sqrt(sum(item * item for item in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    return dot / (left_norm * right_norm)


def _format_pgvector(value: list[float]) -> str:
    return "[" + ",".join(f"{item:.12f}" for item in value) + "]"


def _default_embedding_callable(text: str, *, dimensions: int = 64) -> list[float]:
    """
    Produce a deterministic local embedding.

    This is intentionally model-free so the semantic memory layer can work
    without external embedding infrastructure. It is not meant to replace a
    real embedding model, but it provides stable semantic-ish recall behavior
    and a vector payload for the pgvector-backed path.
    """

    buckets = [0.0] * dimensions
    tokens = [token.strip().lower() for token in text.replace("\n", " ").split() if token.strip()]
    if not tokens:
        return buckets
    for token in tokens:
        token_value = sum(ord(char) for char in token)
        index = token_value % dimensions
        sign = -1.0 if token_value % 2 else 1.0
        buckets[index] += sign * (1.0 + (len(token) / 10.0))
    norm = math.sqrt(sum(item * item for item in buckets))
    if norm == 0.0:
        return buckets
    return [item / norm for item in buckets]


class MemoryService(Protocol):
    descriptor: ServiceDescriptor

    def remember(self, record: MemoryRecord) -> MemoryRecord:
        """Persist a durable memory record."""

    def recall(self, query: MemoryQuery) -> list[MemorySearchResult]:
        """Recall durable memory records."""


class FilesystemMemoryService:
    """Durable memory implementation backed by the runtime ledger."""

    def __init__(self, root: str | Path, *, database_url: str | None = None) -> None:
        self.store = FilesystemMemoryStore(root, database_url=database_url)
        self.descriptor = ServiceDescriptor(
            service_name="memory",
            implementation_id="database_memory_service",
            maturity="simple",
            capabilities=["durable_memory", "namespace_recall", "runtime_ledger_backed"],
        )

    def remember(self, record: MemoryRecord) -> MemoryRecord:
        return self.store.remember(record)

    def recall(self, query: MemoryQuery) -> list[MemorySearchResult]:
        return self.store.recall(query)


class EphemeralMemoryService:
    """In-memory service suited for short-running or test workflows."""

    def __init__(self) -> None:
        self.records: list[MemoryRecord] = []
        self.descriptor = ServiceDescriptor(
            service_name="memory",
            implementation_id="ephemeral_memory_service",
            maturity="simple",
            capabilities=["ephemeral_memory"],
        )

    def remember(self, record: MemoryRecord) -> MemoryRecord:
        self.records.append(record)
        return record

    def recall(self, query: MemoryQuery) -> list[MemorySearchResult]:
        matches: list[MemorySearchResult] = []
        for record in self.records:
            if record.namespace != query.namespace:
                continue
            haystack = f"{record.content} {record.metadata}".lower()
            score = 1.0 if not query.text else float(query.text.lower() in haystack)
            if score > 0:
                matches.append(MemorySearchResult(record=record, score=score))
        matches.sort(key=lambda item: item.score, reverse=True)
        return matches[: query.max_results]


class SemanticMemoryService:
    """Semantic memory service with Postgres-first vector support."""

    def __init__(
        self,
        root: str | Path,
        *,
        database_url: str | None = None,
        embedding_callable: Callable[[str], list[float]] | None = None,
        embedding_dimensions: int = 64,
    ) -> None:
        self.root = Path(root)
        self.store = FilesystemMemoryStore(root, database_url=database_url)
        self.ledger = RuntimeLedger(root, database_url=database_url)
        self.embedding_dimensions = embedding_dimensions
        self.embedding_callable = embedding_callable or (
            lambda text: _default_embedding_callable(text, dimensions=self.embedding_dimensions)
        )
        self._pgvector_ready = False
        self._ensure_embedding_schema()
        capabilities = ["durable_memory", "semantic_memory", "embedding_recall"]
        if self._pgvector_ready:
            capabilities.append("pgvector_backed")
        self.descriptor = ServiceDescriptor(
            service_name="memory",
            implementation_id="semantic_memory_service",
            maturity="advanced",
            capabilities=capabilities,
            metadata={
                "embedding_dimensions": self.embedding_dimensions,
                "database_scheme": self.ledger.scheme,
                "pgvector_ready": self._pgvector_ready,
            },
        )

    def _ensure_embedding_schema(self) -> None:
        statements = [
            """
            CREATE TABLE IF NOT EXISTS memory_embeddings (
                record_id TEXT PRIMARY KEY,
                embedding_json TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        ]
        self.ledger._execute_script(statements)
        if self.ledger.scheme not in {"postgres", "postgresql"}:
            return
        try:
            self.ledger._execute_script(
                [
                    "CREATE EXTENSION IF NOT EXISTS vector",
                    (
                        "ALTER TABLE memory_embeddings "
                        f"ADD COLUMN IF NOT EXISTS embedding vector({self.embedding_dimensions})"
                    ),
                    "CREATE INDEX IF NOT EXISTS idx_memory_records_namespace ON memory_records(namespace)",
                    (
                        "CREATE INDEX IF NOT EXISTS idx_memory_embeddings_embedding "
                        "ON memory_embeddings USING ivfflat (embedding vector_cosine_ops)"
                    ),
                ]
            )
            self._pgvector_ready = True
        except Exception:
            self._pgvector_ready = False

    def remember(self, record: MemoryRecord) -> MemoryRecord:
        stored = self.store.remember(record)
        embedding = self.embedding_callable(stored.content)
        created_at = stored.created_at
        payload_json = json.dumps(embedding)
        with self.ledger._connect() as connection:
            cursor = connection.cursor()
            if self.ledger.scheme == "sqlite":
                cursor.execute(
                    """
                    INSERT INTO memory_embeddings (record_id, embedding_json, dimensions, created_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(record_id) DO UPDATE SET
                        embedding_json=excluded.embedding_json,
                        dimensions=excluded.dimensions,
                        created_at=excluded.created_at
                    """,
                    (stored.record_id, payload_json, len(embedding), created_at),
                )
            elif self._pgvector_ready:
                cursor.execute(
                    """
                    INSERT INTO memory_embeddings (record_id, embedding_json, dimensions, created_at, embedding)
                    VALUES (%s, %s, %s, %s, %s::vector)
                    ON CONFLICT(record_id) DO UPDATE SET
                        embedding_json=EXCLUDED.embedding_json,
                        dimensions=EXCLUDED.dimensions,
                        created_at=EXCLUDED.created_at,
                        embedding=EXCLUDED.embedding
                    """,
                    (stored.record_id, payload_json, len(embedding), created_at, _format_pgvector(embedding)),
                )
            else:
                cursor.execute(
                    """
                    INSERT INTO memory_embeddings (record_id, embedding_json, dimensions, created_at)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT(record_id) DO UPDATE SET
                        embedding_json=EXCLUDED.embedding_json,
                        dimensions=EXCLUDED.dimensions,
                        created_at=EXCLUDED.created_at
                    """,
                    (stored.record_id, payload_json, len(embedding), created_at),
                )
        return stored

    def recall(self, query: MemoryQuery) -> list[MemorySearchResult]:
        query_embedding = self.embedding_callable(query.text)
        if self._pgvector_ready:
            results = self._recall_pgvector(query, query_embedding)
            if results:
                return results
        return self._recall_python(query, query_embedding)

    def _recall_pgvector(self, query: MemoryQuery, query_embedding: list[float]) -> list[MemorySearchResult]:
        where_clause = ["mr.namespace = %s"]
        filter_params: list[object] = [query.namespace]
        if query.memory_types:
            placeholders = ", ".join("%s" for _ in query.memory_types)
            where_clause.append(f"mr.memory_type IN ({placeholders})")
            filter_params.extend(query.memory_types)
        where_sql = " AND ".join(where_clause)
        query_vector = _format_pgvector(query_embedding)
        params: list[object] = [query_vector, *filter_params, query_vector, max(query.max_results, 1)]
        sql = f"""
            SELECT
                mr.record_id,
                mr.namespace,
                mr.memory_type,
                mr.content,
                mr.source_run_id,
                mr.source_step_id,
                mr.created_at,
                mr.expires_at,
                mr.metadata_json,
                mr.structured_payload_json,
                1 - (me.embedding <=> %s::vector) AS score
            FROM memory_records mr
            JOIN memory_embeddings me ON me.record_id = mr.record_id
            WHERE {where_sql}
            ORDER BY me.embedding <=> %s::vector
            LIMIT %s
        """
        with self.ledger._connect() as connection:
            cursor = connection.cursor()
            cursor.execute(sql, tuple(params))
            rows = cursor.fetchall()
        results: list[MemorySearchResult] = []
        for row in rows:
            record = MemoryRecord(
                record_id=row[0],
                namespace=row[1],
                memory_type=row[2],
                content=row[3],
                source_run_id=row[4],
                source_step_id=row[5],
                created_at=row[6],
                expires_at=row[7],
                metadata=json.loads(row[8] or "{}"),
                structured_payload=json.loads(row[9] or "{}"),
            )
            if not _dict_matches_filters(record.metadata, query.metadata_filters):
                continue
            if not _dict_matches_filters(record.structured_payload, query.structured_filters):
                continue
            score = float(row[10] or 0.0)
            if score > 0:
                results.append(MemorySearchResult(record=record, score=score))
        return results

    def _recall_python(self, query: MemoryQuery, query_embedding: list[float]) -> list[MemorySearchResult]:
        where_clause = ["mr.namespace = ?"] if self.ledger.scheme == "sqlite" else ["mr.namespace = %s"]
        params: list[object] = [query.namespace]
        if query.memory_types:
            placeholders = ", ".join(("?" if self.ledger.scheme == "sqlite" else "%s") for _ in query.memory_types)
            where_clause.append(f"mr.memory_type IN ({placeholders})")
            params.extend(query.memory_types)
        sql = f"""
            SELECT
                mr.record_id,
                mr.namespace,
                mr.memory_type,
                mr.content,
                mr.source_run_id,
                mr.source_step_id,
                mr.created_at,
                mr.expires_at,
                mr.metadata_json,
                mr.structured_payload_json,
                me.embedding_json
            FROM memory_records mr
            JOIN memory_embeddings me ON me.record_id = mr.record_id
            WHERE {" AND ".join(where_clause)}
        """
        with self.ledger._connect() as connection:
            cursor = connection.cursor()
            if self.ledger.scheme == "sqlite":
                rows = cursor.execute(sql, tuple(params)).fetchall()
            else:
                cursor.execute(sql, tuple(params))
                rows = cursor.fetchall()
        results: list[MemorySearchResult] = []
        for row in rows:
            record = MemoryRecord(
                record_id=row[0],
                namespace=row[1],
                memory_type=row[2],
                content=row[3],
                source_run_id=row[4],
                source_step_id=row[5],
                created_at=row[6],
                expires_at=row[7],
                metadata=json.loads(row[8] or "{}"),
                structured_payload=json.loads(row[9] or "{}"),
            )
            embedding = json.loads(row[10] or "[]")
            score = _cosine_similarity(query_embedding, embedding)
            if query.text and score <= 0:
                continue
            results.append(MemorySearchResult(record=record, score=score))
        results.sort(key=lambda item: item.score, reverse=True)
        return results[: query.max_results]


class StructuredMemoryService:
    """Structured memory service with field-filtered recall over durable records."""

    def __init__(self, root: str | Path, *, database_url: str | None = None) -> None:
        self.store = FilesystemMemoryStore(root, database_url=database_url)
        self.descriptor = ServiceDescriptor(
            service_name="memory",
            implementation_id="structured_memory_service",
            maturity="advanced",
            capabilities=[
                "durable_memory",
                "structured_memory",
                "field_filtered_recall",
                "runtime_ledger_backed",
            ],
        )

    def remember(self, record: MemoryRecord) -> MemoryRecord:
        return self.store.remember(record)

    def recall(self, query: MemoryQuery) -> list[MemorySearchResult]:
        return self.store.recall(query)

