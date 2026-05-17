"""Memory service contract and default implementations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from agentic_harness.contracts import MemoryQuery, MemoryRecord, MemorySearchResult
from agentic_harness.shared.services import ServiceDescriptor
from agentic_harness.stores import FilesystemMemoryStore


@dataclass(slots=True)
class MemoryServiceSelection:
    """Configuration for choosing a memory service implementation."""

    service_type: str = "filesystem"
    storage_root: str | Path | None = None
    database_url: str | None = None


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

