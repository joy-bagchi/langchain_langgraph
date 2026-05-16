"""Filesystem-backed run and memory stores."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentic_harness.contracts import (
    MemoryQuery,
    MemoryRecord,
    MemorySearchResult,
    dataclass_dict,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    _ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


class FilesystemMemoryStore:
    """Durable memory store with a latest-state index and event log."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.index_path = self.root / "memory" / "memory_index.json"
        self.events_path = self.root / "memory" / "memory_events.jsonl"
        self.root.mkdir(parents=True, exist_ok=True)

    def _load_records(self) -> list[MemoryRecord]:
        payload = _read_json(self.index_path, [])
        return [MemoryRecord.from_dict(item) for item in payload]

    def _save_records(self, records: list[MemoryRecord]) -> None:
        _write_json(self.index_path, [record.to_dict() for record in records])

    def _append_event(self, event: dict[str, Any]) -> None:
        _ensure_parent(self.events_path)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True))
            handle.write("\n")

    def remember(self, record: MemoryRecord) -> MemoryRecord:
        """Insert or merge a record into the durable store."""
        records = self._load_records()
        for existing in records:
            if (
                existing.namespace == record.namespace
                and existing.memory_type == record.memory_type
                and existing.content == record.content
            ):
                existing.metadata = {**existing.metadata, **record.metadata}
                self._save_records(records)
                self._append_event(
                    {
                        "event": "memory_merged",
                        "record_id": existing.record_id,
                        "namespace": existing.namespace,
                        "timestamp": existing.created_at,
                    }
                )
                return existing

        records.append(record)
        self._save_records(records)
        self._append_event(
            {
                "event": "memory_created",
                "record_id": record.record_id,
                "namespace": record.namespace,
                "timestamp": record.created_at,
            }
        )
        return record

    def recall(self, query: MemoryQuery) -> list[MemorySearchResult]:
        """Return simple lexical matches for a namespace-scoped query."""
        query_terms = {
            term.lower()
            for term in query.text.replace("\n", " ").split()
            if term.strip()
        }
        now = _utcnow()
        results: list[MemorySearchResult] = []

        for record in self._load_records():
            if record.namespace != query.namespace:
                continue
            if query.memory_types and record.memory_type not in query.memory_types:
                continue
            if record.expires_at:
                expires_at = datetime.fromisoformat(record.expires_at)
                if expires_at <= now:
                    continue

            haystack = f"{record.content} {json.dumps(record.metadata, sort_keys=True)}".lower()
            score = 0.0
            if not query_terms:
                score = 1.0
            else:
                matches = sum(1 for term in query_terms if term in haystack)
                if matches == 0:
                    continue
                score = matches / len(query_terms)

            results.append(MemorySearchResult(record=record, score=score))

        results.sort(key=lambda item: item.score, reverse=True)
        return results[: query.max_results]


class WorkflowRunStore:
    """Persists workflow runs, checkpoints, and event journals to disk."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.runs_root = self.root / "runs"
        self.runs_root.mkdir(parents=True, exist_ok=True)

    def run_dir(self, run_id: str) -> Path:
        return self.runs_root / run_id

    def save_state(self, state: dict[str, Any]) -> None:
        run_id = state["run_id"]
        run_dir = self.run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        state_path = run_dir / "state.json"
        checkpoints_dir = run_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoints_dir / f"checkpoint_{state.get('checkpoint_index', 0):04d}.json"
        events_path = run_dir / "events.jsonl"

        _write_json(state_path, state)
        _write_json(checkpoint_path, state)
        _ensure_parent(events_path)
        with events_path.open("w", encoding="utf-8") as handle:
            for event in state.get("events", []):
                handle.write(json.dumps(event, sort_keys=True))
                handle.write("\n")

    def load_state(self, run_id: str) -> dict[str, Any]:
        state_path = self.run_dir(run_id) / "state.json"
        if not state_path.exists():
            raise FileNotFoundError(f"No saved workflow run found for '{run_id}'.")
        return _read_json(state_path, {})

    def inspect(self, run_id: str) -> dict[str, Any]:
        return self.load_state(run_id)

    def save_manifest(self, run_id: str, payload: dict[str, Any]) -> None:
        manifest_path = self.run_dir(run_id) / "run_manifest.json"
        _write_json(manifest_path, dataclass_dict(payload))

