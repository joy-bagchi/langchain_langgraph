"""Database-backed runtime ledger with compatibility mirrors."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

from agentic_harness.contracts import (
    AgentInvocation,
    MemoryQuery,
    MemoryRecord,
    MemorySearchResult,
    dataclass_dict,
    utc_now,
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


class RuntimeLedger:
    """Durable runtime ledger backed by SQLite or Postgres."""

    def __init__(self, root: str | Path, *, database_url: str | None = None) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.database_url = database_url or os.getenv("AGENTIC_HARNESS_DB_URL") or self._default_sqlite_url(self.root)
        self.scheme = self.database_url.split(":", 1)[0]
        self._postgres_module = None
        if self.scheme in {"postgres", "postgresql"}:
            try:
                import psycopg
            except ImportError as exc:
                raise ImportError(
                    "psycopg is required for PostgreSQL runtime ledger support."
                ) from exc
            self._postgres_module = psycopg
        self._ensure_schema()

    @staticmethod
    def _default_sqlite_url(root: Path) -> str:
        db_path = (root / "runtime_ledger.db").resolve().as_posix()
        return f"sqlite:///{db_path}"

    def _sqlite_path(self) -> str:
        return self.database_url.removeprefix("sqlite:///")

    @contextmanager
    def _connect(self) -> Iterator[Any]:
        if self.scheme == "sqlite":
            db_path = Path(self._sqlite_path())
            db_path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(db_path, timeout=30, check_same_thread=False)
            connection.row_factory = sqlite3.Row
            try:
                yield connection
                connection.commit()
            finally:
                connection.close()
            return

        parsed = urlparse(self.database_url)
        if self._postgres_module is None:
            raise RuntimeError("PostgreSQL module is not available.")
        connection = self._postgres_module.connect(
            self.database_url,
            autocommit=False,
        )
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _execute_script(self, statements: list[str]) -> None:
        with self._connect() as connection:
            cursor = connection.cursor()
            for statement in statements:
                cursor.execute(statement)

    def _ensure_schema(self) -> None:
        statements = [
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                workflow_id TEXT,
                workflow_path TEXT,
                status TEXT,
                run_kind TEXT,
                state_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS manifests (
                run_id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS checkpoints (
                run_id TEXT NOT NULL,
                checkpoint_index INTEGER NOT NULL,
                state_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (run_id, checkpoint_index)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS events (
                run_id TEXT NOT NULL,
                event_index INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (run_id, event_index)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS artifacts (
                run_id TEXT NOT NULL,
                artifact_key TEXT NOT NULL,
                artifact_type TEXT,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (run_id, artifact_key)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS memory_records (
                record_id TEXT PRIMARY KEY,
                namespace TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                source_run_id TEXT NOT NULL,
                source_step_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT,
                metadata_json TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS agent_invocations (
                invocation_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                workflow_id TEXT NOT NULL,
                status TEXT NOT NULL,
                runtime_profile TEXT NOT NULL,
                parent_run_id TEXT,
                lease_owner TEXT,
                lease_expires_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            )
            """,
        ]
        self._execute_script(statements)

    def save_run_state(self, state: dict[str, Any]) -> None:
        run_id = str(state["run_id"])
        now = utc_now()
        status = str(state.get("status", "unknown"))
        workflow_id = str(state.get("workflow_id", ""))
        workflow_path = state.get("workflow_path")
        run_kind = "dag" if "execution_stages" in state else "workflow"
        state_json = json.dumps(state, sort_keys=True)
        checkpoint_index = int(state.get("checkpoint_index", 0))
        events = list(state.get("events", []))

        with self._connect() as connection:
            cursor = connection.cursor()
            if self.scheme == "sqlite":
                cursor.execute(
                    """
                    INSERT INTO runs (run_id, workflow_id, workflow_path, status, run_kind, state_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id) DO UPDATE SET
                        workflow_id=excluded.workflow_id,
                        workflow_path=excluded.workflow_path,
                        status=excluded.status,
                        run_kind=excluded.run_kind,
                        state_json=excluded.state_json,
                        updated_at=excluded.updated_at
                    """,
                    (run_id, workflow_id, workflow_path, status, run_kind, state_json, now, now),
                )
                cursor.execute(
                    """
                    INSERT INTO checkpoints (run_id, checkpoint_index, state_json, created_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(run_id, checkpoint_index) DO UPDATE SET
                        state_json=excluded.state_json,
                        created_at=excluded.created_at
                    """,
                    (run_id, checkpoint_index, state_json, now),
                )
                cursor.execute("DELETE FROM events WHERE run_id = ?", (run_id,))
                for index, event in enumerate(events):
                    event_type = str(event.get("type") or event.get("event_type") or "event")
                    cursor.execute(
                        """
                        INSERT INTO events (run_id, event_index, event_type, payload_json, created_at)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (run_id, index, event_type, json.dumps(event, sort_keys=True), now),
                    )
                cursor.execute("DELETE FROM artifacts WHERE run_id = ?", (run_id,))
                for artifact_key, payload in dict(state.get("artifacts", {})).items():
                    cursor.execute(
                        """
                        INSERT INTO artifacts (run_id, artifact_key, artifact_type, payload_json, created_at)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            run_id,
                            str(artifact_key),
                            str(payload.get("artifact_type", "")),
                            json.dumps(payload, sort_keys=True),
                            now,
                        ),
                    )
                for artifact_key, payload in dict(state.get("leaf_artifacts", {})).items():
                    cursor.execute(
                        """
                        INSERT INTO artifacts (run_id, artifact_key, artifact_type, payload_json, created_at)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(run_id, artifact_key) DO UPDATE SET
                            artifact_type=excluded.artifact_type,
                            payload_json=excluded.payload_json,
                            created_at=excluded.created_at
                        """,
                        (
                            run_id,
                            f"leaf:{artifact_key}",
                            str(payload.get("artifact_type", "")),
                            json.dumps(payload, sort_keys=True),
                            now,
                        ),
                    )
            else:
                cursor.execute(
                    """
                    INSERT INTO runs (run_id, workflow_id, workflow_path, status, run_kind, state_json, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT(run_id) DO UPDATE SET
                        workflow_id=EXCLUDED.workflow_id,
                        workflow_path=EXCLUDED.workflow_path,
                        status=EXCLUDED.status,
                        run_kind=EXCLUDED.run_kind,
                        state_json=EXCLUDED.state_json,
                        updated_at=EXCLUDED.updated_at
                    """,
                    (run_id, workflow_id, workflow_path, status, run_kind, state_json, now, now),
                )
                cursor.execute(
                    """
                    INSERT INTO checkpoints (run_id, checkpoint_index, state_json, created_at)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT(run_id, checkpoint_index) DO UPDATE SET
                        state_json=EXCLUDED.state_json,
                        created_at=EXCLUDED.created_at
                    """,
                    (run_id, checkpoint_index, state_json, now),
                )
                cursor.execute("DELETE FROM events WHERE run_id = %s", (run_id,))
                for index, event in enumerate(events):
                    event_type = str(event.get("type") or event.get("event_type") or "event")
                    cursor.execute(
                        """
                        INSERT INTO events (run_id, event_index, event_type, payload_json, created_at)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (run_id, index, event_type, json.dumps(event, sort_keys=True), now),
                    )
                cursor.execute("DELETE FROM artifacts WHERE run_id = %s", (run_id,))
                for source_key, payloads in (
                    ("", dict(state.get("artifacts", {}))),
                    ("leaf:", dict(state.get("leaf_artifacts", {}))),
                ):
                    for artifact_key, payload in payloads.items():
                        cursor.execute(
                            """
                            INSERT INTO artifacts (run_id, artifact_key, artifact_type, payload_json, created_at)
                            VALUES (%s, %s, %s, %s, %s)
                            ON CONFLICT(run_id, artifact_key) DO UPDATE SET
                                artifact_type=EXCLUDED.artifact_type,
                                payload_json=EXCLUDED.payload_json,
                                created_at=EXCLUDED.created_at
                            """,
                            (
                                run_id,
                                f"{source_key}{artifact_key}",
                                str(payload.get("artifact_type", "")),
                                json.dumps(payload, sort_keys=True),
                                now,
                            ),
                        )

        if state.get("agent_id"):
            self.save_agent_invocation(
                AgentInvocation(
                    invocation_id=str(state.get("invocation_id") or run_id),
                    run_id=run_id,
                    agent_id=str(state.get("agent_id")),
                    workflow_id=workflow_id,
                    status=status,
                    runtime_profile=str(state.get("runtime_profile") or "default"),
                    metadata={
                        "agent_name": state.get("agent_name"),
                        "agent_role": state.get("agent_role"),
                    },
                )
            )

    def load_run_state(self, run_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            cursor = connection.cursor()
            if self.scheme == "sqlite":
                row = cursor.execute(
                    "SELECT state_json FROM runs WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
            else:
                cursor.execute("SELECT state_json FROM runs WHERE run_id = %s", (run_id,))
                row = cursor.fetchone()

        if row is None:
            raise FileNotFoundError(f"No saved workflow run found for '{run_id}'.")
        raw = row["state_json"] if isinstance(row, sqlite3.Row) else row[0]
        return json.loads(raw)

    def save_manifest(self, run_id: str, payload: dict[str, Any]) -> None:
        now = utc_now()
        payload_json = json.dumps(dataclass_dict(payload), sort_keys=True)
        with self._connect() as connection:
            cursor = connection.cursor()
            if self.scheme == "sqlite":
                cursor.execute(
                    """
                    INSERT INTO manifests (run_id, payload_json, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(run_id) DO UPDATE SET
                        payload_json=excluded.payload_json,
                        updated_at=excluded.updated_at
                    """,
                    (run_id, payload_json, now),
                )
            else:
                cursor.execute(
                    """
                    INSERT INTO manifests (run_id, payload_json, updated_at)
                    VALUES (%s, %s, %s)
                    ON CONFLICT(run_id) DO UPDATE SET
                        payload_json=EXCLUDED.payload_json,
                        updated_at=EXCLUDED.updated_at
                    """,
                    (run_id, payload_json, now),
                )

    def save_agent_invocation(self, invocation: AgentInvocation) -> AgentInvocation:
        payload = invocation.to_dict()
        with self._connect() as connection:
            cursor = connection.cursor()
            if self.scheme == "sqlite":
                cursor.execute(
                    """
                    INSERT INTO agent_invocations (
                        invocation_id, run_id, agent_id, workflow_id, status, runtime_profile,
                        parent_run_id, lease_owner, lease_expires_at, created_at, updated_at, metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(invocation_id) DO UPDATE SET
                        status=excluded.status,
                        runtime_profile=excluded.runtime_profile,
                        parent_run_id=excluded.parent_run_id,
                        lease_owner=excluded.lease_owner,
                        lease_expires_at=excluded.lease_expires_at,
                        updated_at=excluded.updated_at,
                        metadata_json=excluded.metadata_json
                    """,
                    (
                        invocation.invocation_id,
                        invocation.run_id,
                        invocation.agent_id,
                        invocation.workflow_id,
                        invocation.status,
                        invocation.runtime_profile,
                        invocation.parent_run_id,
                        invocation.lease_owner,
                        invocation.lease_expires_at,
                        invocation.created_at,
                        invocation.updated_at,
                        json.dumps(payload.get("metadata", {}), sort_keys=True),
                    ),
                )
            else:
                cursor.execute(
                    """
                    INSERT INTO agent_invocations (
                        invocation_id, run_id, agent_id, workflow_id, status, runtime_profile,
                        parent_run_id, lease_owner, lease_expires_at, created_at, updated_at, metadata_json
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT(invocation_id) DO UPDATE SET
                        status=EXCLUDED.status,
                        runtime_profile=EXCLUDED.runtime_profile,
                        parent_run_id=EXCLUDED.parent_run_id,
                        lease_owner=EXCLUDED.lease_owner,
                        lease_expires_at=EXCLUDED.lease_expires_at,
                        updated_at=EXCLUDED.updated_at,
                        metadata_json=EXCLUDED.metadata_json
                    """,
                    (
                        invocation.invocation_id,
                        invocation.run_id,
                        invocation.agent_id,
                        invocation.workflow_id,
                        invocation.status,
                        invocation.runtime_profile,
                        invocation.parent_run_id,
                        invocation.lease_owner,
                        invocation.lease_expires_at,
                        invocation.created_at,
                        invocation.updated_at,
                        json.dumps(payload.get("metadata", {}), sort_keys=True),
                    ),
                )
        return invocation

    def remember_memory(self, record: MemoryRecord) -> MemoryRecord:
        existing = self._find_duplicate_memory(record)
        now = utc_now()
        if existing is not None:
            merged_metadata = {**existing.metadata, **record.metadata}
            merged = MemoryRecord(
                record_id=existing.record_id,
                namespace=existing.namespace,
                memory_type=existing.memory_type,
                content=existing.content,
                source_run_id=existing.source_run_id,
                source_step_id=existing.source_step_id,
                created_at=existing.created_at,
                expires_at=existing.expires_at,
                metadata=merged_metadata,
            )
            with self._connect() as connection:
                cursor = connection.cursor()
                if self.scheme == "sqlite":
                    cursor.execute(
                        "UPDATE memory_records SET metadata_json = ? WHERE record_id = ?",
                        (json.dumps(merged_metadata, sort_keys=True), merged.record_id),
                    )
                else:
                    cursor.execute(
                        "UPDATE memory_records SET metadata_json = %s WHERE record_id = %s",
                        (json.dumps(merged_metadata, sort_keys=True), merged.record_id),
                    )
            return merged

        with self._connect() as connection:
            cursor = connection.cursor()
            if self.scheme == "sqlite":
                cursor.execute(
                    """
                    INSERT INTO memory_records (
                        record_id, namespace, memory_type, content, source_run_id, source_step_id,
                        created_at, expires_at, metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.record_id,
                        record.namespace,
                        record.memory_type,
                        record.content,
                        record.source_run_id,
                        record.source_step_id,
                        record.created_at or now,
                        record.expires_at,
                        json.dumps(record.metadata, sort_keys=True),
                    ),
                )
            else:
                cursor.execute(
                    """
                    INSERT INTO memory_records (
                        record_id, namespace, memory_type, content, source_run_id, source_step_id,
                        created_at, expires_at, metadata_json
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        record.record_id,
                        record.namespace,
                        record.memory_type,
                        record.content,
                        record.source_run_id,
                        record.source_step_id,
                        record.created_at or now,
                        record.expires_at,
                        json.dumps(record.metadata, sort_keys=True),
                    ),
                )
        return record

    def _find_duplicate_memory(self, record: MemoryRecord) -> MemoryRecord | None:
        with self._connect() as connection:
            cursor = connection.cursor()
            if self.scheme == "sqlite":
                row = cursor.execute(
                    """
                    SELECT * FROM memory_records
                    WHERE namespace = ? AND memory_type = ? AND content = ?
                    LIMIT 1
                    """,
                    (record.namespace, record.memory_type, record.content),
                ).fetchone()
            else:
                cursor.execute(
                    """
                    SELECT * FROM memory_records
                    WHERE namespace = %s AND memory_type = %s AND content = %s
                    LIMIT 1
                    """,
                    (record.namespace, record.memory_type, record.content),
                )
                row = cursor.fetchone()
        if row is None:
            return None
        return self._memory_row_to_record(row)

    def recall_memory(self, query: MemoryQuery) -> list[MemorySearchResult]:
        with self._connect() as connection:
            cursor = connection.cursor()
            if self.scheme == "sqlite":
                if query.memory_types:
                    placeholders = ", ".join("?" for _ in query.memory_types)
                    rows = cursor.execute(
                        f"""
                        SELECT * FROM memory_records
                        WHERE namespace = ? AND memory_type IN ({placeholders})
                        """,
                        (query.namespace, *query.memory_types),
                    ).fetchall()
                else:
                    rows = cursor.execute(
                        "SELECT * FROM memory_records WHERE namespace = ?",
                        (query.namespace,),
                    ).fetchall()
            else:
                if query.memory_types:
                    placeholders = ", ".join("%s" for _ in query.memory_types)
                    cursor.execute(
                        f"""
                        SELECT * FROM memory_records
                        WHERE namespace = %s AND memory_type IN ({placeholders})
                        """,
                        (query.namespace, *query.memory_types),
                    )
                else:
                    cursor.execute(
                        "SELECT * FROM memory_records WHERE namespace = %s",
                        (query.namespace,),
                    )
                rows = cursor.fetchall()

        query_terms = {
            term.lower()
            for term in query.text.replace("\n", " ").split()
            if term.strip()
        }
        now = _utcnow()
        results: list[MemorySearchResult] = []
        for row in rows:
            record = self._memory_row_to_record(row)
            if record.expires_at:
                expires_at = datetime.fromisoformat(record.expires_at)
                if expires_at <= now:
                    continue
            haystack = f"{record.content} {json.dumps(record.metadata, sort_keys=True)}".lower()
            score = 1.0 if not query_terms else 0.0
            if query_terms:
                matches = sum(1 for term in query_terms if term in haystack)
                if matches == 0:
                    continue
                score = matches / len(query_terms)
            results.append(MemorySearchResult(record=record, score=score))

        results.sort(key=lambda item: item.score, reverse=True)
        return results[: query.max_results]

    def _memory_row_to_record(self, row: Any) -> MemoryRecord:
        def _value(key: str, index: int) -> Any:
            if isinstance(row, sqlite3.Row):
                return row[key]
            return row[index]

        return MemoryRecord(
            record_id=_value("record_id", 0),
            namespace=_value("namespace", 1),
            memory_type=_value("memory_type", 2),
            content=_value("content", 3),
            source_run_id=_value("source_run_id", 4),
            source_step_id=_value("source_step_id", 5),
            created_at=_value("created_at", 6),
            expires_at=_value("expires_at", 7),
            metadata=json.loads(_value("metadata_json", 8) or "{}"),
        )


class FilesystemMemoryStore:
    """Compatibility wrapper around the database runtime ledger."""

    def __init__(self, root: str | Path, *, database_url: str | None = None) -> None:
        self.root = Path(root)
        self.index_path = self.root / "memory" / "memory_index.json"
        self.events_path = self.root / "memory" / "memory_events.jsonl"
        self.ledger = RuntimeLedger(root, database_url=database_url)
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
        stored = self.ledger.remember_memory(record)
        records = self._load_records()
        for index, existing in enumerate(records):
            if existing.record_id == stored.record_id:
                records[index] = stored
                break
        else:
            records.append(stored)
        self._save_records(records)
        self._append_event(
            {
                "event": "memory_written",
                "record_id": stored.record_id,
                "namespace": stored.namespace,
                "timestamp": utc_now(),
            }
        )
        return stored

    def recall(self, query: MemoryQuery) -> list[MemorySearchResult]:
        return self.ledger.recall_memory(query)


class WorkflowRunStore:
    """Database-backed workflow run store with JSON compatibility mirrors."""

    def __init__(self, root: str | Path, *, database_url: str | None = None) -> None:
        self.root = Path(root)
        self.runs_root = self.root / "runs"
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.ledger = RuntimeLedger(root, database_url=database_url)

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

        self.ledger.save_run_state(state)
        _write_json(state_path, state)
        _write_json(checkpoint_path, state)
        _ensure_parent(events_path)
        with events_path.open("w", encoding="utf-8") as handle:
            for event in state.get("events", []):
                handle.write(json.dumps(event, sort_keys=True))
                handle.write("\n")

    def load_state(self, run_id: str) -> dict[str, Any]:
        try:
            return self.ledger.load_run_state(run_id)
        except FileNotFoundError:
            state_path = self.run_dir(run_id) / "state.json"
            if not state_path.exists():
                raise
            return _read_json(state_path, {})

    def inspect(self, run_id: str) -> dict[str, Any]:
        return self.load_state(run_id)

    def save_manifest(self, run_id: str, payload: dict[str, Any]) -> None:
        self.ledger.save_manifest(run_id, payload)
        manifest_path = self.run_dir(run_id) / "run_manifest.json"
        _write_json(manifest_path, dataclass_dict(payload))

    def save_agent_invocation(self, invocation: AgentInvocation) -> AgentInvocation:
        return self.ledger.save_agent_invocation(invocation)
