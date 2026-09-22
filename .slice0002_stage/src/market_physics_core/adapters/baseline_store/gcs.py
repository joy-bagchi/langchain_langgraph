"""Transactional GCS baseline history store."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any

from google.api_core.exceptions import PreconditionFailed
from google.cloud import storage

from .artifacts import bundle_relative_path, serialize_bundle, sha256_bytes


class PointerConflict(RuntimeError):
    """A concurrent writer changed a mutable pointer."""


class GCSBaselineStore:
    def __init__(self, project: str, bucket: str, prefix: str, *, client: Any = None) -> None:
        self.bucket_name = bucket
        self.prefix = prefix.strip("/")
        self.client = client or storage.Client(project=project)
        self.bucket = self.client.bucket(bucket)

    def _name(self, relative: str) -> str:
        return f"{self.prefix}/{relative.lstrip('/')}"

    @staticmethod
    def _json(payload: bytes) -> dict[str, Any]:
        value = json.loads(payload.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("history JSON must be an object")
        return value

    def _read_optional(self, relative: str) -> tuple[dict[str, Any] | None, int]:
        blob = self.bucket.blob(self._name(relative))
        try:
            payload = blob.download_as_bytes()
            blob.reload()
        except Exception as exc:
            if exc.__class__.__name__ == "NotFound":
                return None, 0
            raise
        return self._json(payload), int(blob.generation)

    def _put_json(self, relative: str, value: dict[str, Any], generation: int) -> int:
        blob = self.bucket.blob(self._name(relative))
        payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
        try:
            blob.upload_from_string(payload, content_type="application/json", if_generation_match=generation)
        except PreconditionFailed as exc:
            raise PointerConflict(relative) from exc
        blob.reload()
        return int(blob.generation)

    def publish(
        self, summary: dict[str, Any], observations: list[dict[str, Any]], *,
        publication_kind: str = "contemporaneous", advance_latest: bool = True,
        failure_after_artifacts: bool = False,
    ) -> dict[str, Any]:
        summary = dict(summary)
        summary["publication_kind"] = publication_kind
        summary, artifacts = serialize_bundle(summary, observations)
        run_path = bundle_relative_path(summary)
        committed, _ = self._read_optional(f"{run_path}/commit.json")
        if committed:
            self._verify_commit(committed)
            return {"outcome": "idempotent", "run_path": run_path, "commit": committed}

        objects: dict[str, Any] = {}
        for filename, payload in artifacts.items():
            object_name = self._name(f"{run_path}/{filename}")
            blob = self.bucket.blob(object_name)
            try:
                blob.upload_from_string(payload, if_generation_match=0)
            except PreconditionFailed:
                existing = blob.download_as_bytes()
                if sha256_bytes(existing) != sha256_bytes(payload):
                    raise ValueError(f"immutable object collision: gs://{self.bucket_name}/{object_name}")
            readback = blob.download_as_bytes()
            blob.reload()
            if sha256_bytes(readback) != sha256_bytes(payload):
                raise ValueError(f"read-back checksum mismatch: {object_name}")
            objects[filename] = {
                "uri": f"gs://{self.bucket_name}/{object_name}",
                "generation": str(blob.generation), "size": len(payload),
                "sha256": sha256_bytes(payload),
            }
        if failure_after_artifacts:
            raise RuntimeError("injected failure before commit marker")

        commit = {
            "schema_version": "baseline_commit.v1", "completion_status": "committed",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "run_id": summary["run_id"], "idempotency_key": summary["idempotency_key"],
            "as_of_session": summary["as_of_session"], "week_id": summary["week_id"],
            "configuration_hash": summary["configuration_hash"], "code_commit": summary["code_commit"],
            "source_objects": summary["source_objects"], "objects": objects,
            "publication_kind": publication_kind,
        }
        commit_generation = self._put_json(f"{run_path}/commit.json", commit, 0)
        commit["commit_uri"] = f"gs://{self.bucket_name}/{self._name(f'{run_path}/commit.json')}"
        commit["commit_generation"] = str(commit_generation)
        self._update_index(commit, run_path)
        advanced = self._update_latest(commit, run_path) if advance_latest and summary["status"] != "insufficient_data" else False
        return {"outcome": "published", "run_path": run_path, "latest_advanced": advanced, "commit": commit}

    def _verify_commit(self, commit: dict[str, Any]) -> None:
        if commit.get("completion_status") != "committed":
            raise ValueError("uncommitted baseline bundle")
        for descriptor in commit["objects"].values():
            uri = descriptor["uri"]
            object_name = uri.split(f"gs://{self.bucket_name}/", 1)[1]
            payload = self.bucket.blob(object_name).download_as_bytes(if_generation_match=int(descriptor["generation"]))
            if len(payload) != descriptor["size"] or sha256_bytes(payload) != descriptor["sha256"]:
                raise ValueError(f"committed object integrity failure: {uri}")

    def _update_index(self, commit: dict[str, Any], run_path: str) -> None:
        for _ in range(5):
            index, generation = self._read_optional("index.json")
            index = index or {"schema_version": "baseline_index.v1", "records": []}
            if any(item["idempotency_key"] == commit["idempotency_key"] for item in index["records"]):
                return
            same_date = [item for item in index["records"] if item["as_of_session"] == commit["as_of_session"]]
            revision = 1 + max((int(item["revision"]) for item in same_date), default=0)
            record = {"as_of_session": commit["as_of_session"], "week_id": commit["week_id"],
                      "idempotency_key": commit["idempotency_key"], "run_id": commit["run_id"],
                      "run_path": run_path, "revision": revision,
                      "supersedes": same_date[-1]["run_id"] if same_date else None,
                      "publication_kind": commit["publication_kind"]}
            index["records"].append(record)
            index["records"].sort(key=lambda item: (item["as_of_session"], int(item["revision"])))
            try:
                self._put_json("index.json", index, generation)
                return
            except PointerConflict:
                continue
        raise PointerConflict("index.json retry limit")

    def _update_latest(self, commit: dict[str, Any], run_path: str) -> bool:
        for _ in range(5):
            latest, generation = self._read_optional("latest.json")
            candidate = (commit["as_of_session"], commit["run_id"])
            current = ((latest or {}).get("as_of_session", ""), (latest or {}).get("run_id", ""))
            if candidate <= current:
                return candidate == current
            value = {"schema_version": "baseline_latest.v1", "as_of_session": commit["as_of_session"],
                     "week_id": commit["week_id"], "run_id": commit["run_id"],
                     "run_path": run_path, "commit_uri": commit["commit_uri"],
                     "updated_at": datetime.now(timezone.utc).isoformat()}
            try:
                self._put_json("latest.json", value, generation)
                return True
            except PointerConflict:
                continue
        raise PointerConflict("latest.json retry limit")

    def latest(self) -> dict[str, Any] | None:
        pointer, _ = self._read_optional("latest.json")
        if not pointer:
            return None
        commit, _ = self._read_optional(f"{pointer['run_path']}/commit.json")
        if not commit:
            raise ValueError("latest points to uncommitted bundle")
        self._verify_commit(commit)
        summary_uri = commit["objects"]["summary.json"]["uri"]
        name = summary_uri.split(f"gs://{self.bucket_name}/", 1)[1]
        return self._json(self.bucket.blob(name).download_as_bytes())

    def history(self, *, include_revisions: bool = False) -> list[dict[str, Any]]:
        index, _ = self._read_optional("index.json")
        records = list((index or {}).get("records", []))
        if include_revisions:
            return records
        selected: dict[str, dict[str, Any]] = {}
        for item in records:
            selected[item["week_id"]] = item
        return sorted(selected.values(), key=lambda item: item["as_of_session"])


__all__ = ["GCSBaselineStore", "PointerConflict"]
