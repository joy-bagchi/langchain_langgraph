from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

import agentic_vol_regime_app.executors as executors
from agentic_vol_regime_app.app_runtime import run_daily_regime_agent
from agentic_vol_regime_app.data.forecast_gcs import ForecastGCSPublisher
from agentic_vol_regime_app.data.sector_history_gcs import StorageManifestConflictError, StorageObjectMetadata


@dataclass
class FakeStorage:
    objects: dict[tuple[str, str], tuple[bytes, int, str]]
    generation: int = 0

    def bucket_exists(self, bucket: str) -> bool:
        return bucket == "bucket"

    def get_object_metadata(self, bucket: str, object_name: str):
        item = self.objects.get((bucket, object_name))
        return StorageObjectMetadata(bucket, object_name, str(item[1]), len(item[0])) if item else None

    def download_bytes(self, bucket: str, object_name: str) -> bytes:
        return self.objects[(bucket, object_name)][0]

    def upload_bytes(self, *, bucket: str, object_name: str, data: bytes,
                     if_generation_match, content_type: str):
        old = self.objects.get((bucket, object_name))
        if if_generation_match == 0 and old:
            raise StorageManifestConflictError("exists")
        if if_generation_match not in (None, 0) and (not old or str(old[1]) != str(if_generation_match)):
            raise StorageManifestConflictError("race")
        self.generation += 1
        self.objects[(bucket, object_name)] = (data, self.generation, content_type)
        return StorageObjectMetadata(bucket, object_name, str(self.generation), len(data))


def _sample_input(tmp_path: Path) -> dict:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "configs" / "sample_inputs" / "daily_snapshot_watch.json").read_text(encoding="utf-8"))
    payload["report_root"] = str(tmp_path / "reports")
    payload["forecast_publish_enabled"] = True
    return payload


def test_daily_workflow_publishes_verified_forecast_after_report(tmp_path: Path, monkeypatch) -> None:
    storage = FakeStorage({})
    publisher = ForecastGCSPublisher(storage_client=storage, bucket="bucket")
    monkeypatch.setattr(executors, "ForecastGCSPublisher", lambda **_: publisher)

    result = run_daily_regime_agent(input_payload=_sample_input(tmp_path), storage_root=tmp_path / "memory", langsmith_tracing=False)

    publication = result["named_outputs"]["forecast_publication"]
    assert result["status"] == "completed"
    assert publication["verified"] is True
    assert publication["forecast"]["object_name"].endswith("/forecast.json")
    report_bytes = storage.download_bytes("bucket", publication["report"]["object_name"])
    assert report_bytes == Path(result["named_outputs"]["daily_report"]["report_path"]).read_bytes()


def test_publisher_failure_prevents_completed_workflow(tmp_path: Path, monkeypatch) -> None:
    class BrokenPublisher:
        def __init__(self, **_: object) -> None:
            pass

        def publish(self, **_: object):
            raise StorageManifestConflictError("forced publication failure")

    monkeypatch.setattr(executors, "ForecastGCSPublisher", BrokenPublisher)
    result = run_daily_regime_agent(input_payload=_sample_input(tmp_path), storage_root=tmp_path / "memory", langsmith_tracing=False)
    assert result["status"] != "completed"
    assert "forced publication failure" in str(result.get("last_error", ""))


def test_publisher_tool_allowlist_fails_closed(tmp_path: Path, monkeypatch) -> None:
    payload = _sample_input(tmp_path)
    agent = tmp_path / "agent.yaml"
    app_root = Path(__file__).resolve().parents[1]
    agent_source = (app_root / "configs" / "agents" / "daily_regime_orchestrator.yaml").read_text(encoding="utf-8")
    agent.write_text(
        agent_source.replace("workflow_path: ../workflows/daily_belief_report.md", f"workflow_path: {((app_root / 'configs' / 'workflows' / 'daily_belief_report.md').as_posix())}")
        .replace("  - publish_market_physics_forecast\n", ""),
        encoding="utf-8",
    )
    result = run_daily_regime_agent(input_payload=payload, agent_path=agent, storage_root=tmp_path / "memory", langsmith_tracing=False)
    assert result["status"] != "completed"
    assert "not allowed" in str(result.get("last_error", ""))
