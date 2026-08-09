from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from agentic_vol_regime_app.data.forecast_gcs import ForecastGCSPublisher
from agentic_vol_regime_app.data.sector_history_gcs import (
    StorageManifestConflictError,
    StorageNetworkError,
    StorageObjectMetadata,
)
from agentic_vol_regime_app.forecast_publisher_service import PublisherSettings, create_app


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

    def upload_bytes(self, *, bucket: str, object_name: str, data: bytes, if_generation_match, content_type: str):
        old = self.objects.get((bucket, object_name))
        if if_generation_match == 0 and old:
            raise StorageManifestConflictError("exists")
        if if_generation_match not in (None, 0) and (not old or str(old[1]) != str(if_generation_match)):
            raise StorageManifestConflictError("race")
        self.generation += 1
        self.objects[(bucket, object_name)] = (data, self.generation, content_type)
        return StorageObjectMetadata(bucket, object_name, str(self.generation), len(data))


def artifact(**overrides: Any) -> dict[str, Any]:
    value = {
        "schema_version": "market_physics_forecast.v1",
        "forecast_id": "forecast-2026-08-08-heuristic-v1-abc123",
        "generated_at": "2026-08-09T00:00:00Z",
        "market_data_as_of": "2026-08-08T20:00:00Z",
        "model": {"name": "heuristic", "version": "v1"},
        "decision_eligible": True,
        "review_required": False,
        "provenance": {"repository_commit_sha": "abc"},
    }
    value.update(overrides)
    return value


def payload(**overrides: Any) -> dict[str, Any]:
    value = {"forecast": artifact(), "report_markdown": "# Daily Market Physics Forecast"}
    value.update(overrides)
    return value


def client(*, storage: FakeStorage | None = None, now=None) -> TestClient:
    storage = storage or FakeStorage({})
    publisher = ForecastGCSPublisher(storage_client=storage, bucket="bucket")
    return TestClient(create_app(
        settings=PublisherSettings("project", "bucket", "market-manifold/forecasts"),
        publisher_factory=lambda _: publisher,
        now=now or (lambda: datetime(2026, 8, 9, 0, 16, 4, tzinfo=timezone.utc)),
    ))


def test_valid_publish_is_strict_client_compatible_and_delegates() -> None:
    storage = FakeStorage({})
    response = client(storage=storage).post("/v1/forecasts:publish", json=payload())
    assert response.status_code == 200
    receipt = response.json()
    assert set(receipt) == {"status", "forecast_id", "schema_version", "forecast_uri", "report_uri", "manifest_uri", "decision_eligible", "published_at", "verified"}
    assert receipt["status"] == "published" and receipt["verified"] is True
    assert receipt["published_at"] == "2026-08-09T00:16:04Z"
    assert storage.objects[("bucket", "market-manifold/forecasts/runs/forecast-2026-08-08-heuristic-v1-abc123/report.md")][0].startswith(b"# Daily")


def test_identical_retry_is_idempotent() -> None:
    service = client()
    assert service.post("/v1/forecasts:publish", json=payload()).status_code == 200
    assert service.post("/v1/forecasts:publish", json=payload()).json()["verified"] is True


def test_rejects_destination_controls_and_invalid_canonical_inputs_before_storage() -> None:
    storage = FakeStorage({})
    service = client(storage=storage)
    assert service.post("/v1/forecasts:publish", json=payload(bucket="other")).status_code == 422
    for bad in (
        payload(forecast=artifact(schema_version="wrong")),
        payload(forecast=artifact(forecast_id="../unsafe")),
        payload(forecast=artifact(generated_at="2026-08-09T00:00:00")),
        payload(forecast=artifact(decision_eligible="true")),
        payload(report_markdown=" "),
    ):
        assert service.post("/v1/forecasts:publish", json=bad).status_code == 422
    assert storage.objects == {}


def test_rejects_wrong_content_type_oversize_and_non_finite_json() -> None:
    service = client()
    assert service.post("/v1/forecasts:publish", content=b"{}", headers={"Content-Type": "text/plain"}).status_code == 415
    too_small = TestClient(create_app(settings=PublisherSettings("project", "bucket", "prefix", max_request_bytes=1024)))
    assert too_small.post("/v1/forecasts:publish", content=b"x" * 1025, headers={"Content-Type": "application/json"}).status_code == 413
    body = b'{"forecast":{"schema_version":"market_physics_forecast.v1","forecast_id":"forecast-1","generated_at":"2026-08-09T00:00:00Z","market_data_as_of":"2026-08-08T20:00:00Z","model":{},"decision_eligible":true,"review_required":false,"provenance":{"x":NaN}},"report_markdown":"# ok"}'
    assert service.post("/v1/forecasts:publish", content=body, headers={"Content-Type": "application/json"}).status_code == 400


def test_conflict_storage_and_unexpected_failures_are_sanitized(caplog) -> None:
    class ConflictPublisher:
        def publish(self, **_: Any):
            raise StorageManifestConflictError("sensitive storage detail")

    class OfflinePublisher:
        def publish(self, **_: Any):
            raise StorageNetworkError("sensitive storage detail")

    class BrokenPublisher:
        def publish(self, **_: Any):
            raise RuntimeError("Bearer secret-token")

    settings = PublisherSettings("project", "bucket", "prefix")
    for publisher, status, code in ((ConflictPublisher(), 409, "publication_conflict"), (OfflinePublisher(), 503, "storage_unavailable"), (BrokenPublisher(), 500, "internal_error")):
        response = TestClient(create_app(settings=settings, publisher_factory=lambda _: publisher)).post("/v1/forecasts:publish", json=payload())
        assert response.status_code == status and response.json()["error"]["code"] == code
        assert "sensitive storage detail" not in response.text and "secret-token" not in response.text
    assert "secret-token" not in caplog.text


def test_health_is_storage_free_and_readiness_detects_invalid_configuration() -> None:
    calls = 0
    def factory(_: PublisherSettings):
        nonlocal calls
        calls += 1
        raise AssertionError("health must not construct storage")

    app = create_app(settings=PublisherSettings("project", "bucket", "prefix"), publisher_factory=factory)
    assert TestClient(app).get("/healthz").json() == {"status": "ok"}
    assert calls == 0
    invalid = TestClient(create_app(settings=PublisherSettings("", "bucket", "prefix")))
    assert invalid.get("/readyz").status_code == 503


def test_compatibility_with_actual_mcp_publisher_client_when_available(monkeypatch) -> None:
    """Exercise the separately deployed client's real request and receipt models.

    CI can set MARKET_MANIFOLD_PHYSICS_SOURCE to a checkout at the deployed
    client revision. The service itself deliberately does not depend on that
    repository.
    """
    source = os.getenv("MARKET_MANIFOLD_PHYSICS_SOURCE")
    if not source:
        pytest.skip("MARKET_MANIFOLD_PHYSICS_SOURCE is not configured")
    monkeypatch.syspath_prepend(str(Path(source) / "src"))
    from market_physics.forecast_publisher_client import ForecastPublisherClient

    app = create_app(
        settings=PublisherSettings("project", "bucket", "market-manifold/forecasts"),
        publisher_factory=lambda _: ForecastGCSPublisher(storage_client=FakeStorage({}), bucket="bucket", prefix="market-manifold/forecasts"),
        now=lambda: datetime(2026, 8, 9, tzinfo=timezone.utc),
    )
    settings = SimpleNamespace(
        forecast_publisher_url="http://publisher.test",
        forecast_publisher_audience="http://publisher.test",
        forecast_publisher_timeout_seconds=2,
        forecast_publisher_max_attempts=1,
    )
    async def check() -> None:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://publisher.test") as transport:
            mcp_client = ForecastPublisherClient(settings, id_token_provider=lambda _: "id-token", http_client=transport)
            receipt = await mcp_client.publish(payload())
        assert receipt.verified is True and receipt.schema_version == "market_physics_forecast.v1"
    asyncio.run(check())
