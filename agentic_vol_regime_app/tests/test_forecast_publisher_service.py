from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import asyncio
import json
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
from agentic_vol_regime_app.forecast_publisher_service import PublisherSettings, create_app, validate_envelope


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


def mpf_finalization_payload(**overrides: Any) -> dict[str, Any]:
    value = {
        "native_forecast": {
            "schema_version": "mpf_native_forecast.v1", "forecast_id": "MPF-TEST-001",
            "posterior_distributions": {"up": 0.4}, "force_field": {"net": -0.2},
            "law_assessment": {"status": "stable"}, "scenarios": [{"name": "base"}],
            "observables": {"breadth": "weakening"}, "evidence": [{"source": "ibkr"}],
            "uncertainty": {"level": "medium"}, "transmission_stages": ["liquidity"],
            "pending_confirmations": ["close"], "sources": [{"name": "ibkr"}],
        },
        "report_markdown": "# Final MPF report",
        "observation": {"as_of": "2026-08-11T20:30:00Z", "source": "ibkr", "schema_version": "observation.v1"},
        "scientific_model": {"family": "market_physics", "name": "force-field", "version": "v1"},
        "run_context": {"workflow_id": "mpf-postclose", "run_id": "run-1", "agent_id": "mpf-agent", "repository_commit": "abc", "source_data_lineage": {"snapshot": "a"}},
        "alert_record": {"requires_human_review": False},
        "critic_review": {"requires_human_review": True},
        "agent_metadata": {"llm_model": "gpt-test"},
        "scientific_semantics": {
            "transition_probabilities": {"status": "not_applicable", "reason": "MPF has no HMM transitions.", "scientific_context": {"transmission_stages": ["liquidity"]}},
            "policy_recommendation": {"status": "not_applicable", "reason": "No MPF policy is issued."},
        },
    }
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
    assert TestClient(app).get("/health").json() == {"status": "ready"}
    assert calls == 0
    invalid = TestClient(create_app(settings=PublisherSettings("", "bucket", "prefix")))
    assert invalid.get("/health").status_code == 503


def test_mpf_finalization_is_storage_free_and_returns_only_validated_envelope() -> None:
    calls = 0

    def factory(_: PublisherSettings):
        nonlocal calls
        calls += 1
        raise AssertionError("MPF finalization must not construct storage")

    app = create_app(
        settings=PublisherSettings("project", "bucket", "prefix"),
        publisher_factory=factory,
        now=lambda: datetime(2026, 8, 11, 21, 0, 1, tzinfo=timezone.utc),
    )
    response = TestClient(app).post("/v1/mpf-finalizations:prepare", json=mpf_finalization_payload())
    assert response.status_code == 200
    envelope = response.json()["envelope"]
    assert envelope["forecast"]["generated_at"] == "2026-08-11T21:00:01Z"
    assert envelope["forecast"]["scientific_payload"] == mpf_finalization_payload()["native_forecast"]
    assert envelope["forecast"]["review_required"] is True
    assert envelope["forecast"]["decision_eligible"] is False
    assert calls == 0


def test_mpf_finalization_reports_all_missing_inputs_and_never_constructs_storage() -> None:
    calls = 0

    def factory(_: PublisherSettings):
        nonlocal calls
        calls += 1
        raise AssertionError("invalid MPF input must not construct storage")

    value = mpf_finalization_payload(
        observation={}, scientific_model={}, run_context={}, alert_record={}, critic_review={}
    )
    response = TestClient(create_app(settings=PublisherSettings("project", "bucket", "prefix"), publisher_factory=factory)).post(
        "/v1/mpf-finalizations:prepare", json=value
    )
    assert response.status_code == 422
    failures = response.json()["error"]["failures"]
    assert len(failures) >= 8
    assert calls == 0


def test_mpf_finalization_request_limit_is_stream_safe_and_rejects_encoded_input(monkeypatch) -> None:
    import agentic_vol_regime_app.forecast_publisher_service as service_module

    called = 0
    real_finalize = service_module.finalize_mpf_publication_envelope

    def record_finalization(**kwargs: Any):
        nonlocal called
        called += 1
        return real_finalize(**kwargs)

    monkeypatch.setattr(service_module, "finalize_mpf_publication_envelope", record_finalization)
    app = create_app(settings=PublisherSettings("project", "bucket", "prefix", max_request_bytes=1024, finalization_max_request_bytes=1024))
    client = TestClient(app)
    exact_payload = {
        "native_forecast": {
            "schema_version": "mpf_native_forecast.v1",
            "forecast_id": "x",
            "posterior_distributions": {"x": 0},
        },
        "report_markdown": "# x",
        "observation": {"as_of": "2026-08-11T20:30:00Z", "source": "x", "schema_version": "x"},
        "scientific_model": {"family": "market_physics", "name": "x", "version": "x"},
        "run_context": {
            "workflow_id": "x",
            "run_id": "x",
            "agent_id": "x",
            "repository_commit": "x",
            "source_data_lineage": {"x": "x"},
        },
        "alert_record": {"requires_human_review": False},
        "critic_review": {"requires_human_review": False},
        "agent_metadata": {"x": "x"},
        "scientific_semantics": {
            "transition_probabilities": {
                "status": "not_applicable",
                "reason": "x",
                "scientific_context": {"transmission_stages": ["x"]},
            },
            "policy_recommendation": {"status": "not_applicable", "reason": "x"},
        },
    }
    exact_payload["_padding"] = ""
    serialized_without_padding = json.dumps(exact_payload, separators=(",", ":")).encode("utf-8")
    exact_payload["_padding"] = "a" * (1024 - len(serialized_without_padding))
    exact_body = json.dumps(exact_payload, separators=(",", ":")).encode("utf-8")
    assert len(exact_body) == 1024
    exact = client.post("/v1/mpf-finalizations:prepare", content=exact_body, headers={"Content-Type": "application/json"})
    assert exact.status_code == 200, exact.json()
    assert called == 1
    for headers in (
        {"Content-Type": "application/json"},
        {"Content-Type": "application/json", "Content-Length": "1"},
        {"Content-Type": "application/json", "Content-Length": "1025"},
    ):
        response = client.post("/v1/mpf-finalizations:prepare", content=b"x" * 1025, headers=headers)
        assert response.status_code == 413 and response.json()["error"]["code"] == "request_too_large"
    encoded = client.post("/v1/mpf-finalizations:prepare", content=b"{}", headers={"Content-Type": "application/json", "Content-Encoding": "gzip"})
    assert encoded.status_code == 415 and encoded.json()["error"]["code"] == "unsupported_content_encoding"
    assert called == 1


def test_mpf_finalization_rejects_chunked_stream_without_content_length_before_finalization(monkeypatch) -> None:
    import agentic_vol_regime_app.forecast_publisher_service as service_module

    monkeypatch.setattr(service_module, "finalize_mpf_publication_envelope", lambda **_: (_ for _ in ()).throw(AssertionError("must not finalize")))
    app = create_app(settings=PublisherSettings("project", "bucket", "prefix", max_request_bytes=1024, finalization_max_request_bytes=1024))

    async def chunks():
        yield b"x" * 512
        yield b"x" * 513

    async def check() -> None:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as transport:
            response = await transport.post("/v1/mpf-finalizations:prepare", content=chunks(), headers={"Content-Type": "application/json"})
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "request_too_large"

    asyncio.run(check())


@pytest.mark.parametrize("target", ("report", "forecast"))
def test_publisher_detects_post_finalization_mutation_before_storage(target: str) -> None:
    prepared = TestClient(create_app(settings=PublisherSettings("project", "bucket", "prefix"))).post(
        "/v1/mpf-finalizations:prepare", json=mpf_finalization_payload()
    ).json()["envelope"]
    if target == "report":
        prepared["report_markdown"] = "# changed"
    else:
        prepared["forecast"]["scientific_payload"]["force_field"] = {"net": 0.9}
    storage = FakeStorage({})
    response = client(storage=storage).post("/v1/forecasts:publish", json=prepared)
    assert response.status_code == 422
    assert storage.objects == {}


def test_existing_non_mpf_hmm_envelope_remains_compatible() -> None:
    validate_envelope(payload())


@pytest.mark.parametrize("mutation", (
    lambda artifact: artifact["provenance"].pop("native_schema_version"),
    lambda artifact: artifact["provenance"].__setitem__("native_schema_version", "mpf_native_forecast.v999"),
    lambda artifact: artifact["provenance"].__setitem__("native_schema_version", None),
))
def test_mpf_markers_cannot_downgrade_validation_or_reach_storage(mutation) -> None:
    prepared = TestClient(create_app(settings=PublisherSettings("project", "bucket", "prefix"))).post(
        "/v1/mpf-finalizations:prepare", json=mpf_finalization_payload()
    ).json()["envelope"]
    mutation(prepared["forecast"])
    storage = FakeStorage({})
    response = client(storage=storage).post("/v1/forecasts:publish", json=prepared)
    assert response.status_code == 422
    assert storage.objects == {}


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
