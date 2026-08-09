"""Private Cloud Run HTTP facade for :class:`ForecastGCSPublisher`.

Cloud Run IAM authenticates callers.  This module deliberately does not parse
or validate bearer tokens; it only validates the bounded publication payload.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from agentic_vol_regime_app.data.forecast_gcs import ForecastGCSPublisher, ForecastPublishResult
from agentic_vol_regime_app.data.sector_history_gcs import (
    BucketMissingError,
    StorageAuthenticationError,
    StorageManifestConflictError,
    StorageNetworkError,
    StorageObjectConflictError,
    StoragePermissionError,
)
from agentic_vol_regime_app.forecast_contract import FORECAST_SCHEMA_VERSION

logger = logging.getLogger(__name__)
DEFAULT_MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_ALLOWED_REQUEST_BYTES = 5 * 1024 * 1024
FORECAST_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,255}$")


class ServiceValidationError(ValueError):
    """An unsafe or malformed caller payload."""


@dataclass(frozen=True, slots=True)
class PublisherSettings:
    project: str
    bucket: str
    prefix: str
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES
    storage_timeout_seconds: int = 30

    @classmethod
    def from_environment(cls) -> "PublisherSettings":
        return cls(
            project=os.getenv("GOOGLE_CLOUD_PROJECT", "marketphysics").strip(),
            bucket=os.getenv("MARKET_PHYSICS_FORECAST_GCS_BUCKET", "marketphysics-market-manifold-data").strip(),
            prefix=os.getenv("MARKET_PHYSICS_FORECAST_GCS_PREFIX", "market-manifold/forecasts").strip("/"),
            max_request_bytes=_integer_environment("FORECAST_PUBLISHER_MAX_REQUEST_BYTES", DEFAULT_MAX_REQUEST_BYTES),
            storage_timeout_seconds=_integer_environment("FORECAST_PUBLISHER_STORAGE_TIMEOUT_SECONDS", 30),
        )

    def validate(self) -> None:
        if not self.project or not self.bucket or not self.prefix:
            raise ValueError("Publisher configuration is incomplete.")
        if not 1024 <= self.max_request_bytes <= MAX_ALLOWED_REQUEST_BYTES:
            raise ValueError("Publisher request size configuration is outside safe bounds.")
        if not 1 <= self.storage_timeout_seconds <= 120:
            raise ValueError("Publisher storage timeout configuration is outside safe bounds.")


def _integer_environment(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _reject_non_finite(value: str) -> None:
    raise ServiceValidationError("JSON numbers must be finite.")


def _utc_timestamp(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ServiceValidationError(f"artifact.{field} must be a UTC timestamp.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ServiceValidationError(f"artifact.{field} must be a UTC timestamp.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ServiceValidationError(f"artifact.{field} must be a UTC timestamp.")


def validate_envelope(payload: Any) -> tuple[dict[str, Any], str]:
    """Validate the deployed MCP client's strict transport envelope."""
    if not isinstance(payload, dict) or set(payload) != {"forecast", "report_markdown"}:
        raise ServiceValidationError("Request must contain only forecast and report_markdown.")
    artifact = payload.get("forecast")
    report_markdown = payload.get("report_markdown")
    if not isinstance(artifact, dict):
        raise ServiceValidationError("forecast must be an object.")
    if not isinstance(report_markdown, str) or not report_markdown.strip():
        raise ServiceValidationError("report_markdown must be a non-empty string.")
    if artifact.get("schema_version") != FORECAST_SCHEMA_VERSION:
        raise ServiceValidationError("forecast.schema_version is unsupported.")
    required = {
        "forecast_id": str,
        "generated_at": str,
        "market_data_as_of": str,
        "model": dict,
        "decision_eligible": bool,
        "review_required": bool,
        "provenance": dict,
    }
    for field, expected_type in required.items():
        value = artifact.get(field)
        if type(value) is not expected_type or (expected_type in {str, dict} and not value):
            raise ServiceValidationError(f"forecast.{field} is missing or invalid.")
    if not FORECAST_ID_PATTERN.fullmatch(artifact["forecast_id"]):
        raise ServiceValidationError("forecast.forecast_id is not path-safe.")
    _utc_timestamp(artifact["generated_at"], "generated_at")
    _utc_timestamp(artifact["market_data_as_of"], "market_data_as_of")
    _assert_finite(artifact)
    return artifact, report_markdown


def _assert_finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ServiceValidationError("JSON numbers must be finite.")
    if isinstance(value, dict):
        for nested in value.values():
            _assert_finite(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_finite(nested)


def _error(status_code: int, code: str, message: str, request_id: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": {"code": code, "message": message, "request_id": request_id}})


def _receipt(result: ForecastPublishResult, artifact: dict[str, Any], now: Callable[[], datetime]) -> dict[str, Any]:
    if not result.verified:
        raise RuntimeError("Publisher did not verify the final objects.")
    published_at = now().astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return {
        "status": "published",
        "forecast_id": result.forecast_id,
        "schema_version": FORECAST_SCHEMA_VERSION,
        "forecast_uri": result.forecast.gs_uri,
        "report_uri": result.report.gs_uri,
        "manifest_uri": result.manifest_uri,
        "decision_eligible": artifact["decision_eligible"],
        "published_at": published_at,
        "verified": True,
    }


def create_app(
    *,
    settings: PublisherSettings | None = None,
    publisher_factory: Callable[[PublisherSettings], ForecastGCSPublisher] | None = None,
    now: Callable[[], datetime] | None = None,
) -> FastAPI:
    """Build an import-safe application; ADC is only touched during publish."""
    app = FastAPI(title="Market Physics Forecast Publisher", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.settings = settings or PublisherSettings.from_environment()
    app.state.settings_error = None
    try:
        app.state.settings.validate()
    except ValueError:
        app.state.settings_error = "invalid_configuration"
    app.state.publisher_factory = publisher_factory or (
        lambda configured: ForecastGCSPublisher(
            bucket=configured.bucket,
            prefix=configured.prefix,
            project=configured.project,
            storage_timeout_seconds=configured.storage_timeout_seconds,
        )
    )
    app.state.clock = now or (lambda: datetime.now(timezone.utc))

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request.state.request_id = str(uuid.uuid4())
        started = time.monotonic()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        logger.info("forecast_publisher_http request_id=%s method=%s path=%s status=%s latency_ms=%d", request.state.request_id, request.method, request.url.path, response.status_code, (time.monotonic() - started) * 1000)
        return response

    @app.get("/health")
    async def readyz(request: Request):
        if app.state.settings_error:
            return _error(503, "not_ready", "Publisher configuration is invalid.", request.state.request_id)
        return {"status": "ready"}

    @app.post("/v1/forecasts:publish")
    async def publish(request: Request):
        request_id = request.state.request_id
        if app.state.settings_error:
            return _error(503, "not_ready", "Publisher configuration is invalid.", request_id)
        content_type = request.headers.get("content-type", "").split(";", 1)[0].lower()
        if content_type != "application/json":
            return _error(415, "unsupported_media_type", "Content-Type must be application/json.", request_id)
        try:
            content_length = int(request.headers.get("content-length", "0"))
        except ValueError:
            return _error(400, "invalid_request", "Invalid request length.", request_id)
        if content_length > app.state.settings.max_request_bytes:
            return _error(413, "request_too_large", "Request exceeds the maximum size.", request_id)
        body = await request.body()
        if len(body) > app.state.settings.max_request_bytes:
            return _error(413, "request_too_large", "Request exceeds the maximum size.", request_id)
        try:
            payload = json.loads(body, parse_constant=_reject_non_finite)
        except (json.JSONDecodeError, UnicodeDecodeError, ServiceValidationError):
            return _error(400, "invalid_request", "Request body must be valid JSON.", request_id)
        try:
            artifact, report_markdown = validate_envelope(payload)
        except ServiceValidationError as exc:
            return _error(422, "forecast_validation_failed", str(exc), request_id)
        try:
            publisher = app.state.publisher_factory(app.state.settings)
            result = await run_in_threadpool(publisher.publish, artifact=artifact, report_markdown=report_markdown)
            return JSONResponse(status_code=200, content=_receipt(result, artifact, app.state.clock))
        except (StorageObjectConflictError, StorageManifestConflictError):
            return _error(409, "publication_conflict", "Forecast publication conflicts with an immutable object or manifest.", request_id)
        except (BucketMissingError, StorageAuthenticationError, StoragePermissionError, StorageNetworkError):
            return _error(503, "storage_unavailable", "Forecast storage is temporarily unavailable.", request_id)
        except Exception:
            logger.error("forecast_publisher_failure request_id=%s forecast_id=%s category=unexpected", request_id, artifact["forecast_id"])
            return _error(500, "internal_error", "Forecast publication failed unexpectedly.", request_id)

    return app
