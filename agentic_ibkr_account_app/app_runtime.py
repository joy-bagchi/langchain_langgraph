"""Runtime wrapper for running the IBKR account app on agentic_harness."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentic_ibkr_account_app._bootstrap import ensure_repo_imports

ensure_repo_imports()

from agentic_harness.agentic_os.platform import build_platform_services

from agentic_ibkr_account_app.data.ibkr_account_client import (
    IBKRAccountConnectionConfig,
    IBKRAccountDataPipe,
    IBKRAccountSnapshotRequest,
    _ensure_thread_event_loop,
    _patch_ib_insync_get_loop,
)


def fetch_ibkr_account_snapshot(
    *,
    input_payload: dict[str, Any],
    storage_root: str | Path | None = None,
    database_url: str | None = None,
    langsmith_tracing: bool | None = None,
    langsmith_api_key: str | None = None,
    langsmith_endpoint: str | None = None,
    langsmith_project: str | None = None,
    langsmith_workspace_id: str | None = None,
    account_data_pipe: IBKRAccountDataPipe | None = None,
) -> dict[str, Any]:
    services = build_platform_services(
        storage_root=storage_root,
        database_url=database_url,
        langsmith_tracing=langsmith_tracing,
        langsmith_api_key=langsmith_api_key,
        langsmith_endpoint=langsmith_endpoint,
        langsmith_project=langsmith_project,
        langsmith_workspace_id=langsmith_workspace_id,
    )
    connection = IBKRAccountConnectionConfig(
        host=str(input_payload.get("host", "127.0.0.1")),
        port=int(input_payload.get("port", 4001)),
        client_id=int(input_payload.get("client_id", 91)),
        readonly=bool(input_payload.get("readonly", True)),
        timeout_seconds=float(input_payload.get("timeout_seconds", 10.0)),
    )
    pipe = account_data_pipe or IBKRAccountDataPipe(connection=connection)
    request = IBKRAccountSnapshotRequest.from_payload(input_payload)
    with services.observability.trace_span(
        "agentic_ibkr_account_app:fetch_ibkr_account_snapshot",
        run_type="chain",
        inputs={
            "account": request.account,
            "host": connection.host,
            "port": connection.port,
            "client_id": connection.client_id,
            "readonly": connection.readonly,
            "max_fills": request.max_fills,
            "max_orders": request.max_orders,
        },
        tags=["agentic_ibkr_account_app", "ibkr_account_snapshot"],
        metadata={"application": "agentic_ibkr_account_app"},
    ) as app_span:
        snapshot = pipe.fetch_account_snapshot(request)
        result = snapshot.to_dict()
        if hasattr(app_span, "end"):
            app_span.end(outputs=result)
    return result


def fetch_ibkr_option_chain(
    *,
    input_payload: dict[str, Any],
) -> dict[str, Any]:
    _ensure_thread_event_loop()
    _patch_ib_insync_get_loop()

    from agentic_vol_regime_app.data.ibkr_client import (
        IBKRConnectionConfig,
        IBKRDataPipe,
        IBKROptionChainRequest,
    )

    pipe = IBKRDataPipe(
        connection=IBKRConnectionConfig(
            host=str(input_payload.get("host", "127.0.0.1")),
            port=int(input_payload.get("port", 4001)),
            client_id=int(input_payload.get("client_id", 91)),
            readonly=bool(input_payload.get("readonly", True)),
            timeout_seconds=float(input_payload.get("timeout_seconds", 10.0)),
            market_data_type=int(input_payload.get("market_data_type", 1)),
        )
    )
    observation = pipe.fetch_market_snapshot(
        IBKROptionChainRequest(
            symbol=str(input_payload.get("symbol", "SPY")),
            exchange=str(input_payload.get("exchange", "SMART")),
            currency=str(input_payload.get("currency", "USD")),
            option_exchange=str(input_payload.get("option_exchange", "SMART")),
            rights=("C", "P"),
            expiry_count=int(input_payload.get("expiry_count", 6)),
            strike_count=int(input_payload.get("strike_count", 25)),
            expirations=tuple(str(item) for item in input_payload.get("expirations", [])),
            strikes=tuple(float(item) for item in input_payload.get("strikes", [])),
            min_days_to_expiry=int(input_payload.get("min_days_to_expiry", 0)),
        )
    )
    return observation.to_dict() if hasattr(observation, "to_dict") else dict(observation)
