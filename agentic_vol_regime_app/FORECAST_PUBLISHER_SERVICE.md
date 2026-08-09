# Private Cloud Run forecast publisher

This service is the GCS-writing boundary. MarketManifoldPhysics sends the finalized canonical forecast and Markdown report to the private Cloud Run URL; this repository's `ForecastGCSPublisher` alone names, writes, verifies, and advances GCS artifacts. It contains no MCP, Auth0, forecast-consumer, strategy, or trading logic.

The deployed MCP client uses this request envelope (the `forecast` name is required for compatibility):

```json
{"forecast":{"schema_version":"market_physics_forecast.v1","forecast_id":"forecast-...","generated_at":"2026-08-09T00:00:00Z","market_data_as_of":"2026-08-08T20:00:00Z","model":{},"decision_eligible":true,"review_required":false,"provenance":{}},"report_markdown":"# Daily Market Physics Forecast"}
```

It returns a strict receipt with `status` (`published` or `already_exists`), `forecast_id`, `schema_version`, `forecast_uri`, `report_uri`, `manifest_uri`, `decision_eligible`, timezone-aware `published_at`, and `verified`. `published_at` is the service clock after final verification; it is separate from the artifact's `generated_at` and `market_data_as_of`.

`GOOGLE_CLOUD_PROJECT` (default `marketphysics`), `MARKET_PHYSICS_FORECAST_GCS_BUCKET` (default `marketphysics-market-manifold-data`), and `MARKET_PHYSICS_FORECAST_GCS_PREFIX` (default `market-manifold/forecasts`) are server-only. `FORECAST_PUBLISHER_MAX_REQUEST_BYTES` defaults to 2 MiB and is capped at 5 MiB; `FORECAST_PUBLISHER_STORAGE_TIMEOUT_SECONDS` defaults to 30 and is bounded to 1--120 seconds. ADC supplies credentials. Never send destination, credentials, or `dry_run` in a request.

Run locally after installing the service extra:

```powershell
pip install -e .\agentic_vol_regime_app[publisher-service]
uvicorn agentic_vol_regime_app.forecast_publisher_service:create_app --factory --host 0.0.0.0 --port 8080
python -m pytest agentic_vol_regime_app/tests/test_forecast_publisher_service.py
```

`/healthz` is process-only. `/readyz` validates configuration without touching GCS. Non-2xx publish failures are safe JSON errors: 400 malformed JSON, 413 too large, 415 media type, 422 validation, 409 immutable/manifest conflict, 503 storage unavailable, and 500 sanitized unexpected failure. Alert on 409 (investigate a divergent forecast or pointer race), repeated 503, and any 500. Logs never contain payloads, reports, authorization headers, or tokens.

Identical bytes are safe to retry: the publisher reuses immutable objects and verifies the winning manifest. The forecast identity excludes runtime `generated_at`, so reconstruction does not alter identity; callers should retry the exact artifact bytes to reuse the immutable object.

Deployment is intentionally not automatic. The reviewed custom-role definition is `forecast_publisher_storage_role.yaml`; it contains only `storage.objects.create`, `storage.objects.get`, and `storage.objects.update` and is applied on the bucket with a forecast-prefix IAM condition. Then run:

```powershell
.\agentic_vol_regime_app\scripts\deploy_forecast_publisher.ps1
```

The script verifies the active project, bucket, publisher account, and actual `market-manifold-mcp` runtime identity before mutation; it deploys private Cloud Run and grants only that resolved identity `roles/run.invoker` on this service. The publisher account receives the custom, prefix-restricted GCS object role plus target-bucket metadata read needed by the existing publisher's bucket-exists check. It prints the canonical origin (without a trailing slash) twice for the MCP settings:

```text
FORECAST_PUBLISHER_URL=https://...run.app
FORECAST_PUBLISHER_AUDIENCE=https://...run.app
```

Set those exact values in MarketManifoldPhysics and redeploy that service only under separate authorization. Do not append `/v1/forecasts:publish` to the audience. Roll back with `gcloud run services update-traffic market-physics-forecast-publisher --region us-west1 --to-revisions PREVIOUS=100`; inspect traffic/revision state first.
