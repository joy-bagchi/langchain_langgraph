from pathlib import Path


BATCH_PATH = Path(__file__).resolve().parents[1] / "run_all_publishers.bat"


def test_svrpo_stage_uses_cboe_cli_and_preserves_existing_stages() -> None:
    batch = BATCH_PATH.read_text(encoding="utf-8")

    assert "[DRY_RUN]" in batch.splitlines()[3]
    assert "MARKET_MANIFOLD_SVRPO_GCS_PREFIX" in batch
    assert "market-manifold/svrpo-history" in batch
    assert "SVRPO_DRY_RUN_FLAG=--dry-run" in batch
    assert (
        "python -m agentic_vol_regime_app.data.cboe_svrpo_history "
        '--output "%SVRPO_PARQUET%" --metadata-output "%SVRPO_METADATA%"'
    ) in batch
    assert "%SVRPO_DRY_RUN_FLAG%" in batch
    assert "--client-id 76" not in batch
    assert "--symbols SVRPO" not in batch
    assert "if errorlevel 1 exit /b %errorlevel%" in batch

    assert "sector_history_cli update-and-publish-gcs" in batch
    assert "sector_history_cli sync-vol-regime-history-gcs" in batch
    assert "vol_surface_publisher.cli" in batch
    assert "--client-id 73" in batch
    assert "--client-id 75" in batch
    assert "--client-id 74" in batch


def test_svrpo_stage_quotes_isolated_paths_and_keeps_six_arguments() -> None:
    batch = BATCH_PATH.read_text(encoding="utf-8")

    assert "[VOL_SURFACE_PREFIX] [DRY_RUN]" in batch.splitlines()[3]
    assert 'set "DRY_RUN=%~6"' in batch
    assert '--output "%SVRPO_PARQUET%"' in batch
    assert '--metadata-output "%SVRPO_METADATA%"' in batch
