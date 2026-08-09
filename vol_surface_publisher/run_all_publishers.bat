@echo off
setlocal

rem Usage: run_all_publishers.bat [PROJECT_ID] [BUCKET_NAME] [MARKET_PRICE_PREFIX] [VOL_REGIME_PREFIX] [VOL_SURFACE_PREFIX] [DRY_RUN]
rem Set DRY_RUN to 1 to contact IBKR but skip all GCS writes and verification.
rem Arguments take precedence over PROJECT_ID / BUCKET_NAME / MARKET_MANIFOLD_* environment variables.

set "DEFAULT_PROJECT_ID=marketphysics"
set "DEFAULT_BUCKET_NAME=marketphysics-market-manifold-data"
set "DEFAULT_MARKET_PRICE_PREFIX=market-manifold"
set "DEFAULT_VOL_REGIME_PREFIX=market-manifold/vol-regime-history"
set "DEFAULT_VOL_SURFACE_PREFIX=market-manifold/option-chain-iv"

set "RESOLVED_PROJECT_ID=%~1"
if not defined RESOLVED_PROJECT_ID if defined PROJECT_ID set "RESOLVED_PROJECT_ID=%PROJECT_ID%"
if not defined RESOLVED_PROJECT_ID if defined MARKET_MANIFOLD_GCP_PROJECT set "RESOLVED_PROJECT_ID=%MARKET_MANIFOLD_GCP_PROJECT%"
if not defined RESOLVED_PROJECT_ID set "RESOLVED_PROJECT_ID=%DEFAULT_PROJECT_ID%"

set "RESOLVED_BUCKET_NAME=%~2"
if not defined RESOLVED_BUCKET_NAME if defined BUCKET_NAME set "RESOLVED_BUCKET_NAME=%BUCKET_NAME%"
if not defined RESOLVED_BUCKET_NAME if defined MARKET_MANIFOLD_GCS_BUCKET set "RESOLVED_BUCKET_NAME=%MARKET_MANIFOLD_GCS_BUCKET%"
if not defined RESOLVED_BUCKET_NAME set "RESOLVED_BUCKET_NAME=%DEFAULT_BUCKET_NAME%"

set "RESOLVED_MARKET_PRICE_PREFIX=%~3"
if not defined RESOLVED_MARKET_PRICE_PREFIX if defined MARKET_MANIFOLD_GCS_PREFIX set "RESOLVED_MARKET_PRICE_PREFIX=%MARKET_MANIFOLD_GCS_PREFIX%"
if not defined RESOLVED_MARKET_PRICE_PREFIX set "RESOLVED_MARKET_PRICE_PREFIX=%DEFAULT_MARKET_PRICE_PREFIX%"

set "RESOLVED_VOL_REGIME_PREFIX=%~4"
if not defined RESOLVED_VOL_REGIME_PREFIX if defined MARKET_MANIFOLD_VOL_REGIME_GCS_PREFIX set "RESOLVED_VOL_REGIME_PREFIX=%MARKET_MANIFOLD_VOL_REGIME_GCS_PREFIX%"
if not defined RESOLVED_VOL_REGIME_PREFIX set "RESOLVED_VOL_REGIME_PREFIX=%DEFAULT_VOL_REGIME_PREFIX%"

set "RESOLVED_VOL_SURFACE_PREFIX=%~5"
if not defined RESOLVED_VOL_SURFACE_PREFIX if defined MARKET_MANIFOLD_VOL_SURFACE_GCS_PREFIX set "RESOLVED_VOL_SURFACE_PREFIX=%MARKET_MANIFOLD_VOL_SURFACE_GCS_PREFIX%"
if not defined RESOLVED_VOL_SURFACE_PREFIX set "RESOLVED_VOL_SURFACE_PREFIX=%DEFAULT_VOL_SURFACE_PREFIX%"

set "DRY_RUN=%~6"
set "PRICE_DRY_RUN_FLAG="
set "REGIME_DRY_RUN_FLAG="
set "SURFACE_DRY_RUN_FLAG="
if "%DRY_RUN%"=="1" set "PRICE_DRY_RUN_FLAG=--dry-run-publish"
if "%DRY_RUN%"=="1" set "REGIME_DRY_RUN_FLAG=--dry-run"
if "%DRY_RUN%"=="1" set "SURFACE_DRY_RUN_FLAG=--dry-run"
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "REPO_ROOT=%%~fI"
cd /d "%REPO_ROOT%" || ( echo Failed to switch to repo root & exit /b 1 )

echo === Market-price delta publisher (existing sector store, including SPY) ===
python -m agentic_vol_regime_app.data.sector_history_cli update-and-publish-gcs --project "%RESOLVED_PROJECT_ID%" --bucket "%RESOLVED_BUCKET_NAME%" --prefix "%RESOLVED_MARKET_PRICE_PREFIX%" --host "127.0.0.1" --port 4001 --client-id 73 %PRICE_DRY_RUN_FLAG%
if errorlevel 1 exit /b %errorlevel%

echo === Volatility-regime publisher (SPY/VIX/VVIX) ===
python -m agentic_vol_regime_app.data.sector_history_cli sync-vol-regime-history-gcs --project "%RESOLVED_PROJECT_ID%" --bucket "%RESOLVED_BUCKET_NAME%" --prefix "%RESOLVED_VOL_REGIME_PREFIX%" --host "127.0.0.1" --port 4001 --client-id 75 %REGIME_DRY_RUN_FLAG%
if errorlevel 1 exit /b %errorlevel%

echo === Option IV-surface publisher (SPY) ===
python -m vol_surface_publisher.cli --project "%RESOLVED_PROJECT_ID%" --bucket "%RESOLVED_BUCKET_NAME%" --prefix "%RESOLVED_VOL_SURFACE_PREFIX%" --host "127.0.0.1" --port 4001 --client-id 74 --symbol SPY %SURFACE_DRY_RUN_FLAG%
if errorlevel 1 exit /b %errorlevel%
