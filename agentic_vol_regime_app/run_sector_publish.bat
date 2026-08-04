@echo off
setlocal

rem Usage: run_sector_publish.bat [PROJECT_ID] [BUCKET_NAME] [PREFIX]
rem Command-line values take precedence over PROJECT_ID / BUCKET_NAME and
rem MARKET_MANIFOLD_* environment variables.

set "DEFAULT_PROJECT_ID=marketphysics"
set "DEFAULT_BUCKET_NAME=marketphysics-market-manifold-data"
set "DEFAULT_PREFIX=market-manifold"

set "RESOLVED_PROJECT_ID=%~1"
if not defined RESOLVED_PROJECT_ID if defined PROJECT_ID set "RESOLVED_PROJECT_ID=%PROJECT_ID%"
if not defined RESOLVED_PROJECT_ID if defined MARKET_MANIFOLD_GCP_PROJECT set "RESOLVED_PROJECT_ID=%MARKET_MANIFOLD_GCP_PROJECT%"
if not defined RESOLVED_PROJECT_ID set "RESOLVED_PROJECT_ID=%DEFAULT_PROJECT_ID%"

set "RESOLVED_BUCKET_NAME=%~2"
if not defined RESOLVED_BUCKET_NAME if defined BUCKET_NAME set "RESOLVED_BUCKET_NAME=%BUCKET_NAME%"
if not defined RESOLVED_BUCKET_NAME if defined MARKET_MANIFOLD_GCS_BUCKET set "RESOLVED_BUCKET_NAME=%MARKET_MANIFOLD_GCS_BUCKET%"
if not defined RESOLVED_BUCKET_NAME set "RESOLVED_BUCKET_NAME=%DEFAULT_BUCKET_NAME%"

set "RESOLVED_PREFIX=%~3"
if not defined RESOLVED_PREFIX if defined MARKET_MANIFOLD_GCS_PREFIX set "RESOLVED_PREFIX=%MARKET_MANIFOLD_GCS_PREFIX%"
if not defined RESOLVED_PREFIX set "RESOLVED_PREFIX=%DEFAULT_PREFIX%"

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "REPO_ROOT=%%~fI"

cd /d "%REPO_ROOT%" || (
    echo Failed to switch to repo root: "%REPO_ROOT%"
    exit /b 1
)

python -m agentic_vol_regime_app.data.sector_history_cli update-and-publish-gcs ^
  --project "%RESOLVED_PROJECT_ID%" ^
  --bucket "%RESOLVED_BUCKET_NAME%" ^
  --prefix "%RESOLVED_PREFIX%"

if errorlevel 1 (
    echo Sector update-and-publish command failed with exit code %errorlevel%.
    exit /b %errorlevel%
)
