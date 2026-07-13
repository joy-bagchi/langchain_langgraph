@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "REPO_ROOT=%%~fI"
set "CLI_PATH=%REPO_ROOT%\agentic_vol_regime_app\data\sector_history_cli.py"
set "STORE_DIR=%SCRIPT_DIR%data\market_history"
set "PARQUET_PATH=%STORE_DIR%\sector_prices_daily.parquet"
set "METADATA_PATH=%STORE_DIR%\sector_prices_daily.metadata.json"
set "SYMBOLS=XLK,XLF,XLE,XLY,XLP,XLI,XLB,XLV,XLU,XLRE,SPY"
set "BOOTSTRAP_START_DATE=2015-01-01"

cd /d "%REPO_ROOT%" || (
    echo Failed to switch to repo root: "%REPO_ROOT%"
    pause
    exit /b 1
)

if not exist "%CLI_PATH%" (
    echo Sector history CLI not found: "%CLI_PATH%"
    pause
    exit /b 1
)

if not exist "%STORE_DIR%" (
    mkdir "%STORE_DIR%"
)

if exist "%PARQUET_PATH%" if exist "%METADATA_PATH%" (
    echo Existing sector history store found.
    echo Running delta update through the latest completed session...
    python "%CLI_PATH%" update ^
      --output "%PARQUET_PATH%" ^
      --metadata-output "%METADATA_PATH%" ^
      --symbols "%SYMBOLS%"
    if errorlevel 1 (
        echo Sector history update failed.
        pause
        exit /b 1
    )
    goto :eof
)

echo No existing sector history store found.
echo Running initial bootstrap from %BOOTSTRAP_START_DATE% through the latest completed session...
python "%CLI_PATH%" bootstrap ^
  --output "%PARQUET_PATH%" ^
  --metadata-output "%METADATA_PATH%" ^
  --symbols "%SYMBOLS%" ^
  --start-date "%BOOTSTRAP_START_DATE%"
if errorlevel 1 (
    echo Sector history bootstrap failed.
    pause
    exit /b 1
)
