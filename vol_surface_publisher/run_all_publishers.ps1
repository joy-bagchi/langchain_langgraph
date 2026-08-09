[CmdletBinding()]
param(
    [string]$ProjectId,
    [string]$BucketName,
    [string]$MarketPricePrefix,
    [string]$VolRegimePrefix,
    [string]$VolSurfacePrefix,
    [string]$Host = "127.0.0.1",
    [int]$Port = 4001,
    [string]$Symbol = "SPY",
    [switch]$DryRun,
    [switch]$SkipPricePublishIfAlreadyCurrent
)

<#
Runs all three independent publishers from this module directory:
  1. Incremental market-price history (the existing sector store, including SPY)
  2. SPY/VIX/VVIX volatility-regime history
  3. A point-in-time SPY option IV surface

Explicit parameters win over environment variables, which win over defaults.
-DryRun still contacts IBKR; it suppresses GCS writes and verification.
#>

$ErrorActionPreference = "Stop"

function Resolve-Setting([string]$Value, [string[]]$EnvironmentNames, [string]$Default) {
    if (-not [string]::IsNullOrWhiteSpace($Value)) { return $Value }
    foreach ($name in $EnvironmentNames) {
        $candidate = [Environment]::GetEnvironmentVariable($name)
        if (-not [string]::IsNullOrWhiteSpace($candidate)) { return $candidate }
    }
    return $Default
}

$ProjectId = Resolve-Setting $ProjectId @("PROJECT_ID", "MARKET_MANIFOLD_GCP_PROJECT") "marketphysics"
$BucketName = Resolve-Setting $BucketName @("BUCKET_NAME", "MARKET_MANIFOLD_GCS_BUCKET") "marketphysics-market-manifold-data"
$MarketPricePrefix = Resolve-Setting $MarketPricePrefix @("MARKET_MANIFOLD_SECTOR_PRICES_GCS_PREFIX", "MARKET_MANIFOLD_GCS_PREFIX") "market-manifold/sector-prices"
$VolRegimePrefix = Resolve-Setting $VolRegimePrefix @("MARKET_MANIFOLD_VOL_REGIME_GCS_PREFIX") "market-manifold/vol-regime-history"
$VolSurfacePrefix = Resolve-Setting $VolSurfacePrefix @("MARKET_MANIFOLD_VOL_SURFACE_GCS_PREFIX") "market-manifold/option-chain-iv"
$repoRoot = Split-Path -Parent $PSScriptRoot

function Invoke-Publisher([string]$Name, [string[]]$Arguments) {
    Write-Host "`n=== $Name ===" -ForegroundColor Cyan
    & python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE." }
}

Push-Location $repoRoot
try {
    $priceArgs = @("-m", "agentic_vol_regime_app.data.sector_history_cli", "update-and-publish-gcs", "--project", $ProjectId, "--bucket", $BucketName, "--prefix", $MarketPricePrefix, "--host", $Host, "--port", $Port, "--client-id", "73")
    if ($DryRun) { $priceArgs += "--dry-run-publish" }
    if ($SkipPricePublishIfAlreadyCurrent) { $priceArgs += "--skip-publish-if-already-current" }
    Invoke-Publisher "Market-price delta publisher (existing sector store, including SPY)" $priceArgs

    $regimeArgs = @("-m", "agentic_vol_regime_app.data.sector_history_cli", "sync-vol-regime-history-gcs", "--project", $ProjectId, "--bucket", $BucketName, "--prefix", $VolRegimePrefix, "--host", $Host, "--port", $Port, "--client-id", "75")
    if ($DryRun) { $regimeArgs += "--dry-run" }
    Invoke-Publisher "Volatility-regime publisher (SPY/VIX/VVIX)" $regimeArgs

    $surfaceArgs = @("-m", "vol_surface_publisher.cli", "--project", $ProjectId, "--bucket", $BucketName, "--prefix", $VolSurfacePrefix, "--host", $Host, "--port", $Port, "--client-id", "74", "--symbol", $Symbol)
    if ($DryRun) { $surfaceArgs += "--dry-run" }
    Invoke-Publisher "Option IV-surface publisher ($Symbol)" $surfaceArgs
} finally {
    Pop-Location
}
