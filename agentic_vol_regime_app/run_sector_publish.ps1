[CmdletBinding()]
param(
    [string]$ProjectId,
    [string]$BucketName,
    [string]$Prefix,
    [switch]$DryRunPublish,
    [switch]$SkipPublishIfAlreadyCurrent
)

$ErrorActionPreference = "Stop"

$defaultProjectId = "marketphysics"
$defaultBucketName = "marketphysics-market-manifold-data"
$defaultPrefix = "market-manifold"

if ([string]::IsNullOrWhiteSpace($ProjectId)) {
    $ProjectId = if (-not [string]::IsNullOrWhiteSpace($env:PROJECT_ID)) {
        $env:PROJECT_ID
    } elseif (-not [string]::IsNullOrWhiteSpace($env:MARKET_MANIFOLD_GCP_PROJECT)) {
        $env:MARKET_MANIFOLD_GCP_PROJECT
    } else {
        $defaultProjectId
    }
}

if ([string]::IsNullOrWhiteSpace($BucketName)) {
    $BucketName = if (-not [string]::IsNullOrWhiteSpace($env:BUCKET_NAME)) {
        $env:BUCKET_NAME
    } elseif (-not [string]::IsNullOrWhiteSpace($env:MARKET_MANIFOLD_GCS_BUCKET)) {
        $env:MARKET_MANIFOLD_GCS_BUCKET
    } else {
        $defaultBucketName
    }
}

if ([string]::IsNullOrWhiteSpace($Prefix)) {
    $Prefix = if ([string]::IsNullOrWhiteSpace($env:MARKET_MANIFOLD_GCS_PREFIX)) {
        $defaultPrefix
    } else {
        $env:MARKET_MANIFOLD_GCS_PREFIX
    }
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$cliArguments = @(
    "-m", "agentic_vol_regime_app.data.sector_history_cli", "update-and-publish-gcs",
    "--project", $ProjectId,
    "--bucket", $BucketName,
    "--prefix", $Prefix
)

if ($DryRunPublish) {
    $cliArguments += "--dry-run-publish"
}

if ($SkipPublishIfAlreadyCurrent) {
    $cliArguments += "--skip-publish-if-already-current"
}

Push-Location $repoRoot
try {
    & python @cliArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Sector update-and-publish command failed with exit code $LASTEXITCODE."
    }
} finally {
    Pop-Location
}
