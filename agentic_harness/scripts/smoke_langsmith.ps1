[CmdletBinding()]
param(
    [string]$ProjectName = "agentic_harness",
    [string]$AgentPath = "agents/research_analyst.yaml",
    [string]$InputPath = "examples/workflows/research_brief_input.json",
    [string]$StorageRoot = ".workflow_memory_langsmith_smoke",
    [string]$DbUrl = "",
    [int]$RecentWindowSeconds = 180
)

$ErrorActionPreference = "Stop"

function Require-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' is not available on PATH."
    }
}

function Read-JsonOutput {
    param([string]$Raw)
    if ([string]::IsNullOrWhiteSpace($Raw)) {
        throw "Command did not return JSON output."
    }
    return $Raw | ConvertFrom-Json
}

Require-Command "python"

if (-not $env:LANGSMITH_API_KEY) {
    throw "LANGSMITH_API_KEY is not set."
}

$projectRoot = Split-Path -Parent $PSScriptRoot
$resolvedStorageRoot = if ([System.IO.Path]::IsPathRooted($StorageRoot)) { $StorageRoot } else { Join-Path $projectRoot $StorageRoot }
$effectiveDbUrl = if ($DbUrl) { $DbUrl } elseif ($env:AGENTIC_HARNESS_DB_URL) { $env:AGENTIC_HARNESS_DB_URL } else { "" }
$startedAt = [DateTimeOffset]::UtcNow

Write-Host "Project root: $projectRoot"
Write-Host "LangSmith project: $ProjectName"
Write-Host "Storage root: $resolvedStorageRoot"
if ($effectiveDbUrl) {
    Write-Host "Runtime ledger DB: $effectiveDbUrl"
} else {
    Write-Host "Runtime ledger DB: default local SQLite under storage root"
}

if (Test-Path -LiteralPath $resolvedStorageRoot) {
    Remove-Item -LiteralPath $resolvedStorageRoot -Recurse -Force
}

Push-Location $projectRoot
try {
    if ($effectiveDbUrl) {
        $env:AGENTIC_HARNESS_DB_URL = $effectiveDbUrl
    }
    $env:LANGSMITH_TRACING = "true"
    $env:LANGSMITH_PROJECT = $ProjectName

    Write-Host ""
    Write-Host "Running LangSmith-traced agent smoke test..."
    $runRaw = python -m agentic_harness run-agent `
        --agent $AgentPath `
        --input $InputPath `
        --storage-root $resolvedStorageRoot `
        --langsmith-tracing `
        --langsmith-project $ProjectName `
        --output-mode internal
    $runResult = Read-JsonOutput -Raw $runRaw

    $verificationRaw = python -c @"
from datetime import datetime, timedelta, timezone
import json
from langsmith import Client

project_name = r'''$ProjectName'''
window_seconds = int(r'''$RecentWindowSeconds''')
started_at = datetime.fromisoformat(r'''$($startedAt.ToString("o"))''')
threshold = started_at - timedelta(seconds=window_seconds)

client = Client()
runs = list(client.list_runs(project_name=project_name, limit=50))

def to_utc(value):
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)

recent = []
for run in runs:
    start_time = to_utc(getattr(run, "start_time", None))
    if start_time is None or start_time >= threshold:
        recent.append(
            {
                "id": str(getattr(run, "id", "")),
                "name": getattr(run, "name", ""),
                "run_type": getattr(run, "run_type", ""),
                "status": getattr(run, "status", ""),
                "start_time": start_time.isoformat() if start_time else None,
            }
        )

print(json.dumps({"recent_count": len(recent), "recent_runs": recent[:10]}, indent=2))
"@
    $verification = Read-JsonOutput -Raw $verificationRaw

    Write-Host ""
    Write-Host "LangSmith smoke summary:"
    Write-Host "  - agent run status: $($runResult.status)"
    Write-Host "  - harness run_id: $($runResult.run_id)"
    Write-Host "  - LangSmith project: $ProjectName"
    Write-Host "  - recent LangSmith runs found: $($verification.recent_count)"

    if ([int]$verification.recent_count -le 0) {
        throw "LangSmith did not return any recent runs for project '$ProjectName'."
    }

    Write-Host ""
    Write-Host "Recent LangSmith runs:"
    foreach ($run in $verification.recent_runs) {
        Write-Host ("  - {0} | {1} | {2} | {3}" -f $run.id, $run.name, $run.run_type, $run.status)
    }
}
finally {
    Pop-Location
}
