$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    Write-Host "Project root: $projectRoot"
    if ($env:AGENTIC_HARNESS_DB_URL) {
        Write-Host "Runtime ledger DB: $($env:AGENTIC_HARNESS_DB_URL)"
    }
    else {
        Write-Host "Runtime ledger DB: postgresql://postgres:postgres@localhost:5432/agentic_harness"
    }

    python scripts/smoke_semantic_memory.py
}
finally {
    Pop-Location
}
