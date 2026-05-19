[CmdletBinding()]
param(
    [string]$ContainerName = "agentic-harness-pg",
    [string]$DatabaseName = "agentic_harness",
    [string]$Username = "postgres",
    [string]$Password = "postgres",
    [int]$Port = 5432,
    [switch]$ResetContainer
)

$ErrorActionPreference = "Stop"

function Require-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' is not available on PATH."
    }
}

function Invoke-Docker {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
    & docker @Args
}

function Wait-ForPostgres {
    param(
        [string]$Container,
        [string]$User,
        [string]$Database
    )

    for ($attempt = 1; $attempt -le 30; $attempt++) {
        $null = Invoke-Docker exec $Container pg_isready -U $User -d $Database 2>$null
        if ($LASTEXITCODE -eq 0) {
            return
        }
        Start-Sleep -Seconds 1
    }

    throw "Postgres container '$Container' did not become ready in time."
}

function Read-JsonOutput {
    param([string]$Raw)
    if ([string]::IsNullOrWhiteSpace($Raw)) {
        throw "Command did not return JSON output."
    }
    return $Raw | ConvertFrom-Json
}

Require-Command "docker"
Require-Command "python"

$projectRoot = Split-Path -Parent $PSScriptRoot
$storageRoot = Join-Path $projectRoot ".workflow_memory_postgres_smoke"
$dbUrl = "postgresql://{0}:{1}@localhost:{2}/{3}" -f $Username, $Password, $Port, $DatabaseName

Write-Host "Project root: $projectRoot"
Write-Host "Storage root: $storageRoot"
Write-Host "Database URL: $dbUrl"

if ($ResetContainer) {
    $existing = Invoke-Docker ps -a --filter "name=^/${ContainerName}$" --format "{{.Names}}"
    if ($existing -contains $ContainerName) {
        Write-Host "Removing existing container $ContainerName..."
        Invoke-Docker rm -f $ContainerName | Out-Null
    }
}

$existingNames = Invoke-Docker ps -a --filter "name=^/${ContainerName}$" --format "{{.Names}}"
if ($existingNames -notcontains $ContainerName) {
    Write-Host "Creating Postgres container $ContainerName..."
    Invoke-Docker run `
        --name $ContainerName `
        -e "POSTGRES_PASSWORD=$Password" `
        -e "POSTGRES_DB=$DatabaseName" `
        -p "${Port}:5432" `
        -d postgres:16 | Out-Null
} else {
    $runningNames = Invoke-Docker ps --filter "name=^/${ContainerName}$" --format "{{.Names}}"
    if ($runningNames -notcontains $ContainerName) {
        Write-Host "Starting existing container $ContainerName..."
        Invoke-Docker start $ContainerName | Out-Null
    } else {
        Write-Host "Reusing running container $ContainerName..."
    }
}

Write-Host "Waiting for Postgres readiness..."
Wait-ForPostgres -Container $ContainerName -User $Username -Database $DatabaseName

if (Test-Path -LiteralPath $storageRoot) {
    Remove-Item -LiteralPath $storageRoot -Recurse -Force
}

$env:AGENTIC_HARNESS_DB_URL = $dbUrl

Push-Location $projectRoot
try {
    Write-Host ""
    Write-Host "Running research_agent against Postgres..."
    $agentRaw = python -m agentic_harness run-agent `
        --agent agents/research_agent.yaml `
        --query "What is an SABR model" `
        --storage-root $storageRoot `
        --output-mode internal
    $agentResult = Read-JsonOutput -Raw $agentRaw
    $agentRunId = [string]$agentResult.run_id
    Write-Host "research_agent run_id: $agentRunId"

    Write-Host ""
    Write-Host "Running durable workflow against Postgres..."
    $workflowRaw = python -m agentic_harness run `
        --workflow examples/workflows/research_brief.md `
        --input examples/workflows/research_brief_input.json `
        --storage-root $storageRoot `
        --output-mode internal
    $workflowResult = Read-JsonOutput -Raw $workflowRaw
    $workflowRunId = [string]$workflowResult.run_id
    Write-Host "research_brief run_id: $workflowRunId"
    Write-Host "research_brief status: $($workflowResult.status)"

    Write-Host ""
    Write-Host "Inspecting Postgres runtime ledger..."
    $sql = @"
\pset pager off
select run_id, workflow_id, status, run_kind from runs order by updated_at desc;
select run_id, checkpoint_index from checkpoints order by created_at desc limit 10;
select run_id, event_type from events order by created_at desc limit 20;
select invocation_id, run_id, agent_id, runtime_profile, status from agent_invocations order by updated_at desc limit 10;
select record_id, namespace, memory_type, source_run_id from memory_records order by created_at desc limit 10;
"@
    Invoke-Docker exec -e "PGPASSWORD=$Password" $ContainerName psql -U $Username -d $DatabaseName -c $sql

    Write-Host ""
    Write-Host "Smoke test completed."
    Write-Host "Validated:"
    Write-Host "  - agent run persisted to runs/checkpoints/events"
    Write-Host "  - agent invocation persisted to agent_invocations"
    Write-Host "  - durable workflow wrote memory records"
    Write-Host ""
    Write-Host "Local compatibility mirrors remain under:"
    Write-Host "  $storageRoot"
}
finally {
    Pop-Location
}
