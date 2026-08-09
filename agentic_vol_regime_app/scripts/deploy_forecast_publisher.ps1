[CmdletBinding()]
param(
    [string]$ProjectId = "marketphysics",
    [string]$Region = "us-west1",
    [string]$ServiceName = "market-physics-forecast-publisher",
    [string]$BucketName = "marketphysics-market-manifold-data",
    [string]$Prefix = "market-manifold/forecasts"
)

$ErrorActionPreference = "Stop"
$PublisherServiceAccount = "market-physics-forecast-publisher@$ProjectId.iam.gserviceaccount.com"

# Read-only preflight: fail rather than guessing an MCP identity or overwriting a
# naming collision.  Run only after reviewing the resulting plan and IAM policy.
$activeProject = (& gcloud config get-value project 2>$null).Trim()
if ($activeProject -ne $ProjectId) { throw "Active gcloud project '$activeProject' is not '$ProjectId'." }
$mcpServiceAccount = (& gcloud run services describe market-manifold-mcp --region $Region --project $ProjectId --format="value(spec.template.spec.serviceAccountName)").Trim()
if ([string]::IsNullOrWhiteSpace($mcpServiceAccount)) { throw "Could not resolve market-manifold-mcp runtime service account." }
& gcloud storage buckets describe "gs://$BucketName" --project $ProjectId | Out-Null
& gcloud iam service-accounts describe $PublisherServiceAccount --project $ProjectId | Out-Null

& gcloud iam roles describe marketPhysicsForecastPublisherObjects --project $ProjectId 2>$null | Out-Null
if ($LASTEXITCODE -eq 0) {
    & gcloud iam roles update marketPhysicsForecastPublisherObjects --project $ProjectId --file agentic_vol_regime_app/forecast_publisher_storage_role.yaml
} else {
    & gcloud iam roles create marketPhysicsForecastPublisherObjects --project $ProjectId --file agentic_vol_regime_app/forecast_publisher_storage_role.yaml
}
if ($LASTEXITCODE -ne 0) { throw "Could not create or update the narrow publisher storage role." }

$buildCommit = (& git rev-parse HEAD).Trim()
$image = "gcr.io/$ProjectId/$ServiceName`:$buildCommit"
& gcloud builds submit --project $ProjectId --config agentic_vol_regime_app/cloudbuild.forecast-publisher.yaml --substitutions "_IMAGE=$image" .
if ($LASTEXITCODE -ne 0) { throw "Container build failed." }
& gcloud run deploy $ServiceName --project $ProjectId --region $Region --image $image --service-account $PublisherServiceAccount --no-allow-unauthenticated --concurrency 1 --timeout 60s --memory 512Mi --max-instances 2 --set-env-vars "GOOGLE_CLOUD_PROJECT=$ProjectId,MARKET_PHYSICS_FORECAST_GCS_BUCKET=$BucketName,MARKET_PHYSICS_FORECAST_GCS_PREFIX=$Prefix,BUILD_COMMIT_SHA=$buildCommit,FORECAST_PUBLISHER_MAX_REQUEST_BYTES=2097152,FORECAST_PUBLISHER_STORAGE_TIMEOUT_SECONDS=30"
if ($LASTEXITCODE -ne 0) { throw "Cloud Run deployment failed." }

& gcloud run services add-iam-policy-binding $ServiceName --project $ProjectId --region $Region --member "serviceAccount:$mcpServiceAccount" --role roles/run.invoker
if ($LASTEXITCODE -ne 0) { throw "Could not grant the narrow Cloud Run invoker binding." }

# The custom role is deliberately limited to the publisher's read/create/update
# operations. Apply it with a bucket IAM condition before using the service.
$condition = "expression=resource.name.startsWith('projects/_/buckets/$BucketName/objects/$Prefix/'),title=forecast-publisher-prefix,description=Forecast publisher objects only"
& gcloud storage buckets add-iam-policy-binding "gs://$BucketName" --member "serviceAccount:$PublisherServiceAccount" --role "projects/$ProjectId/roles/marketPhysicsForecastPublisherObjects" --condition $condition
if ($LASTEXITCODE -ne 0) { throw "Could not grant prefix-scoped publisher object access." }
& gcloud storage buckets add-iam-policy-binding "gs://$BucketName" --member "serviceAccount:$PublisherServiceAccount" --role roles/storage.legacyBucketReader
if ($LASTEXITCODE -ne 0) { throw "Could not grant bucket metadata access required for readiness of the publisher." }
& gcloud storage buckets get-iam-policy "gs://$BucketName" --format=json | Select-String -SimpleMatch $PublisherServiceAccount | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Could not verify publisher bucket IAM bindings." }

$origin = (& gcloud run services describe $ServiceName --project $ProjectId --region $Region --format="value(status.url)").TrimEnd('/')
Write-Output "FORECAST_PUBLISHER_URL=$origin"
Write-Output "FORECAST_PUBLISHER_AUDIENCE=$origin"
Write-Output "MCP_RUNTIME_SERVICE_ACCOUNT=$mcpServiceAccount"
