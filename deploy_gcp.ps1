# ==============================================================================
# Google Cloud Run 1-Click Deployment Script for Windows PowerShell
# Service: Enterprise Document Intelligence & Compliance Audit RAG Engine
# ==============================================================================

param (
    [string]$ProjectId = "",
    [string]$Region = "asia-south1",
    [string]$ServiceName = "document-intelligence-rag"
)

Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host " GCP Cloud Run Deployment: Document Intelligence RAG Engine   " -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan

# Verify Project ID
if (-not $ProjectId) {
    $currentProject = gcloud config get-value project 2>$null
    if ($currentProject -and $currentProject -ne "(unset)") {
        $ProjectId = $currentProject
    } else {
        $ProjectId = Read-Host "Please enter your GCP Project ID"
    }
}

if (-not $ProjectId) {
    Write-Host "[ERROR] GCP Project ID is required. Exiting." -ForegroundColor Red
    exit 1
}

# Dynamically read credentials from local .env if available
$GroqApiKey = ""
$DatabaseUrl = ""

if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#")) {
            $parts = $line.Split("=", 2)
            if ($parts.Length -eq 2) {
                if ($parts[0].Trim() -eq "GROQ_API_KEY") { $GroqApiKey = $parts[1].Trim() }
                if ($parts[0].Trim() -eq "DATABASE_URL") { $DatabaseUrl = $parts[1].Trim() }
            }
        }
    }
}

if (-not $GroqApiKey) {
    $GroqApiKey = Read-Host "Please enter your GROQ_API_KEY"
}

Write-Host "`n[1/3] Setting GCP Project to: $ProjectId" -ForegroundColor Yellow
gcloud config set project $ProjectId

Write-Host "`n[2/3] Enabling Cloud Run & Cloud Build APIs..." -ForegroundColor Yellow
gcloud services enable run.googleapis.com cloudbuild.googleapis.com

Write-Host "`n[3/3] Building container image remotely and deploying to Cloud Run..." -ForegroundColor Yellow
Write-Host "Allocating: 4Gi RAM, 2 vCPUs in region: $Region (Mumbai matching Supabase DB)" -ForegroundColor Gray

gcloud run deploy $ServiceName `
    --source . `
    --region $Region `
    --platform managed `
    --memory 4Gi `
    --cpu 2 `
    --timeout 1800 `
    --concurrency 20 `
    --no-cpu-throttling `
    --set-env-vars "..." `
    --allow-unauthenticated

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n==============================================================" -ForegroundColor Green
    Write-Host " Deployment Successful! " -ForegroundColor Green
    Write-Host "==============================================================" -ForegroundColor Green
    $url = gcloud run services describe $ServiceName --platform managed --region $Region --format 'value(status.url)'
    Write-Host "`nOpen your live application URL in your browser:" -ForegroundColor Cyan
    Write-Host "$url/app/index.html" -ForegroundColor White
} else {
    Write-Host "`n[ERROR] Deployment failed. Please review the Cloud Build error output above." -ForegroundColor Red
}
