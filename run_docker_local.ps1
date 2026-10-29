# ==============================================================================
# Local Docker Build & Run Automation Script (Windows PowerShell)
# Service: Enterprise Document Intelligence & Compliance Audit RAG Engine
# ==============================================================================

Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "  Document Intelligence RAG - Local Docker Build & Test       " -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan

# Step 1: Verify Docker Daemon is Running
Write-Host "`n[1/4] Checking Docker Daemon status..." -ForegroundColor Yellow
$maxAttempts = 15
$attempt = 1
$dockerReady = $false

while ($attempt -le $maxAttempts) {
    docker ps 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) {
        $dockerReady = $true
        Write-Host "  Docker Daemon is active and ready!" -ForegroundColor Green
        break
    }
    Write-Host "  Waiting for Docker Desktop to finish booting... (Attempt $attempt/$maxAttempts)" -ForegroundColor Gray
    Start-Sleep -Seconds 4
    $attempt++
}

if (-not $dockerReady) {
    Write-Host "`n[ERROR] Docker Desktop is still not responding." -ForegroundColor Red
    Write-Host "Please ensure Docker Desktop is open and the engine whale icon in the taskbar is steady green." -ForegroundColor Yellow
    exit 1
}

# Step 2: Remove any existing test container
Write-Host "`n[2/4] Cleaning up any prior test container..." -ForegroundColor Yellow
docker rm -f rag-engine-test 2>$null | Out-Null

# Step 3: Build the Docker Image Locally
Write-Host "`n[3/4] Building Docker image 'document-intelligence-rag:local'..." -ForegroundColor Yellow
Write-Host "  - Installing PyTorch CPU wheel (lightweight ~180MB)" -ForegroundColor Gray
Write-Host "  - Pre-caching BAAI/bge-m3 weights for instant boot" -ForegroundColor Gray
Write-Host "  - Packaging frontend and backend" -ForegroundColor Gray

docker build -t document-intelligence-rag:local .

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n[ERROR] Docker build failed. Please review the output above." -ForegroundColor Red
    exit 1
}

# Step 4: Run the Docker Container
Write-Host "`n[4/4] Starting container 'rag-engine-test' on port 8080..." -ForegroundColor Yellow
Write-Host "  Note: PROD_MODE=N is used for local container testing so localhost requests are accepted." -ForegroundColor Gray

docker run -d --name rag-engine-test `
  -p 8080:8080 `
  -v "${PWD}/data:/app/data" `
  --env-file .env `
  document-intelligence-rag:local

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n==============================================================" -ForegroundColor Green
    Write-Host "  Container Successfully Started!                             " -ForegroundColor Green
    Write-Host "==============================================================" -ForegroundColor Green
    Write-Host "`nContainer Name: rag-engine-test" -ForegroundColor White
    Write-Host "Port Mapping:   http://localhost:8080" -ForegroundColor White
    Write-Host "Health Check:   http://localhost:8080/api/v1/health" -ForegroundColor White
    Write-Host "Web Interface:  http://localhost:8080/app/index.html" -ForegroundColor Cyan
    Write-Host "`nView live logs with:" -ForegroundColor Gray
    Write-Host "  docker logs -f rag-engine-test`n" -ForegroundColor White
} else {
    Write-Host "`n[ERROR] Failed to start container." -ForegroundColor Red
}
