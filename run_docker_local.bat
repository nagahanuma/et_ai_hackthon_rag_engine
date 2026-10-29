@echo off
REM ==============================================================================
REM Windows Command Prompt (CMD.EXE) 1-Click Docker Runner
REM Service: Enterprise Document Intelligence & Compliance Audit RAG Engine
REM ==============================================================================

echo ==============================================================
echo   Document Intelligence RAG - Local Docker Runner (CMD)
echo ==============================================================

echo.
echo [1/3] Removing prior test container if running...
docker rm -f rag-engine-test >nul 2>&1

echo.
echo [2/3] Building latest Docker image...
docker build -t document-intelligence-rag:local .

echo.
echo [3/3] Starting container 'rag-engine-test' on port 8080...
docker run -d --name rag-engine-test ^
  -p 8080:8080 ^
  -v "%cd%/data:/app/data" ^
  --env-file .env ^
  document-intelligence-rag:local

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ==============================================================
    echo   Container Started Successfully!
    echo ==============================================================
    echo Web App URL:    http://localhost:8080/app/index.html
    echo Health Status:  http://localhost:8080/api/v1/health
    echo Container Name: rag-engine-test
    echo.
    echo To view logs, run:
    echo   docker logs -f rag-engine-test
) else (
    echo.
    echo [ERROR] Failed to start container. Ensure Docker Desktop is running.
)
