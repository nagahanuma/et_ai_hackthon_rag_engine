#!/usr/bin/env bash
# ==============================================================================
# Google Cloud Run 1-Click Deployment Script for Bash / Cloud Shell
# Service: Enterprise Document Intelligence & Compliance Audit RAG Engine
# ==============================================================================

set -e

SERVICE_NAME="document-intelligence-rag"
REGION="asia-south1"

echo "=============================================================="
echo " GCP Cloud Run Deployment: Document Intelligence RAG Engine   "
echo "=============================================================="

PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT_ID" ] || [ "$PROJECT_ID" = "(unset)" ]; then
    read -p "Please enter your GCP Project ID: " PROJECT_ID
fi

echo "[1/3] Setting GCP Project to: $PROJECT_ID"
gcloud config set project "$PROJECT_ID"

echo "[2/3] Enabling Cloud Run & Cloud Build APIs..."
gcloud services enable run.googleapis.com cloudbuild.googleapis.com

# Load credentials dynamically from .env if present
if [ -f .env ]; then
    GROQ_API_KEY=$(grep -E '^GROQ_API_KEY=' .env | cut -d '=' -f2- | tr -d '"\r')
    DATABASE_URL=$(grep -E '^DATABASE_URL=' .env | cut -d '=' -f2- | tr -d '"\r')
fi

if [ -z "$GROQ_API_KEY" ]; then
    read -p "Please enter your GROQ_API_KEY: " GROQ_API_KEY
fi

echo "[3/3] Building container image remotely and deploying to Cloud Run..."
gcloud run deploy "$SERVICE_NAME" \
    --source . \
    --region "$REGION" \
    --platform managed \
    --memory 4Gi \
    --cpu 2 \
    --timeout 1800 \
    --concurrency 20 \
    --no-cpu-throttling \
    --set-env-vars "DATABASE_URL=$DATABASE_URL,GROQ_API_KEY=$GROQ_API_KEY,PROD_MODE=Y,MAX_UPLOAD_SIZE_MB=50,ADMIN_USERNAME=nagahanuma,ADMIN_PASSWORD=Rag@Prod,ADMIN_EMAIL=nagahanuma@compliance.ai,EMBEDDING_PROVIDER=local,EMBEDDING_MODEL=BAAI/bge-m3,EMBEDDING_DEVICE=cpu,LLM_PROVIDER=groq,LLM_MODEL=openai/gpt-oss-120b" \
    --allow-unauthenticated

URL=$(gcloud run services describe "$SERVICE_NAME" --platform managed --region "$REGION" --format 'value(status.url)')
echo ""
echo "=============================================================="
echo " Deployment Complete!"
echo " Live Web App URL: $URL/app/index.html"
echo " Health Endpoint:  $URL/api/v1/health"
echo "=============================================================="
