# 🚀 GCP Cloud Run Deployment & Live Testing Guide

This guide walks you through deploying the **Enterprise Document Intelligence & Compliance Audit RAG Engine** to **Google Cloud Platform (GCP) Cloud Run** and testing all features post-deployment.

---

## 📋 Prerequisites
1. **Google Cloud SDK (`gcloud`)** installed locally, OR use **[Google Cloud Shell](https://shell.cloud.google.com/)** directly in your browser.
2. A GCP project with billing enabled.

---

## ⚡ Deployment Options

### Option A: 1-Click Script (Recommended)

#### On Windows PowerShell:
```powershell
cd e:\2026\ET_Hackthon\EThack
.\deploy_gcp.ps1
```

#### On Linux / macOS / Google Cloud Shell:
```bash
chmod +x deploy_gcp.sh
./deploy_gcp.sh
```

---

### Option B: Manual `gcloud` Command

Run the following commands in your terminal:

```bash
# 1. Authenticate and select your project
gcloud auth login
gcloud config set project YOUR_GCP_PROJECT_ID

# 2. Enable necessary APIs
gcloud services enable run.googleapis.com cloudbuild.googleapis.com

# 3. Build & Deploy directly from source
gcloud run deploy document-intelligence-rag \
  --source . \
  --region asia-south1 \
  --platform managed \
  --memory 4Gi \
  --cpu 2 \
  --timeout 1800 \
  --concurrency 20 \
  --no-cpu-throttling \
  --set-env-vars "DATABASE_URL=postgresql://postgres.ilocytyhyskabofjdupe:8uhqa41JYs1MVUKK@aws-0-ap-south-1.pooler.supabase.com:6543/postgres,GROQ_API_KEY=gsk_your_groq_api_key_here,PROD_MODE=Y,ADMIN_USERNAME=nagahanuma,ADMIN_PASSWORD=Rag@Prod,ADMIN_EMAIL=nagahanuma@compliance.ai,EMBEDDING_PROVIDER=local,EMBEDDING_MODEL=BAAI/bge-m3,EMBEDDING_DEVICE=cpu,LLM_PROVIDER=groq,LLM_MODEL=openai/gpt-oss-120b" \
  --allow-unauthenticated
```

> **Why `asia-south1` (Mumbai)?**
> The Supabase PostgreSQL + pgvector instance is in Mumbai (`aws-0-ap-south-1`). Deploying Cloud Run in `asia-south1` ensures sub-millisecond database queries and fast vector retrieval.

---

## 🧪 Post-Deployment Testing Checklist

Once deployment completes, Cloud Run prints your live HTTPS URL (e.g., `https://document-intelligence-rag-xyz-el.a.run.app`).

### Test 1: Health Probe Check
Open in browser or terminal:
```bash
curl https://document-intelligence-rag-xyz-el.a.run.app/api/v1/health
```
**Expected Response:**
```json
{
  "status": "healthy",
  "prod_mode": true,
  "vector_db": "postgresql+pgvector",
  "embedding_model": "BAAI/bge-m3"
}
```

### Test 2: Access Modern Web Application
Navigate to:
```
https://document-intelligence-rag-xyz-el.a.run.app/app/index.html
```
- Verify the **2026 Obsidian Glass UI** loads cleanly.
- Check title: `Document Intelligence & Compliance Audit RAG Engine`.

### Test 3: Admin & User Authentication
1. Click **Sign In**.
2. Enter default production credentials:
   - **Username:** `nagahanuma`
   - **Password:** `Rag@Prod`
3. Verify the user badge updates in the top right to `nagahanuma (Admin)`.
4. Test Registration:
   - Click **Register**, create a new test user (e.g., `audit_officer_1` / `Compliance2026!`).
   - Try registering the same username/email again: verify unique constraint error message.

### Test 4: Document Ingestion
1. Under **Upload & Ingest Document**, choose any PDF document (financial report, contract, compliance policy, or audit manual).
2. Click **Process Document**.
3. Observe progress: `Extracting text -> BAAI/bge-m3 Embedding -> Storing in Supabase pgvector`.
4. Verify document shows up in the user's active document list.

### Test 5: Universal Multi-Turn Chat
1. Click one of the quick prompt chips or type a question:
   - *"Executive Summary: Summarize the main topics and critical findings in this document."*
   - *"What are the key obligations, compliance dates, and risks?"*
2. Verify:
   - Streaming LLM answer is generated cleanly.
   - Answer draws directly from uploaded document content.

### Test 6: Audit Provenance Verification (Tab 3)
1. Switch to **Tab 3: Audit Provenance & Sources**.
2. Verify:
   - Exact source page numbers are displayed.
   - Hybrid similarity scores (Dense Vector + BM25 Lexical) are visible.
   - Verified verbatim text citations are shown with audit confidence score.

### Test 7: Multi-Tenant Data Isolation
1. Sign out of `nagahanuma`.
2. Sign in as another user (or guest).
3. Verify that documents uploaded by `nagahanuma` are **not** accessible to other users (row-level security and document isolation).
