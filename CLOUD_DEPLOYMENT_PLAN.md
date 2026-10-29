# 🚀 Cloud Deployment & Production RAG Architecture Plan
**Branch:** `feature/prod_rag_application_cloud`  
**Target Infrastructure:** GCP Cloud Run + Supabase (`pgvector`) + Groq / Llama-3.3-70B  
**Embedding Engine:** `BAAI/bge-m3` (1024 dimensions, 8,192 token context)

---

## 📋 Table of Contents
1. [Architecture Overview](#1-architecture-overview)
2. [Phase 1: Supabase Database Integration (PostgreSQL + pgvector)](#phase-1-supabase-database-integration)
3. [Phase 2: Local Docker Testing on 8GB RAM Laptop](#phase-2-local-docker-testing-on-8gb-ram-laptop)
4. [Phase 3: Security, CORS & Rate Limiting](#phase-3-security-cors--rate-limiting)
5. [Phase 4: Authentication & Conversational Chat Memory](#phase-4-authentication--conversational-chat-memory)
6. [Phase 5: GCP Cloud Run Deployment](#phase-5-gcp-cloud-run-deployment)
7. [Checklist for Tomorrow](#checklist-for-tomorrow)

---

## 1. Architecture Overview

```
                      [ User Browser ]
                             │
                             ▼ (HTTPS)
       ┌──────────────────────────────────────────────┐
       │           GCP Cloud Run Container            │
       │     (Allocated: 4 GB RAM, 2 vCPUs)           │
       │                                              │
       │  ┌────────────────────┐ ┌─────────────────┐  │
       │  │ HTML/CSS/JS UI     │ │ FastAPI Engine  │  │
       │  │ (/app/index.html)  │ │ (/api/v1/...)   │  │
       │  └────────────────────┘ └────────┬────────┘  │
       │                                  │           │
       │  ┌───────────────────────────────┴────────┐  │
       │  │ BAAI/bge-m3 Model (Local Cache)        │  │
       │  │ (1024-dim dense embeddings + PyTorch)  │  │
       │  └────────────────────────────────────────┘  │
       └──────────────────────┬───────────────────────┘
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
   [ Supabase Cloud Postgres ]        [ Groq LPU Cloud ]
   - documents & parent_chunks        - openai/gpt-oss-20b or
   - child_chunks (pgvector 1024d)      llama-3.3-70b-versatile
   - chat_sessions & messages         - Fast legal reasoning
   - Supabase Auth (JWT)
```

---

## Phase 1: Supabase Database Integration

### 1.1 Supabase Schema Migration (`pgvector`)
Run this in the Supabase SQL Editor:
```sql
-- 1. Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Documents Registry Table
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    file_hash TEXT UNIQUE NOT NULL,
    file_type TEXT DEFAULT 'pdf',
    total_pages INTEGER NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    version INTEGER DEFAULT 1,
    is_active INTEGER DEFAULT 1,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. Parent Chunks (Relational Ground Truth Context)
CREATE TABLE IF NOT EXISTS parent_chunks (
    parent_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    content_type TEXT DEFAULT 'text',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 4. Child Chunks with 1024-dimensional BGE-M3 Vector Embeddings
CREATE TABLE IF NOT EXISTS child_chunks (
    child_id TEXT PRIMARY KEY,
    parent_id TEXT NOT NULL REFERENCES parent_chunks(parent_id) ON DELETE CASCADE,
    doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1024) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 5. HNSW High-Speed Vector Index (< 5ms Cosine Similarity Search)
CREATE INDEX IF NOT EXISTS idx_child_vector_hnsw 
ON child_chunks USING hnsw (embedding vector_cosine_ops);

-- 6. Ingestion Background Jobs Table
CREATE TABLE IF NOT EXISTS ingestion_jobs (
    job_id TEXT PRIMARY KEY,
    doc_id TEXT,
    filename TEXT NOT NULL,
    status TEXT NOT NULL,
    progress_percentage REAL DEFAULT 0.0,
    retry_count INTEGER DEFAULT 0,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

### 1.2 Connection Pooling in Cloud Run
* **Port 5432:** Direct connection (session limit easily reached in serverless).
* **Port 6543:** **PgBouncer Transaction Pooler** (MUST use this for Cloud Run!):
  ```ini
  DATABASE_URL=postgresql://postgres:[PASSWORD]@db.[PROJECT-REF].supabase.co:6543/postgres?pgbouncer=true
  ```

---

## Phase 2: Local Docker Testing on 8GB RAM Laptop

### 2.1 Protect Your 8GB Laptop from RAM Overload
Create `C:\Users\Dell\.wslconfig` (if using WSL2/Docker Desktop):
```ini
[wsl2]
memory=4GB
swap=2GB
```

### 2.2 Local Docker Run Settings
* In `.env`, set:
  ```ini
  EMBEDDING_BATCH_SIZE=32   # Keeps RAM usage under 2.5 GB during local tests
  ```

### 2.3 Production `Dockerfile` Blueprint
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system utilities & C++ dependencies for PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-cache BGE-M3 model weights inside image (prevents 2.2GB download on every container boot)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

# Copy application source & frontend
COPY . .

# Run Uvicorn binding dynamically to Cloud Run $PORT
ENV PORT=8000
CMD uvicorn src.main:app --host 0.0.0.0 --port ${PORT}
```

---

## Phase 3: Security, CORS & Rate Limiting

### 3.1 Strict CORS Configuration
In [src/main.py](file:///e:/2026/ET_Hackthon/EThack/src/main.py):
```python
# Replace allow_origins=["*"] with production origins
ALLOWED_ORIGINS = [
    "http://localhost:8000",
    "https://your-cloud-run-service.run.app",
    "https://your-custom-domain.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)
```

### 3.2 Rate Limiting with `slowapi`
Add `slowapi` to `requirements.txt`:
* **Query Endpoint:** Max 20 queries / minute per IP (prevents Groq TPM exhaustion).
* **Upload Endpoint:** Max 5 PDF uploads / minute per IP.

---

## Phase 4: Authentication & Conversational Chat Memory

### 4.1 Authentication (Supabase Auth)
* Use Supabase's pre-built JWT authentication:
  * Users log in on frontend & receive JWT.
  * FastApi validates `Authorization: Bearer <JWT>` header via Supabase public key.

### 4.2 Multi-Turn Chat History Tables
```sql
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    title TEXT DEFAULT 'Compliance Audit Session',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chat_messages (
    message_id TEXT PRIMARY KEY,
    session_id TEXT REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
    role TEXT NOT NULL, -- 'user' or 'assistant'
    content TEXT NOT NULL,
    citations JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```
* **Conversational Memory In RAG:** Include the last 3 turns of dialogue history in the system context so follow-up questions (*"What were the exceptions to that?"*) are answered accurately.

---

## Phase 5: GCP Cloud Run Deployment

Deploy with one command using the Google Cloud CLI:
```bash
gcloud run deploy banking-compliance-rag \
  --source . \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2 \
  --timeout 600 \
  --concurrency 40 \
  --allow-unauthenticated
```

---

## 📝 Checklist for Tomorrow
- [ ] Connect Supabase: paste `DATABASE_URL` in `.env` and run SQL migration.
- [ ] Adapt [src/db.py](file:///e:/2026/ET_Hackthon/EThack/src/db.py) to read from Supabase PostgreSQL instead of SQLite.
- [ ] Add `Dockerfile` with pre-cached `BAAI/bge-m3` weights.
- [ ] Test Docker build and run on your 8GB laptop.
- [ ] Add `slowapi` rate limiting and restrict CORS in [src/main.py](file:///e:/2026/ET_Hackthon/EThack/src/main.py).
- [ ] Add chat history tables and conversational prompting.
