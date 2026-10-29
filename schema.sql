-- ==============================================================================
-- BANKING REGULATORY COMPLIANCE RAG ENGINE - PRODUCTION DATABASE SCHEMA
-- Compatible with: PostgreSQL 15+ / Supabase / AWS RDS / AWS Aurora / Azure Flexible / GCP Cloud SQL
-- ==============================================================================
--
-- This script sets up the complete relational and vector database schema required
-- by the RAG compliance engine, including:
-- 1. pgvector extension for dense embeddings (BAAI/bge-m3 1024-dim vectors)
-- 2. Multi-tenant document registry, deduplication hashes, and ownership
-- 3. Two-stage hierarchical retrieval (Parent context chunks & Child vector chunks)
-- 4. High-performance HNSW index for sub-15ms cosine vector similarity search
-- 5. Multi-turn conversational session history with automatic cascading cleanup
-- 6. User accounts, BYOK encryption vaults, resource tracking, and Master Admin seed
--
-- ==============================================================================

-- ------------------------------------------------------------------------------
-- 0. EXTENSIONS & PREREQUISITES
-- ------------------------------------------------------------------------------
-- Cloud Provider Setup Notes:
-- AWS RDS / Aurora: Ensure 'vector' is added to shared_preload_libraries in your DB Parameter Group.
-- Azure Flexible: In Azure Portal -> Server Parameters -> 'azure.extensions' -> Add 'VECTOR'.
-- GCP Cloud SQL: In Database Flags -> Set 'cloudsql.enable_pgvector = on'.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ------------------------------------------------------------------------------
-- 1. USERS & ACCESS CONTROL (Multi-Tenant & Admin Directory)
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT DEFAULT 'user' CHECK (role IN ('user', 'admin')),
    custom_groq_key TEXT,
    custom_openai_key TEXT,
    custom_model TEXT,
    total_queries INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_active_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username ON users(LOWER(username));
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(LOWER(email));
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
CREATE INDEX IF NOT EXISTS idx_users_last_active ON users(last_active_at);

-- ------------------------------------------------------------------------------
-- 2. DOCUMENTS REGISTRY & MULTI-TENANT OWNERSHIP
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    user_id TEXT REFERENCES users(user_id) ON DELETE CASCADE,
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

CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id);
CREATE INDEX IF NOT EXISTS idx_documents_file_hash ON documents(file_hash);
CREATE INDEX IF NOT EXISTS idx_documents_active ON documents(is_active);

-- Multi-Tenant Document Association Table:
-- Allows multiple users to share a deduplicated document (e.g. RBI Master Directions)
-- while maintaining strict tenant-level access isolation and individual deletion.
CREATE TABLE IF NOT EXISTS user_documents (
    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (user_id, doc_id)
);

CREATE INDEX IF NOT EXISTS idx_user_documents_user ON user_documents(user_id);
CREATE INDEX IF NOT EXISTS idx_user_documents_doc ON user_documents(doc_id);

-- ------------------------------------------------------------------------------
-- 3. TWO-STAGE RETRIEVAL: PARENT CONTEXT CHUNKS
-- ------------------------------------------------------------------------------
-- Large context chunks (1,500 tokens) providing statutory and narrative depth
-- injected directly into LLM prompts without vector fragmentation.
CREATE TABLE IF NOT EXISTS parent_chunks (
    parent_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    content_type TEXT DEFAULT 'text',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_parent_doc ON parent_chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_parent_page ON parent_chunks(doc_id, page_number);

-- ------------------------------------------------------------------------------
-- 4. TWO-STAGE RETRIEVAL: CHILD SEARCH VECTORS (pgvector)
-- ------------------------------------------------------------------------------
-- Fine-grained chunks (500 tokens) with 1024-dimensional dense embeddings
-- indexed via HNSW for ultra-fast semantic similarity search.
CREATE TABLE IF NOT EXISTS child_chunks (
    child_id TEXT PRIMARY KEY,
    parent_id TEXT NOT NULL REFERENCES parent_chunks(parent_id) ON DELETE CASCADE,
    doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1024) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_child_parent ON child_chunks(parent_id);
CREATE INDEX IF NOT EXISTS idx_child_doc ON child_chunks(doc_id);

-- HNSW Vector Index:
-- Uses cosine distance ops. Provides sub-15ms vector nearest neighbor retrieval.
CREATE INDEX IF NOT EXISTS idx_child_vector_hnsw 
ON child_chunks 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- ------------------------------------------------------------------------------
-- 5. ASYNCHRONOUS INGESTION JOBS
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ingestion_jobs (
    job_id TEXT PRIMARY KEY,
    doc_id TEXT,
    filename TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('PENDING', 'PROCESSING', 'INDEXING', 'COMPLETED', 'FAILED')),
    progress_percentage REAL DEFAULT 0.0,
    retry_count INTEGER DEFAULT 0,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_job_status ON ingestion_jobs(status);
CREATE INDEX IF NOT EXISTS idx_job_created ON ingestion_jobs(created_at DESC);

-- ------------------------------------------------------------------------------
-- 6. MULTI-TURN CONVERSATION MEMORY (Chat Sessions & Messages)
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT DEFAULT 'anonymous',
    title TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chat_sessions_user ON chat_sessions(user_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS chat_messages (
    message_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    citations JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chat_msg_session ON chat_messages(session_id, created_at ASC);

-- ------------------------------------------------------------------------------
-- 7. SEED PRIMARY ADMINISTRATOR ACCOUNT
-- ------------------------------------------------------------------------------
-- Default master admin:
-- Username: nagahanuma
-- Email: nagahanuma@compliance.ai
-- Password hash for 'Rag@Prod': SHA-256
INSERT INTO users (user_id, username, email, password_hash, role)
VALUES (
    'usr_admin_master',
    'nagahanuma',
    'nagahanuma@compliance.ai',
    'b852994e43fc0e82f4702951dc8399e52516a75f0a3ee93c8303f0b240ff8d17',
    'admin'
)
ON CONFLICT (user_id) DO UPDATE SET 
    role = 'admin';

-- ------------------------------------------------------------------------------
-- 8. STORAGE & HEALTH VERIFICATION QUERIES (Optional Diagnostics)
-- ------------------------------------------------------------------------------
-- View database storage consumed by each table:
-- SELECT relname AS table_name, pg_size_pretty(pg_total_relation_size(relid)) AS total_size
-- FROM pg_catalog.pg_statio_user_tables
-- ORDER BY pg_total_relation_size(relid) DESC;
--
-- View total vector chunks stored:
-- SELECT COUNT(*) AS total_vectors, pg_size_pretty(pg_total_relation_size('child_chunks')) AS vector_table_size
-- FROM child_chunks;
