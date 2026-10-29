"""
Configuration and settings management using Pydantic.
Ensures fail-fast environment validation and type-safe access.
"""

from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    # LLM Settings
    LLM_PROVIDER: str = Field(default="groq", description="Provider: groq | openai | ollama")
    LLM_MODEL: str = Field(default="llama-3.3-70b-versatile", description="Model identifier")
    LLM_TEMPERATURE: float = Field(default=0.0, description="Temperature for generation")
    
    # API Keys
    GROQ_API_KEY: str = Field(default="", description="Free Groq API key")
    OPENAI_API_KEY: str = Field(default="", description="Optional paid OpenAI key")

    # Embedding & Vector Settings
    EMBEDDING_PROVIDER: str = Field(default="local", description="Embedding provider: local | openai")
    EMBEDDING_MODEL: str = Field(default="BAAI/bge-m3", description="HuggingFace model name or OpenAI model")
    EMBEDDING_DEVICE: str = Field(default="cpu", description="PyTorch tensor execution device: 'cpu' | 'cuda'")
    EMBEDDING_BATCH_SIZE: int = Field(default=128, description="Batch size for generating embeddings (128 optimized for 2+ vCPU)")
    VECTOR_COLLECTION_NAME: str = Field(default="compliance_bge_m3_v2", description="ChromaDB collection name")

    # Storage Paths
    DATABASE_URL: Optional[str] = Field(default=None, description="PostgreSQL / Supabase connection URI")
    SQLITE_DB_PATH: Path = Field(default=Path("data/compliance_master.db"), description="Path to SQLite DB")
    CHROMA_DB_PATH: Path = Field(default=Path("data/chroma_storage"), description="Directory for ChromaDB")

    # Parent-Child Chunking Parameters
    CHUNK_CHILD_SIZE: int = Field(default=500, description="Target token size for child search chunks")
    CHUNK_CHILD_OVERLAP: int = Field(default=80, description="Token overlap between child chunks")
    CHUNK_PARENT_SIZE: int = Field(default=1500, description="Target token size for parent context chunks")
    TOP_PARENT_CONTEXTS: int = Field(default=3, description="Number of top parent context chunks injected into LLM prompt")

    # Ingestion & Upload limits
    MAX_UPLOAD_SIZE_MB: int = Field(default=50, description="Maximum allowed file upload size in megabytes")

    # Auth & Admin Configuration
    ADMIN_USERNAME: str = Field(default="nagahanuma", description="Primary administrator username")
    ADMIN_EMAIL: str = Field(default="nagahanuma@compliance.ai", description="Primary administrator email")
    ADMIN_PASSWORD: str = Field(default="Rag@Prod", description="Primary administrator password")
    SECRET_KEY: str = Field(default="rag_super_secret_auth_token_key_2026", description="Secret key for auth tokens")

    # Deployment & Environment Gatekeeper
    # PROD_MODE: "Y" = Production (rejects local requests from localhost / 127.0.0.1)
    #            "N" = Local Development (accepts local requests)
    PROD_MODE: str = Field(default="N", description="Y = Production mode (blocks local requests), N = Local mode")

    # Automatically loads variables from .env file
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


# Singleton instance accessible across the application
settings = AppSettings()
