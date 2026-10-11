"""
Configuration and settings management using Pydantic.
Ensures fail-fast environment validation and type-safe access.
"""

from pathlib import Path
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
    EMBEDDING_BATCH_SIZE: int = Field(default=64, description="Batch size for generating embeddings")
    VECTOR_COLLECTION_NAME: str = Field(default="compliance_bge_m3_v1", description="ChromaDB collection name")

    # Storage Paths
    SQLITE_DB_PATH: Path = Field(default=Path("data/compliance_master.db"), description="Path to SQLite DB")
    CHROMA_DB_PATH: Path = Field(default=Path("data/chroma_storage"), description="Directory for ChromaDB")

    # Parent-Child Chunking Parameters
    CHUNK_CHILD_SIZE: int = Field(default=200, description="Target token size for child search chunks")
    CHUNK_CHILD_OVERLAP: int = Field(default=50, description="Token overlap between child chunks")
    CHUNK_PARENT_SIZE: int = Field(default=1200, description="Target token size for parent context chunks")

    # Automatically loads variables from .env file
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


# Singleton instance accessible across the application
settings = AppSettings()
