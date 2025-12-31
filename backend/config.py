"""Configuration management for the Research Paper RAG System."""

from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Required API keys can be provided via:
    1. Environment variables (ANTHROPIC_API_KEY, VOYAGE_API_KEY, COHERE_API_KEY)
    2. A .env file in the project root

    If keys are not provided, the application will start but API calls will fail.
    """

    # API Keys - optional with None default to allow graceful startup
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key for Claude")
    voyage_api_key: Optional[str] = Field(default=None, description="Voyage AI API key for embeddings")
    cohere_api_key: Optional[str] = Field(default=None, description="Cohere API key for reranking")

    # Database Settings
    database_url: str = Field(default="sqlite:///./data/app.db", description="SQLite database URL")

    # Auth Settings
    jwt_secret: str = Field(default="change-me-in-production", description="Secret key for JWT tokens")
    jwt_expiry_hours: int = Field(default=24, description="JWT token expiry in hours")
    default_username: str = Field(default="admin", description="Default username for single-user mode")
    default_password: str = Field(default="changeme", description="Default password for single-user mode")

    # Reranker Settings
    reranker_model: str = "rerank-v3.5"
    rerank_top_n: int = 15

    # Chunk Type Settings
    abstract_max_tokens: int = 300
    section_max_tokens: int = 2000
    fine_chunk_tokens: int = 500
    fine_chunk_overlap: int = 128

    # Retrieval Settings
    retrieval_top_k: int = 50  # Per chunk type before reranking
    final_top_k: int = 15      # After reranking

    # Query Classification
    enable_query_classification: bool = True
    enable_query_expansion: bool = True

    # Phase 1 Settings
    validation_sample_size: int = 50
    test_queries_path: Path = Path("./data/test_queries.json")
    evaluation_results_path: Path = Path("./data/evaluation_results.json")

    # Qdrant Configuration
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "research_papers"

    # Application Settings
    environment: str = "development"
    log_level: str = "INFO"

    # Paths - pdf_source_dir is optional to allow app startup without indexing
    pdf_source_dir: Optional[Path] = Field(default=None, description="Directory containing PDF files to index")
    processed_data_dir: Path = Path("./processed_data")
    upload_dir: Path = Field(default=Path("./uploads"), description="Directory for uploaded PDF files")

    # Upload settings
    max_upload_size_mb: int = Field(default=50, description="Maximum upload file size in MB")

    # Processing Settings
    chunk_size: int = 512
    chunk_overlap: int = 128
    batch_size: int = 100

    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:3000,http://localhost:5173"

    # Embedding Settings
    embedding_model: str = "voyage-3-large"
    embedding_dimension: int = 1024

    # LLM Settings
    # Main model for answer generation
    claude_model: str = "claude-opus-4-5-20251101"
    # Fast model for HyDE, query rewriting, entity extraction, citation verification
    claude_model_fast: str = "claude-haiku-4-5-20251001"
    # Model for query classification (needs good reasoning but not full opus)
    claude_model_classifier: str = "claude-sonnet-4-5-20250929"
    max_tokens: int = 4096
    temperature: float = 0.7

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",  # Ignore extra env vars
    }

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins from comma-separated string."""
        return [origin.strip() for origin in self.cors_origins.split(",")]

    def validate_api_keys(self) -> None:
        """Validate that required API keys are set. Raises ValueError if missing."""
        missing = []
        if not self.anthropic_api_key:
            missing.append("ANTHROPIC_API_KEY")
        if not self.voyage_api_key:
            missing.append("VOYAGE_API_KEY")
        if not self.cohere_api_key:
            missing.append("COHERE_API_KEY")
        if missing:
            raise ValueError(f"Missing required API keys: {', '.join(missing)}. Set them in .env or environment.")

    def validate_for_indexing(self) -> None:
        """Validate settings required for PDF indexing."""
        if not self.pdf_source_dir:
            raise ValueError("PDF_SOURCE_DIR must be set for indexing operations")
        if not self.pdf_source_dir.exists():
            raise ValueError(f"PDF source directory does not exist: {self.pdf_source_dir}")


def get_settings() -> Settings:
    """Get settings instance. Use this for lazy loading in modules that may not need settings."""
    return Settings()


# Global settings instance - now safe to import even without .env
settings = Settings()
