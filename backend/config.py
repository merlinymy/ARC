"""Configuration management for the Research Paper RAG System."""

from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys
    anthropic_api_key: str
    voyage_api_key: str

    # Reranker Settings
    cohere_api_key: str
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

    # Paths
    pdf_source_dir: Path
    processed_data_dir: Path = Path("./processed_data")

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
    claude_model: str = "claude-opus-4-5-20251101"
    max_tokens: int = 4096
    temperature: float = 0.7

    class Config:
        env_file = ".env"
        case_sensitive = False

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins from comma-separated string."""
        return [origin.strip() for origin in self.cors_origins.split(",")]


# Global settings instance
settings = Settings()
