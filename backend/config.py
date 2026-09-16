"""Configuration management for the Research Paper RAG System."""

from pydantic_settings import BaseSettings
from pydantic import Field, model_validator
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Embedding profiles — the model and its collection as one unit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EmbeddingProfile:
    """An embedding model bound to the one Qdrant collection its vectors live in.

    The model and the collection are a *pair*, never two independent settings.
    A query vector produced by one model and scored against document vectors
    produced by another returns plausible-looking nonsense — no exception, no
    empty result, just silently wrong retrieval.  So there is deliberately no
    way to set the two inconsistently: changing embedding model means changing
    profile, and the collection moves with it.

    Add a profile when you add a model.  Never edit one in place once its
    collection has been built — that is the same mismatch wearing a hat.
    """

    #: How the profile is selected: ``EMBEDDING_PROFILE=<name>``.
    name: str
    #: Voyage model id.
    model: str
    #: Vector size, as the collection was created.
    dimension: int
    #: The collection built by *this* model, and by no other.
    collection: str
    #: ``voyage-context-*``: a paper's chunks go nested to
    #: ``contextualized_embed`` and come back document-aware.  Measured
    #: 2026-09-16: ``client.embed()`` rejects the context models outright and
    #: ``contextualized_embed`` rejects every other model, so the endpoint
    #: follows from the model rather than being a separate choice.
    contextualized: bool = False
    #: Per-document context window, in tokens of the model's own tokenizer.
    #: Measured 2026-09-16: a hard 400, and "contextualized chunk embeddings
    #: do not support truncation".  0 = not applicable.
    context_window_tokens: int = 0
    #: Ceiling on the sum of every document in one request (measured
    #: 2026-09-16: "The max allowed tokens per submitted batch is 120000").
    max_batch_tokens: int = 0


EMBEDDING_PROFILES: Dict[str, "EmbeddingProfile"] = {
    profile.name: profile
    for profile in (
        # The live index: 212,953 points on 2026-09-16.  Keeps serving until
        # the cutover, so this stays the default.
        EmbeddingProfile(
            name="voyage-3-large",
            model="voyage-3-large",
            dimension=1024,
            collection="research_papers",
        ),
        # §3b.8 — contextualized chunk embeddings.  Built by the W1+W2 reindex
        # into its own collection; the cutover is a profile switch.
        EmbeddingProfile(
            name="voyage-context-4",
            model="voyage-context-4",
            dimension=1024,
            collection="research_papers_ctx4",
            contextualized=True,
            context_window_tokens=32_000,
            max_batch_tokens=120_000,
        ),
    )
}

DEFAULT_EMBEDDING_PROFILE = "voyage-3-large"


def get_embedding_profile(name: str) -> EmbeddingProfile:
    """Look up a profile by name, failing with the list of real options."""
    try:
        return EMBEDDING_PROFILES[name]
    except KeyError:
        raise ValueError(
            f"Unknown embedding profile {name!r}. "
            f"Known profiles: {', '.join(sorted(EMBEDDING_PROFILES))}."
        ) from None


def profile_for_collection(collection: str) -> Optional[EmbeddingProfile]:
    """The profile that built ``collection``, if any.

    For a caller holding only a collection name that needs to know which model
    is allowed to write to it or query it.
    """
    for profile in EMBEDDING_PROFILES.values():
        if profile.collection == collection:
            return profile
    return None


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

    # Retrieval Mode
    enable_hybrid_search: bool = Field(
        default=True,
        description=(
            "Fuse dense (Voyage) and sparse (BM25) retrieval with Qdrant's RRF. "
            "Requires the sparse index to be built under the current scheme version; "
            "see backend/scripts/rebuild_bm25_index.py."
        ),
    )

    # Phase 1 Settings
    validation_sample_size: int = 50
    test_queries_path: Path = Path("./data/test_queries.json")
    evaluation_results_path: Path = Path("./data/evaluation_results.json")

    # Qdrant Configuration.  The collection name is NOT here: it belongs to
    # the embedding profile, and is exposed as `qdrant_collection_name` below.
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # Application Settings
    environment: str = "development"
    log_level: str = "INFO"

    # Logging Settings
    log_dir: str = Field(default="logs", description="Directory for log files")
    enable_file_logging: bool = Field(default=True, description="Enable logging to files")
    enable_json_logging: bool = Field(default=False, description="Use JSON format for logs")
    enable_access_logging: bool = Field(default=True, description="Enable detailed API access logging")

    # Paths - pdf_source_dir is optional to allow app startup without indexing
    pdf_source_dir: Optional[Path] = Field(default=None, description="Directory containing PDF files to index")
    processed_data_dir: Path = Path("./processed_data")
    upload_dir: Path = Field(default=Path("/Volumes/ARC/ARC/papers"), description="Directory for uploaded PDF files")

    # Upload settings
    max_upload_size_mb: int = Field(default=200, description="Maximum upload file size in MB")

    # Processing Settings
    chunk_size: int = 512
    chunk_overlap: int = 128
    batch_size: int = 100
    pdf_extraction_timeout: int = Field(
        default=900,
        description="Timeout in seconds for PDF extraction (MinerU). After timeout, falls back to simple extraction."
    )

    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8001
    cors_origins: str = "http://localhost:3000,http://localhost:5173"

    # Embedding Settings — one knob, because the model, its dimension and its
    # collection cannot be chosen independently.  See EmbeddingProfile.
    embedding_profile: str = Field(
        default=DEFAULT_EMBEDDING_PROFILE,
        description=(
            "Embedding profile name: selects the Voyage model, the vector "
            "dimension and the Qdrant collection together. One of: "
            + ", ".join(sorted(EMBEDDING_PROFILES))
        ),
    )

    # Retired settings, declared only to catch a stale environment.  Each used
    # to be settable on its own, which is exactly the inconsistency the profile
    # exists to prevent; a value that disagrees with the active profile is an
    # error rather than something to ignore quietly.
    retired_qdrant_collection_name: Optional[str] = Field(
        default=None, validation_alias="QDRANT_COLLECTION_NAME", exclude=True,
    )
    retired_embedding_model: Optional[str] = Field(
        default=None, validation_alias="EMBEDDING_MODEL", exclude=True,
    )
    retired_embedding_dimension: Optional[int] = Field(
        default=None, validation_alias="EMBEDDING_DIMENSION", exclude=True,
    )

    # LLM Settings
    # Main model for answer generation
    claude_model: str = "claude-opus-5"
    # Fast model for HyDE, query rewriting, entity extraction, citation verification
    claude_model_fast: str = "claude-haiku-4-5"
    # Model for query classification (needs good reasoning but not full opus)
    claude_model_classifier: str = "claude-sonnet-5"
    # Model for the server-side web search tool
    claude_model_web_search: str = "claude-sonnet-5"
    max_tokens: int = 4096

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",  # Ignore extra env vars
    }

    # ------------------------------------------------------------------
    # Embedding profile: model, dimension and collection, never apart
    # ------------------------------------------------------------------

    @property
    def embedding(self) -> EmbeddingProfile:
        """The active profile."""
        return get_embedding_profile(self.embedding_profile)

    @property
    def embedding_model(self) -> str:
        """Voyage model id.  Read-only: set EMBEDDING_PROFILE instead."""
        return self.embedding.model

    @property
    def embedding_dimension(self) -> int:
        """Vector size.  Read-only: set EMBEDDING_PROFILE instead."""
        return self.embedding.dimension

    @property
    def qdrant_collection_name(self) -> str:
        """The collection this model's vectors live in.  Read-only."""
        return self.embedding.collection

    @model_validator(mode="after")
    def _embedding_profile_is_consistent(self) -> "Settings":
        """Reject an environment that still sets the model or collection apart.

        Silently ignoring `QDRANT_COLLECTION_NAME=research_papers` while the
        profile says otherwise would recreate the exact failure the profile
        removes, one .env file later.
        """
        profile = get_embedding_profile(self.embedding_profile)
        for env_name, value, pinned in (
            ("QDRANT_COLLECTION_NAME", self.retired_qdrant_collection_name, profile.collection),
            ("EMBEDDING_MODEL", self.retired_embedding_model, profile.model),
            ("EMBEDDING_DIMENSION", self.retired_embedding_dimension, profile.dimension),
        ):
            if value is not None and str(value) != str(pinned):
                raise ValueError(
                    f"{env_name}={value!r} contradicts EMBEDDING_PROFILE="
                    f"{profile.name!r}, which pins it to {pinned!r}. The "
                    f"embedding model and its collection move together — drop "
                    f"{env_name} from the environment and select a profile "
                    f"instead (known: {', '.join(sorted(EMBEDDING_PROFILES))})."
                )
        return self

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
