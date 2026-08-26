from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):

    # --- API Keys ---
    GROQ_API_KEY: str
    COHERE_API_KEY: str

    # --- Qdrant ---
    QDRANT_URL: str
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION: str = "rich_dad_poor_dad"

    # --- Redis ---
    REDIS_URL: str = "redis://localhost:6379"

    # --- Monitoring ---
    SENTRY_DSN: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_PUBLIC_KEY: str = ""

    # --- App ---
    ENVIRONMENT: str = "development"

    # --- Embedding ---
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384

    # --- Chunking ---
    CHUNK_OVERLAP_PERCENT: float = 0.15

    # --- Retrieval ---
    TOP_K: int = 5

    # --- LLM ---
    LLM_MODEL: str = "llama-3.1-70b-versatile"
    MAX_TOKENS: int = 1024
    TEMPERATURE: float = 0.1

    # --- Memory ---
    MAX_HISTORY_TURNS: int = 10
    SUMMARY_THRESHOLD: int = 8

    # --- Paths (plain properties, no pydantic decorator needed) ---
    @property
    def BASE_DIR(self) -> Path:
        return Path(__file__).resolve().parent.parent.parent

    @property
    def RAW_DIR(self) -> Path:
        return self.BASE_DIR / "data" / "raw"

    @property
    def PROCESSED_DIR(self) -> Path:
        return self.BASE_DIR / "data" / "processed"

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()