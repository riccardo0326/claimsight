from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Default to SQLite for local/dev; docker-compose overrides with Postgres.
    db_url: str = "sqlite:///./claimsight.db"
    redis_url: str = "redis://localhost:6379/0"
    storage_dir: str = "./storage"
    doc_qa_model: str = "impira/layoutlm-document-qa"
    doc_qa_min_confidence: float = 0.5
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rag_top_k: int = 5
    max_upload_mb: int = 10
    celery_task_always_eager: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
