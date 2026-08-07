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
    vision_detection_model: str = "google/owlvit-base-patch32"
    vision_classification_model: str = "openai/clip-vit-base-patch32"
    vision_vqa_model: str = "Salesforce/blip-vqa-base"
    vision_detection_threshold: float = 0.15
    vision_low_confidence_threshold: float = 0.4
    max_upload_mb: int = 10
    celery_task_always_eager: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
