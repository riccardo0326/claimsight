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
    # Calibrated against fixtures/images/real/: with scene labels alone,
    # undamaged peaks ~0.43 on "damaged car"; real damage often clears 0.45+.
    vision_detection_threshold: float = 0.45
    vision_low_confidence_threshold: float = 0.4
    # Fraud/Risk zero-shot (MNLI); see DECISIONS.md Slice 4.
    fraud_zero_shot_model: str = "typeform/distilbert-base-uncased-mnli"
    # Outbound verifier HTTP (Nominatim + NWS require a descriptive User-Agent).
    http_user_agent: str = (
        "ClaimSight/0.1 (portfolio claims triage; contact: claimsight-dev@example.com)"
    )
    external_api_timeout_seconds: float = 5.0
    external_api_max_attempts: int = 2
    nhtsa_cache_ttl_hours: int = 24
    # Precipitation (mm) at/above which weather is treated as a storm-like event.
    weather_storm_precip_mm: float = 5.0
    # Adjudicator frontier LLM (OpenAI Chat Completions via httpx). See DECISIONS.md.
    openai_api_key: str | None = None
    adjudicator_model: str = "gpt-4o"
    adjudicator_base_url: str = "https://api.openai.com/v1"
    adjudicator_timeout_seconds: float = 60.0
    max_upload_mb: int = 10
    celery_task_always_eager: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
