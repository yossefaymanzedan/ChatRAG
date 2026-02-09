from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: str = Field(default="ollama", alias="LLM_PROVIDER")

    ollama_base_url: str = Field(default="http://127.0.0.1:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3.2:1b", alias="OLLAMA_MODEL")
    ollama_temperature: float = Field(default=0.1, alias="OLLAMA_TEMPERATURE")

    openai_base_url: str = Field(default="https://api.openai.com/v1", alias="OPENAI_BASE_URL")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")

    deepseek_base_url: str = Field(default="https://api.deepseek.com/v1", alias="DEEPSEEK_BASE_URL")
    deepseek_model: str = Field(default="deepseek-chat", alias="DEEPSEEK_MODEL")
    deepseek_api_key: str = Field(default="", alias="DEEPSEEK_API_KEY")

    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5", alias="EMBEDDING_MODEL")
    embedding_cache_dir: str = Field(default=".rag/models", alias="EMBEDDING_CACHE_DIR")
    embedding_offline: bool = Field(default=False, alias="EMBEDDING_OFFLINE")
    embedding_batch_size: int = Field(default=16, alias="EMBEDDING_BATCH_SIZE")
    chroma_dir: str = Field(default=".rag/chroma", alias="CHROMA_DIR")
    chroma_collection: str = Field(default="chatrag", alias="CHROMA_COLLECTION")
    sqlite_path: str = Field(default=".rag/rag.db", alias="SQLITE_PATH")

    pdf_text_threshold: int = Field(default=40, alias="PDF_TEXT_THRESHOLD")
    ignore_front_matter: bool = Field(default=True, alias="IGNORE_FRONT_MATTER")
    front_matter_scan_pages: int = Field(default=24, alias="FRONT_MATTER_SCAN_PAGES")
    index_summary_timeout_sec: float = Field(default=20.0, alias="INDEX_SUMMARY_TIMEOUT_SEC")
    index_summary_max_points: int = Field(default=8, alias="INDEX_SUMMARY_MAX_POINTS")
    index_summary_excerpt_chars: int = Field(default=200, alias="INDEX_SUMMARY_EXCERPT_CHARS")
    index_summary_char_budget: int = Field(default=1800, alias="INDEX_SUMMARY_CHAR_BUDGET")
    index_summary_skip_llm_chunk_threshold: int = Field(default=1200, alias="INDEX_SUMMARY_SKIP_LLM_CHUNK_THRESHOLD")
    index_vector_upsert_retries: int = Field(default=3, alias="INDEX_VECTOR_UPSERT_RETRIES")
    index_vector_upsert_retry_delay_ms: int = Field(default=250, alias="INDEX_VECTOR_UPSERT_RETRY_DELAY_MS")
    low_confidence_threshold: float = Field(default=0.35, alias="LOW_CONFIDENCE_THRESHOLD")
    hard_not_found_threshold: float = Field(default=0.12, alias="HARD_NOT_FOUND_THRESHOLD")


settings = Settings()


def ensure_runtime_dirs() -> None:
    Path(settings.chroma_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
    Path(settings.embedding_cache_dir).mkdir(parents=True, exist_ok=True)
