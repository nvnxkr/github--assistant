"""
config.py — Central configuration management (Pydantic Settings).
Vector store is FAISS only.
"""

from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


PROJECT_ROOT = Path(__file__).resolve().parent


class LLMConfig(BaseSettings):
    llm_provider: Literal["mistral", "groq", "local"] = Field(default="mistral")
    mistral_api_key: Optional[str] = Field(default=None, env="MISTRAL_API_KEY")
    mistral_model: str = Field(default="mistral-small-latest")
    groq_api_key: Optional[str] = Field(default=None, env="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile")
    local_llm_url: str = Field(default="http://localhost:11434")
    local_llm_model: str = Field(default="llama3")
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    max_tokens: int = Field(default=2048, ge=256)
    max_retries: int = Field(default=2, ge=0, le=10)
    class Config:
        env_file = ".env"
        extra = "ignore"


class EmbeddingConfig(BaseSettings):
    embedding_provider: Literal["sentence_transformers", "mistral"] = Field(default="sentence_transformers")
    sentence_transformer_model: str = Field(default="BAAI/bge-large-en-v1.5")
    mistral_embedding_model: str = Field(default="mistral-embed")
    embedding_dimension: int = Field(default=1024)
    embedding_batch_size: int = Field(default=32)
    class Config:
        env_file = ".env"
        extra = "ignore"


class VectorStoreConfig(BaseSettings):
    """FAISS-only vector store configuration."""
    faiss_index_path: Path = Field(default=PROJECT_ROOT / "data" / "faiss_index")
    top_k: int = Field(default=5, ge=1, le=20)
    mmr_diversity_score: float = Field(default=0.5, ge=0.0, le=1.0)
    hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    class Config:
        env_file = ".env"
        extra = "ignore"


class IngestionConfig(BaseSettings):
    temp_dir: Path = Field(default=PROJECT_ROOT / "data" / "repos")
    github_token: Optional[str] = Field(default=None, env="GITHUB_TOKEN")
    clone_timeout: int = Field(default=120)
    chunk_size: int = Field(default=1500, ge=200)
    chunk_overlap: int = Field(default=200, ge=0)
    max_file_size_mb: float = Field(default=5.0)
    supported_extensions: list[str] = Field(default=[
        ".py", ".js", ".ts", ".tsx", ".jsx",
        ".java", ".cpp", ".c", ".go", ".rs",
        ".php", ".html", ".css", ".json", ".ipynb",
        ".yaml", ".yml", ".md", ".txt", ".sql",
    ])
    ignored_dirs: list[str] = Field(default=[
        "node_modules", ".git", "venv", "__pycache__",
        "dist", "build", ".env", "env", ".venv",
        "coverage", ".nyc_output", "target", "out",
    ])
    class Config:
        env_file = ".env"
        extra = "ignore"


class DatabaseConfig(BaseSettings):
    db_path: Path = Field(default=PROJECT_ROOT / "database" / "logs.db")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    class Config:
        env_file = ".env"
        extra = "ignore"


class AppConfig(BaseSettings):
    app_name: str = Field(default="GitHub Repository Assistant")
    app_version: str = Field(default="1.0.0")
    debug: bool = Field(default=False)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)

    class Config:
        env_file = ".env"
        extra = "ignore"

    def ensure_directories(self) -> None:
        self.ingestion.temp_dir = self.ingestion.temp_dir.resolve()
        self.vector_store.faiss_index_path = self.vector_store.faiss_index_path.resolve()
        self.database.db_path = self.database.db_path.resolve()
        for d in [
            self.ingestion.temp_dir,
            self.vector_store.faiss_index_path,
            self.database.db_path.parent,
            PROJECT_ROOT / "data",
        ]:
            d.mkdir(parents=True, exist_ok=True)


_config: Optional[AppConfig] = None

def get_config() -> AppConfig:
    global _config
    if _config is None:
        _config = AppConfig()
        _config.ensure_directories()
    return _config
