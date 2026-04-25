from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    es_host: str = Field("localhost", env="ES_HOST")
    es_port: int = Field(9200, env="ES_PORT")
    es_scheme: str = Field("http", env="ES_SCHEME")
    es_index: str = Field("documents", env="ES_INDEX")

    openai_api_key: str = Field(..., env="OPENAI_API_KEY")

    embedding_model: str = Field("text-embedding-3-small", env="EMBEDDING_MODEL")
    embedding_dim: int = Field(1536, env="EMBEDDING_DIM")

    generator_model: str = Field("gpt-4o-mini", env="GENERATOR_MODEL")
    max_new_tokens: int = Field(256, env="MAX_NEW_TOKENS")

    chunk_size: int = Field(512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(64, env="CHUNK_OVERLAP")
    top_k: int = Field(5, env="TOP_K")

    balequeue_port: int = Field(8000, env="BALEQUEUE_PORT")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
