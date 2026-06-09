from functools import lru_cache

from pydantic import ConfigDict, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    environment: str = Field("development")

    es_scheme: str = Field("http")
    es_host: str = Field("localhost")
    es_port: int = Field(9200)
    es_index: str = Field("documents")
    es_analyzer: str = Field("english")

    minio_root_user: str = Field(...)
    minio_root_password: str = Field(...)
    minio_secure: bool = Field(False)
    minio_host: str = Field("localhost")
    minio_port: int = Field(9000)
    minio_console_port: int = Field(9001)

    auth_database_url: str = Field("sqlite:///./data/needle.db")
    jwt_secret_key: str = Field(...)
    auth_algorithm: str = Field("HS256")
    access_token_expire_minutes: int = Field(30)

    openai_api_key: str = Field("")
    hf_api_token: str = Field("")

    generator_device: str = Field("cuda:0")

    embedding_provider: str = Field("openai")
    embedding_model: str = Field("text-embedding-3-small")
    embedding_dim: int = Field(1536)

    generator_provider: str = Field("openai")
    generator_model: str = Field("gpt-4o-mini")
    max_new_tokens: int = Field(256)

    chunk_size: int = Field(8)
    chunk_overlap: int = Field(2)
    top_k: int = Field(5)

    needle_port: int = Field(8000)

    model_config = ConfigDict(
        env_file=".env", env_file_encoding="utf-8", dotenv_filtering="only_existing"
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
