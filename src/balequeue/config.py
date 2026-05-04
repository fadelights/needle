from functools import lru_cache

from pydantic import ConfigDict, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    environment: str = Field("development", env="ENVIRONMENT")

    es_scheme: str = Field("http", env="ES_SCHEME")
    es_host: str = Field("localhost", env="ES_HOST")
    es_port: int = Field(9200, env="ES_PORT")
    es_index: str = Field("documents", env="ES_INDEX")

    minio_root_user: str = Field(..., env="MINIO_ROOT_USER")
    minio_root_password: str = Field(..., env="MINIO_ROOT_PASSWORD")
    minio_secure: bool = Field(False, env="MINIO_SECURE")
    minio_host: str = Field("localhost", env="MINIO_HOST")
    minio_port: int = Field(9000, env="MINIO_PORT")
    minio_console_port: int = Field(9001, env="MINIO_CONSOLE_PORT")

    auth_database_url: str = Field("sqlite:///./data/balequeue.db", env="AUTH_DATABASE_URL")
    jwt_secret_key: str = Field(..., env="JWT_SECRET_KEY")
    auth_algorithm: str = Field("HS256", env="AUTH_ALGORITHM")
    access_token_expire_minutes: int = Field(30, env="ACCESS_TOKEN_EXPIRE_MINUTES")

    openai_api_key: str = Field("", env="OPENAI_API_KEY")
    hf_api_token: str = Field("", env="HF_API_TOKEN")

    embedding_provider: str = Field("openai", env="EMBEDDING_PROVIDER")
    embedding_model: str = Field("text-embedding-3-small", env="EMBEDDING_MODEL")
    embedding_dim: int = Field(1536, env="EMBEDDING_DIM")

    generator_provider: str = Field("openai", env="GENERATOR_PROVIDER")
    generator_model: str = Field("gpt-4o-mini", env="GENERATOR_MODEL")
    max_new_tokens: int = Field(256, env="MAX_NEW_TOKENS")

    chunk_size: int = Field(8, env="CHUNK_SIZE")
    chunk_overlap: int = Field(2, env="CHUNK_OVERLAP")
    top_k: int = Field(5, env="TOP_K")

    balequeue_port: int = Field(8000, env="BALEQUEUE_PORT")

    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
