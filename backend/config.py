"""Application settings loaded from environment variables and `.env`."""

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, PostgresDsn, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Annotated[
    str,
    Field(pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"),
]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    database_url: PostgresDsn
    provider_api_key: SecretStr
    provider_api_secret: SecretStr
    environment: Literal["development", "staging", "production"] = "development"
    log_level: LogLevel = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
