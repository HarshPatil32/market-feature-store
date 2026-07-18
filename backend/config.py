"""Application settings loaded from environment variables and `.env`."""

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, PostgresDsn, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Annotated[
    str,
    Field(pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"),
]

ProviderName = Annotated[
    str,
    Field(min_length=1),
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
    provider_name: ProviderName = "alpaca"
    environment: Literal["development", "staging", "production"] = "development"
    log_level: LogLevel = "INFO"

    @field_validator("provider_name")
    @classmethod
    def validate_provider_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("provider_name must not be blank")
        return normalized


@lru_cache
def get_settings() -> Settings:
    return Settings()
