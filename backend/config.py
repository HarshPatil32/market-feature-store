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
    s3_endpoint_url: str | None = None
    s3_bucket: str | None = None
    s3_access_key_id: SecretStr | None = None
    s3_secret_access_key: SecretStr | None = None
    s3_region: str = "us-east-1"
    raw_payload_s3_threshold_bytes: int = 262_144

    @property
    def s3_enabled(self) -> bool:
        return (
            self.s3_bucket is not None
            and self.s3_access_key_id is not None
            and self.s3_secret_access_key is not None
        )

    @field_validator("s3_bucket", "s3_endpoint_url", mode="before")
    @classmethod
    def empty_optional_str_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("s3_access_key_id", "s3_secret_access_key", mode="before")
    @classmethod
    def empty_optional_secret_to_none(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, SecretStr):
            value = value.get_secret_value()
        if isinstance(value, str) and not value.strip():
            return None
        return value

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
