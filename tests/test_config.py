"""Tests for environment-based application settings."""

from collections.abc import Generator

import pytest
from pydantic import ValidationError

from backend.config import Settings, get_settings

VALID_DB_URL = (
    "postgresql+asyncpg://postgres:changeme@localhost:5433/market_feature_store"
)


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Generator[None, None, None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_settings_loads_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", VALID_DB_URL)
    monkeypatch.setenv("PROVIDER_API_KEY", "test-key")
    monkeypatch.setenv("PROVIDER_API_SECRET", "test-secret")

    settings = Settings(_env_file=None)

    assert str(settings.database_url) == VALID_DB_URL
    assert settings.provider_api_key.get_secret_value() == "test-key"
    assert settings.provider_api_secret.get_secret_value() == "test-secret"


def test_provider_api_key_is_masked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", VALID_DB_URL)
    monkeypatch.setenv("PROVIDER_API_KEY", "secret-value")
    monkeypatch.setenv("PROVIDER_API_SECRET", "secret-secret")

    settings = Settings(_env_file=None)

    assert "secret-value" not in repr(settings)
    assert "secret-secret" not in repr(settings)
    assert settings.provider_api_key.get_secret_value() == "secret-value"


def test_missing_database_url_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("PROVIDER_API_KEY", "test-key")
    monkeypatch.setenv("PROVIDER_API_SECRET", "test-secret")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_missing_provider_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", VALID_DB_URL)
    monkeypatch.setenv("PROVIDER_API_SECRET", "test-secret")
    monkeypatch.delenv("PROVIDER_API_KEY", raising=False)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_missing_provider_api_secret_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", VALID_DB_URL)
    monkeypatch.setenv("PROVIDER_API_KEY", "test-key")
    monkeypatch.delenv("PROVIDER_API_SECRET", raising=False)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_invalid_database_url_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "not-a-url")
    monkeypatch.setenv("PROVIDER_API_KEY", "test-key")
    monkeypatch.setenv("PROVIDER_API_SECRET", "test-secret")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", VALID_DB_URL)
    monkeypatch.setenv("PROVIDER_API_KEY", "test-key")
    monkeypatch.setenv("PROVIDER_API_SECRET", "test-secret")

    settings = Settings(_env_file=None)

    assert settings.environment == "development"
    assert settings.log_level == "INFO"
    assert settings.provider_name == "alpaca"


def test_get_settings_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", VALID_DB_URL)
    monkeypatch.setenv("PROVIDER_API_KEY", "test-key")
    monkeypatch.setenv("PROVIDER_API_SECRET", "test-secret")

    assert get_settings() is get_settings()


def test_invalid_environment_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", VALID_DB_URL)
    monkeypatch.setenv("PROVIDER_API_KEY", "test-key")
    monkeypatch.setenv("PROVIDER_API_SECRET", "test-secret")
    monkeypatch.setenv("ENVIRONMENT", "invalid")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_invalid_log_level_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", VALID_DB_URL)
    monkeypatch.setenv("PROVIDER_API_KEY", "test-key")
    monkeypatch.setenv("PROVIDER_API_SECRET", "test-secret")
    monkeypatch.setenv("LOG_LEVEL", "VERBOSE")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_blank_provider_name_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", VALID_DB_URL)
    monkeypatch.setenv("PROVIDER_API_KEY", "test-key")
    monkeypatch.setenv("PROVIDER_API_SECRET", "test-secret")
    monkeypatch.setenv("PROVIDER_NAME", "   ")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)
