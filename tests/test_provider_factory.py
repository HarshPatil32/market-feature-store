"""Tests for the market data provider factory."""

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from backend.config import Settings
from backend.providers.base import ProviderError
from backend.providers.factory import get_market_data_provider
from backend.providers.provider_adapter import AlpacaProvider

VALID_DB_URL = (
    "postgresql+asyncpg://postgres:changeme@localhost:5433/market_feature_store"
)


def _settings(*, provider_name: str = "alpaca") -> Settings:
    return Settings(
        database_url=VALID_DB_URL,
        provider_api_key="env-key-id",
        provider_api_secret="env-secret",
        provider_name=provider_name,
    )


def test_get_market_data_provider_uses_settings_credentials() -> None:
    with patch(
        "backend.providers.factory.AlpacaProvider",
        autospec=True,
    ) as mock_cls:
        mock_cls.return_value = mock_cls
        get_market_data_provider(_settings())

        mock_cls.assert_called_once_with(
            api_key="env-key-id",
            api_secret="env-secret",
        )


def test_get_market_data_provider_normalizes_provider_name() -> None:
    provider = get_market_data_provider(_settings(provider_name=" Alpaca "))

    assert isinstance(provider, AlpacaProvider)


def test_get_market_data_provider_raises_for_unknown_provider() -> None:
    with pytest.raises(ProviderError, match="Unknown provider 'polygon'"):
        get_market_data_provider(_settings(provider_name="polygon"))


def test_get_market_data_provider_raises_when_api_key_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PROVIDER_API_KEY", raising=False)

    with pytest.raises(ValidationError):
        get_market_data_provider(Settings(_env_file=None))


def test_get_market_data_provider_raises_when_api_secret_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PROVIDER_API_SECRET", raising=False)

    with pytest.raises(ValidationError):
        get_market_data_provider(Settings(_env_file=None))
