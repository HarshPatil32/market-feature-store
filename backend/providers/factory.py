"""Provider factory for constructing MarketDataProvider instances by name."""

from collections.abc import Callable

from backend.config import Settings, get_settings
from backend.providers.base import MarketDataProvider, ProviderError
from backend.providers.provider_adapter import AlpacaProvider


def _build_alpaca_provider(settings: Settings) -> AlpacaProvider:
    return AlpacaProvider(
        api_key=settings.provider_api_key.get_secret_value(),
        api_secret=settings.provider_api_secret.get_secret_value(),
    )


_PROVIDERS: dict[str, Callable[[Settings], MarketDataProvider]] = {
    "alpaca": _build_alpaca_provider,
}


def get_market_data_provider(settings: Settings | None = None) -> MarketDataProvider:
    """Build a market data provider using configuration from settings/env."""
    resolved_settings = settings or get_settings()
    provider_name = resolved_settings.provider_name.strip().lower()

    builder = _PROVIDERS.get(provider_name)
    if builder is None:
        available = ", ".join(sorted(_PROVIDERS))
        raise ProviderError(
            f"Unknown provider {provider_name!r}. Available providers: {available}"
        )

    return builder(resolved_settings)
