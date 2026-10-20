"""Orchestrate feature computation for one or more symbols."""

from __future__ import annotations

from datetime import datetime

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

import backend.features  # noqa: F401 — register feature implementations
from backend.features.calculator import (
    FeatureDefinitionLookbackMismatchError,
    FeatureDefinitionNotFoundError,
    compute_and_store_features,
)
from backend.features.registry import registry
from backend.services.features import FeatureNotFoundError
from backend.services.symbols import get_symbol

logger = structlog.get_logger(__name__)


def _validate_feature_names(feature_names: list[str] | None) -> None:
    if feature_names is None:
        return
    registered = {feature.spec.name for feature in registry.all()}
    unknown = set(feature_names) - registered
    if unknown:
        raise FeatureNotFoundError(sorted(unknown)[0])


def _failure_message(exc: BaseException) -> str:
    if isinstance(
        exc,
        (
            FeatureDefinitionNotFoundError,
            FeatureDefinitionLookbackMismatchError,
        ),
    ):
        return str(exc)
    return "Feature computation failed"


async def compute_features_for_symbols(
    session: AsyncSession,
    *,
    symbols: list[str],
    features: list[str] | None,
    timeframe: str,
    start: datetime | None,
    end: datetime | None,
) -> tuple[dict[str, int], dict[str, str]]:
    _validate_feature_names(features)

    resolved: list[tuple[str, int]] = []
    for ticker in symbols:
        row = await get_symbol(session, ticker)
        resolved.append((ticker, row.id))

    computed: dict[str, int] = {}
    failed: dict[str, str] = {}

    for ticker, symbol_id in resolved:
        try:
            async with session.begin_nested():
                count = await compute_and_store_features(
                    session,
                    symbol_id,
                    timeframe,
                    feature_names=features,
                    start=start,
                    end=end,
                )
            computed[ticker] = count
        except Exception as exc:
            message = _failure_message(exc)
            if message == "Feature computation failed":
                logger.exception("feature_compute_symbol_failed", symbol=ticker)
            failed[ticker] = message

    return computed, failed
