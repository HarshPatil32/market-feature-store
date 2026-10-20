"""In-process registry of feature implementations keyed by (name, version)."""

from __future__ import annotations

from typing import TypeVar

from backend.features.base import Feature

_F = TypeVar("_F", bound=type[Feature])


class FeatureNotRegisteredError(LookupError):
    """Raised when no feature is registered for the requested (name, version)."""


class FeatureRegistry:
    def __init__(self) -> None:
        self._features: dict[tuple[str, int], Feature] = {}

    def register(self, feature_cls: _F) -> _F:
        instance = feature_cls()
        key = (instance.spec.name, instance.spec.version)
        if key in self._features:
            raise ValueError(
                f"feature already registered: name={key[0]!r} version={key[1]}"
            )
        self._features[key] = instance
        return feature_cls

    def get(self, name: str, version: int) -> Feature:
        key = (name, version)
        try:
            return self._features[key]
        except KeyError as exc:
            raise FeatureNotRegisteredError(
                f"feature not registered: name={name!r} version={version}"
            ) from exc

    def all(self) -> list[Feature]:
        return [self._features[key] for key in sorted(self._features)]


registry = FeatureRegistry()
register_feature = registry.register
