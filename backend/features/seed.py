"""Load feature definitions from JSON config into the database."""

import json
from pathlib import Path

import structlog
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.storage.repository import FeatureDefinitionRepository

logger = structlog.get_logger(__name__)


class FeatureDefinitionSeedEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=100)
    version: int = Field(ge=1)
    lookback_window: int = Field(ge=1)
    description: str | None = Field(default=None, max_length=255)
    requirements: list[str] | None = None


def _load_seed_entries(path: Path) -> list[FeatureDefinitionSeedEntry]:
    if not path.is_file():
        raise FileNotFoundError(f"Feature definitions file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Feature definitions file must contain a JSON array")
    entries: list[FeatureDefinitionSeedEntry] = []
    for index, item in enumerate(raw):
        try:
            entries.append(FeatureDefinitionSeedEntry.model_validate(item))
        except ValidationError as exc:
            raise ValueError(
                f"Invalid feature definition at index {index}: {exc}"
            ) from exc
    return entries


async def seed_feature_definitions(session: AsyncSession, path: Path) -> int:
    """Insert definitions from path that are not already present (by name + version)."""
    entries = _load_seed_entries(path)
    repo = FeatureDefinitionRepository(session)
    inserted = 0
    for entry in entries:
        existing = await repo.get_by_name_and_version(entry.name, entry.version)
        if existing is not None:
            logger.debug(
                "feature_definition_skipped",
                name=entry.name,
                version=entry.version,
                reason="already_exists",
            )
            continue
        try:
            async with session.begin_nested():
                await repo.create(
                    name=entry.name,
                    version=entry.version,
                    lookback_window=entry.lookback_window,
                    description=entry.description,
                    requirements=entry.requirements,
                )
        except IntegrityError:
            logger.debug(
                "feature_definition_skipped",
                name=entry.name,
                version=entry.version,
                reason="concurrent_insert",
            )
            continue
        inserted += 1
    return inserted
