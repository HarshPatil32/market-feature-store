"""Feature definition registry business logic."""

from collections.abc import Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from backend.storage.models import FeatureDefinition
from backend.storage.repository import FeatureDefinitionRepository


class FeatureNotFoundError(Exception):
    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"Feature not found: {name}")


async def list_features(
    session: AsyncSession,
    *,
    active_only: bool = False,
    limit: int | None = None,
    offset: int = 0,
) -> Sequence[FeatureDefinition]:
    repo = FeatureDefinitionRepository(session)
    return await repo.list(
        active_only=active_only,
        limit=limit,
        offset=offset,
    )


async def get_feature_by_name(
    session: AsyncSession,
    name: str,
) -> Sequence[FeatureDefinition]:
    repo = FeatureDefinitionRepository(session)
    rows = await repo.get_by_name(name)
    if not rows:
        raise FeatureNotFoundError(name)
    return rows


async def get_active_feature_version(
    session: AsyncSession,
    name: str,
) -> FeatureDefinition:
    repo = FeatureDefinitionRepository(session)
    row = await repo.get_active_version(name)
    if row is None:
        raise FeatureNotFoundError(name)
    return row
