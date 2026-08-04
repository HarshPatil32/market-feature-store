"""Data quality check persistence service."""

from collections.abc import Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from backend.storage.models import DataQualityCheck
from backend.storage.repository import DataQualityCheckRepository
from backend.validation.validator import QualityCheckResult


async def persist_check_results(
    session: AsyncSession,
    results: Sequence[QualityCheckResult],
    *,
    symbol_id: int,
    run_id: int | None = None,
) -> Sequence[DataQualityCheck]:
    """Bulk-persist validation results for a single symbol and optional run."""
    if not results:
        return []

    rows = [
        result.to_check_row(symbol_id=symbol_id, run_id=run_id) for result in results
    ]
    return await DataQualityCheckRepository(session).bulk_create(rows)
