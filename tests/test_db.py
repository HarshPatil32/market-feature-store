"""Tests for the ephemeral Postgres test harness."""

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


async def test_db_reachable(db_session: AsyncSession) -> None:
    result = await db_session.execute(text("SELECT 1"))
    assert result.scalar() == 1
