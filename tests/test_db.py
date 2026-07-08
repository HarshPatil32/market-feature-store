"""Tests for the ephemeral Postgres test harness."""

from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from backend.main import app


async def test_db_reachable(db_session: AsyncSession) -> None:
    result = await db_session.execute(text("SELECT 1"))
    assert result.scalar() == 1


async def test_health_db_endpoint(engine: AsyncEngine) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/health/db")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
