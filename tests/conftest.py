"""pytest configuration and shared fixtures."""

import os
from collections.abc import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    create_async_engine,
)
from testcontainers.postgres import PostgresContainer

from backend.config import get_settings

PG_IMAGE = "postgres:16-alpine"


@pytest.fixture(scope="session")
def pg_container() -> Generator[PostgresContainer, None, None]:
    with PostgresContainer(PG_IMAGE, driver="asyncpg") as pg:
        yield pg


@pytest.fixture(scope="session")
def db_url(pg_container: PostgresContainer) -> str:
    return str(pg_container.get_connection_url())


@pytest.fixture(scope="session", autouse=True)
def override_settings(db_url: str) -> Generator[None, None, None]:
    original_db = os.environ.get("DATABASE_URL")
    original_key = os.environ.get("PROVIDER_API_KEY")
    os.environ["DATABASE_URL"] = db_url
    os.environ["PROVIDER_API_KEY"] = "test-key"
    get_settings.cache_clear()
    yield
    if original_db is None:
        os.environ.pop("DATABASE_URL", None)
    else:
        os.environ["DATABASE_URL"] = original_db
    if original_key is None:
        os.environ.pop("PROVIDER_API_KEY", None)
    else:
        os.environ["PROVIDER_API_KEY"] = original_key
    get_settings.cache_clear()


@pytest_asyncio.fixture(scope="session")
async def engine(db_url: str) -> AsyncGenerator[AsyncEngine, None]:
    eng = create_async_engine(db_url, echo=False)
    # TODO: run `alembic upgrade head` once migrations exist.
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def db_session(engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    async with engine.connect() as connection:
        transaction = await connection.begin()
        session = AsyncSession(connection, expire_on_commit=False)
        yield session
        await session.close()
        await transaction.rollback()
