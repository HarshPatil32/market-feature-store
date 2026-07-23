"""pytest configuration and shared fixtures."""

import os
from collections.abc import AsyncGenerator, Generator
from pathlib import Path

import pytest
import pytest_asyncio
from alembic.command import upgrade
from alembic.config import Config
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from testcontainers.postgres import PostgresContainer

from backend.config import get_settings
from backend.db import clear_db_cache, get_engine
from backend.providers.fake import FakeProvider
from backend.storage.object_store import clear_object_store_cache

PG_IMAGE = "postgres:16-alpine"


@pytest.fixture(scope="session")
def pg_container() -> Generator[PostgresContainer, None, None]:
    with PostgresContainer(PG_IMAGE, driver="asyncpg") as pg:
        yield pg


@pytest.fixture(scope="session")
def db_url(pg_container: PostgresContainer) -> str:
    return str(pg_container.get_connection_url())


S3_DISABLE_KEYS = (
    "S3_ENDPOINT_URL",
    "S3_BUCKET",
    "S3_ACCESS_KEY_ID",
    "S3_SECRET_ACCESS_KEY",
)


@pytest.fixture(scope="session", autouse=True)
def override_settings(db_url: str) -> Generator[None, None, None]:
    original_db = os.environ.get("DATABASE_URL")
    original_key = os.environ.get("PROVIDER_API_KEY")
    original_secret = os.environ.get("PROVIDER_API_SECRET")
    original_s3 = {key: os.environ.get(key) for key in S3_DISABLE_KEYS}
    os.environ["DATABASE_URL"] = db_url
    os.environ["PROVIDER_API_KEY"] = "test-key"
    os.environ["PROVIDER_API_SECRET"] = "test-secret"
    for key in S3_DISABLE_KEYS:
        os.environ[key] = ""
    get_settings.cache_clear()
    clear_object_store_cache()
    clear_db_cache()
    yield
    if original_db is None:
        os.environ.pop("DATABASE_URL", None)
    else:
        os.environ["DATABASE_URL"] = original_db
    if original_key is None:
        os.environ.pop("PROVIDER_API_KEY", None)
    else:
        os.environ["PROVIDER_API_KEY"] = original_key
    if original_secret is None:
        os.environ.pop("PROVIDER_API_SECRET", None)
    else:
        os.environ["PROVIDER_API_SECRET"] = original_secret
    for key, value in original_s3.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    get_settings.cache_clear()
    clear_object_store_cache()
    clear_db_cache()


@pytest.fixture(scope="session")
def _migrated(db_url: str) -> None:
    root = Path(__file__).resolve().parents[1]
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", db_url)
    upgrade(config, "head")


@pytest_asyncio.fixture(scope="session")
async def engine(_migrated: None) -> AsyncGenerator[AsyncEngine, None]:
    eng = get_engine()
    yield eng
    await eng.dispose()


# Default fake provider for tests that do not need a fixed clock.
# test_provider_fake.py overrides this fixture with a fixed `now`.
@pytest.fixture
def fake_provider() -> FakeProvider:
    return FakeProvider()


@pytest_asyncio.fixture
async def db_session(engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    async with engine.connect() as connection:
        transaction = await connection.begin()
        session = AsyncSession(connection, expire_on_commit=False)
        yield session
        await session.close()
        await transaction.rollback()
