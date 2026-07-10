"""SQLAlchemy async engine and session management."""

from collections.abc import AsyncGenerator
from functools import lru_cache

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from backend.config import get_settings


@lru_cache
def get_engine() -> AsyncEngine:
    return create_async_engine(str(get_settings().database_url), echo=False)


@lru_cache
def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        get_engine(),
        expire_on_commit=False,
        class_=AsyncSession,
    )


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with get_sessionmaker()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def clear_db_cache() -> None:
    get_engine.cache_clear()
    get_sessionmaker.cache_clear()


async def close_db() -> None:
    await get_engine().dispose()
    clear_db_cache()
