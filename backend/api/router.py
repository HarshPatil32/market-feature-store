"""FastAPI route definitions."""

from collections.abc import Sequence

from fastapi import APIRouter, Depends, Query
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db import get_db_session
from backend.services.symbols import list_symbols
from backend.storage.models import Symbol
from backend.storage.schemas import SymbolRead

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/db")
async def health_db(session: AsyncSession = Depends(get_db_session)) -> dict[str, str]:
    await session.execute(text("SELECT 1"))
    return {"status": "ok"}


@router.get("/symbols", response_model=list[SymbolRead])
async def get_symbols(
    active: bool = Query(default=False),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_db_session),
) -> Sequence[Symbol]:
    return await list_symbols(session, active_only=active, limit=limit, offset=offset)
