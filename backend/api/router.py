"""FastAPI route definitions."""

from collections.abc import Sequence

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db import get_db_session
from backend.services.symbols import DuplicateSymbolError, add_symbol, list_symbols
from backend.storage.models import Symbol
from backend.storage.schemas import SymbolCreate, SymbolRead

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/db")
async def health_db(session: AsyncSession = Depends(get_db_session)) -> dict[str, str]:
    await session.execute(text("SELECT 1"))
    return {"status": "ok"}


@router.post("/symbols", response_model=SymbolRead, status_code=status.HTTP_201_CREATED)
async def create_symbol(
    payload: SymbolCreate,
    session: AsyncSession = Depends(get_db_session),
) -> Symbol:
    try:
        return await add_symbol(session, payload)
    except DuplicateSymbolError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc


@router.get("/symbols", response_model=list[SymbolRead])
async def get_symbols(
    active: bool = Query(default=False),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_db_session),
) -> Sequence[Symbol]:
    return await list_symbols(session, active_only=active, limit=limit, offset=offset)
