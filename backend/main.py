from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.api.router import router
from backend.db import close_db
from backend.logging import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    configure_logging()
    yield
    await close_db()


app = FastAPI(title="market-feature-store", lifespan=lifespan)
app.include_router(router)
