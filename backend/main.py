from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from backend.api.router import router
from backend.db import close_db
from backend.jobs.scheduler import build_scheduler
from backend.logging import configure_logging

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    configure_logging()
    scheduler = build_scheduler()
    if scheduler is not None:
        scheduler.start()
    yield
    if scheduler is not None:
        logger.warning(
            "scheduler_shutting_down",
            message="in-flight scheduled jobs may be interrupted",
        )
        scheduler.shutdown(wait=False)
    await close_db()


app = FastAPI(title="market-feature-store", lifespan=lifespan)
app.include_router(router)
