from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from backend.api.router import router
from backend.config import get_settings
from backend.db import close_db, get_sessionmaker
from backend.features.seed import seed_feature_definitions
from backend.jobs.scheduler import build_scheduler
from backend.logging import configure_logging

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    configure_logging()
    settings = get_settings()
    if settings.feature_definitions_path is not None:
        async with get_sessionmaker()() as session:
            async with session.begin():
                inserted = await seed_feature_definitions(
                    session,
                    settings.feature_definitions_path,
                )
            logger.info("feature_definitions_seeded", inserted=inserted)
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
