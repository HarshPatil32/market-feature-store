"""Background job scheduling and workers."""

from datetime import timedelta

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from backend.config import get_settings
from backend.db import get_sessionmaker
from backend.ingestion.incremental import incremental_symbols
from backend.providers.factory import get_market_data_provider
from backend.services.symbols import detect_stale_symbols

logger = structlog.get_logger(__name__)


async def _run_incremental_job(
    *,
    timeframe: str,
    lookback_days: int,
    stale_threshold_hours: int,
) -> None:
    try:
        settings = get_settings()
        async with get_sessionmaker()() as session:
            stale = await detect_stale_symbols(
                session,
                threshold=timedelta(hours=stale_threshold_hours),
            )
            symbols = [row.symbol for row in stale]
            if not symbols:
                logger.info("scheduled_incremental_job_no_stale_symbols")
                return
            provider = get_market_data_provider(settings)
            await incremental_symbols(
                session,
                symbols=symbols,
                timeframe=timeframe,
                provider=provider,
                source=settings.provider_name,
                lookback_days=lookback_days,
            )
    except Exception:
        logger.exception("scheduled_incremental_job_failed")


def build_scheduler() -> AsyncIOScheduler | None:
    """Return a configured scheduler, or None if incremental scheduling is disabled."""
    settings = get_settings()
    if not settings.incremental_schedule:
        return None

    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        _run_incremental_job,
        CronTrigger.from_crontab(settings.incremental_schedule),
        kwargs={
            "timeframe": "1d",
            "lookback_days": settings.incremental_lookback_days,
            "stale_threshold_hours": settings.incremental_stale_threshold_hours,
        },
        id="incremental_ingestion",
        replace_existing=True,
        misfire_grace_time=300,
    )
    return scheduler
