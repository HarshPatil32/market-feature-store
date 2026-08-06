"""Date-range chunking for provider-safe backfill windows."""

from __future__ import annotations

from datetime import timedelta

from pydantic import AwareDatetime

_DEFAULT_WINDOW = timedelta(days=365)

_TIMEFRAME_WINDOWS: dict[str, timedelta] = {
    "1m": timedelta(days=7),
    "5m": timedelta(days=30),
    "15m": timedelta(days=30),
    "30m": timedelta(days=60),
    "1h": timedelta(days=90),
    "1d": timedelta(days=365),
    "1wk": timedelta(days=730),
    "1mo": timedelta(days=3650),
}


def chunk_date_range(
    start: AwareDatetime,
    end: AwareDatetime,
    timeframe: str,
) -> list[tuple[AwareDatetime, AwareDatetime]]:
    """Split [start, end] into inclusive, non-overlapping provider-safe windows."""
    window = _TIMEFRAME_WINDOWS.get(timeframe, _DEFAULT_WINDOW)
    chunks: list[tuple[AwareDatetime, AwareDatetime]] = []
    chunk_start = start

    while chunk_start <= end:
        chunk_end = min(chunk_start + window, end)
        chunks.append((chunk_start, chunk_end))
        if chunk_end >= end:
            break
        chunk_start = chunk_end + timedelta(microseconds=1)

    return chunks
