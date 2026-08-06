"""Tests for date-range chunking."""

from datetime import UTC, datetime, timedelta

import pytest

from backend.ingestion.chunking import chunk_date_range


def test_single_chunk_when_start_equals_end() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = start

    chunks = chunk_date_range(start, end, "1d")

    assert chunks == [(start, end)]


def test_returns_empty_when_start_after_end() -> None:
    start = datetime(2024, 1, 3, tzinfo=UTC)
    end = datetime(2024, 1, 1, tzinfo=UTC)

    chunks = chunk_date_range(start, end, "1d")

    assert chunks == []


def test_single_chunk_when_range_fits_window() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 3, tzinfo=UTC)

    chunks = chunk_date_range(start, end, "1d")

    assert chunks == [(start, end)]


def test_splits_into_multiple_windows() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 15, tzinfo=UTC)

    chunks = chunk_date_range(start, end, "1m")

    assert len(chunks) == 2
    assert chunks[0][0] == start
    assert chunks[0][1] == datetime(2024, 1, 8, tzinfo=UTC)
    assert chunks[1][0] == chunks[0][1] + timedelta(microseconds=1)
    assert chunks[1][1] == end


def test_chunks_are_adjacent_without_overlap() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 2, 1, tzinfo=UTC)

    chunks = chunk_date_range(start, end, "1m")

    assert chunks[0][0] == start
    assert chunks[-1][1] == end
    for index in range(len(chunks) - 1):
        assert chunks[index + 1][0] == chunks[index][1] + timedelta(microseconds=1)
        assert chunks[index][1] < chunks[index + 1][0]


@pytest.mark.parametrize(
    ("timeframe", "expected_chunk_count"),
    [
        ("1m", 2),
        ("1d", 1),
    ],
)
def test_representative_timeframe_tiers(
    timeframe: str,
    expected_chunk_count: int,
) -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 15, tzinfo=UTC)

    chunks = chunk_date_range(start, end, timeframe)

    assert len(chunks) == expected_chunk_count


def test_unmapped_timeframe_uses_default_window() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 6, 1, tzinfo=UTC)

    chunks = chunk_date_range(start, end, "custom")

    assert len(chunks) == 1
    assert chunks == [(start, end)]
