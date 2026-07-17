"""Tests for structured logging configuration."""

import json
from collections.abc import Generator

import pytest
import structlog

from backend.config import get_settings
from backend.logging import (
    bind_ingestion_run_id,
    clear_ingestion_run_id,
    configure_logging,
)

VALID_DB_URL = (
    "postgresql+asyncpg://postgres:changeme@localhost:5433/market_feature_store"
)


@pytest.fixture(autouse=True)
def reset_logging_state() -> Generator[None, None, None]:
    get_settings.cache_clear()
    structlog.reset_defaults()
    structlog.contextvars.clear_contextvars()
    yield
    get_settings.cache_clear()
    structlog.reset_defaults()
    structlog.contextvars.clear_contextvars()


def _configure(
    monkeypatch: pytest.MonkeyPatch,
    *,
    environment: str = "production",
    log_level: str = "INFO",
) -> None:
    monkeypatch.setenv("DATABASE_URL", VALID_DB_URL)
    monkeypatch.setenv("PROVIDER_API_KEY", "test-key")
    monkeypatch.setenv("PROVIDER_API_SECRET", "test-secret")
    monkeypatch.setenv("ENVIRONMENT", environment)
    monkeypatch.setenv("LOG_LEVEL", log_level)
    configure_logging()


def test_json_log_shape(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _configure(monkeypatch, environment="production")

    log = structlog.get_logger()
    log.info("test.event")

    output = capsys.readouterr().out.strip()
    payload = json.loads(output)

    assert payload["event"] == "test.event"
    assert payload["level"] == "info"
    assert "timestamp" in payload
    assert "ingestion_run_id" not in payload


def test_ingestion_run_id_propagates(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _configure(monkeypatch, environment="production")
    bind_ingestion_run_id("run-123")

    log = structlog.get_logger()
    log.info("ingestion.started")

    output = capsys.readouterr().out.strip()
    payload = json.loads(output)

    assert payload["ingestion_run_id"] == "run-123"


def test_ingestion_run_id_cleared(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _configure(monkeypatch, environment="production")
    bind_ingestion_run_id("run-123")
    clear_ingestion_run_id()

    log = structlog.get_logger()
    log.info("ingestion.finished")

    output = capsys.readouterr().out.strip()
    payload = json.loads(output)

    assert "ingestion_run_id" not in payload


def test_log_level_filtering(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _configure(monkeypatch, environment="production", log_level="WARNING")

    log = structlog.get_logger()
    log.debug("debug.event")

    assert capsys.readouterr().out == ""


def test_staging_renderer_emits_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _configure(monkeypatch, environment="staging")

    log = structlog.get_logger()
    log.info("staging.event")

    output = capsys.readouterr().out.strip()
    payload = json.loads(output)

    assert payload["event"] == "staging.event"
    assert payload["level"] == "info"


def test_dev_renderer_is_not_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _configure(monkeypatch, environment="development")

    log = structlog.get_logger()
    log.info("dev.event")

    output = capsys.readouterr().out.strip()

    with pytest.raises(json.JSONDecodeError):
        json.loads(output)
