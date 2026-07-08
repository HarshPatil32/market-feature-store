"""Smoke tests for Alembic migrations."""

from pathlib import Path

from alembic.command import downgrade, upgrade
from alembic.config import Config


def _alembic_config() -> Config:
    root = Path(__file__).resolve().parents[1]
    return Config(str(root / "alembic.ini"))


def test_alembic_migrations_apply_cleanly(db_url: str) -> None:
    config = _alembic_config()
    config.set_main_option("sqlalchemy.url", db_url)
    upgrade(config, "head")
    downgrade(config, "base")
    upgrade(config, "head")
