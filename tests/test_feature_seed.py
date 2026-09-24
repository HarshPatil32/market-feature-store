"""Tests for feature definition seed loader."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.features.seed import FeatureDefinitionSeedEntry, seed_feature_definitions
from backend.storage.repository import FeatureDefinitionRepository

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_DEFINITIONS_PATH = REPO_ROOT / "config" / "feature_definitions.json"


@pytest.mark.asyncio
async def test_seed_inserts_new_definitions(
    db_session: AsyncSession,
    tmp_path: Path,
) -> None:
    path = tmp_path / "features.json"
    path.write_text(
        json.dumps(
            [
                {"name": "sma_20", "version": 1, "lookback_window": 20},
                {"name": "rsi_14", "version": 1, "lookback_window": 14},
            ]
        ),
        encoding="utf-8",
    )

    inserted = await seed_feature_definitions(db_session, path)

    assert inserted == 2
    repo = FeatureDefinitionRepository(db_session)
    rows = await repo.list()
    assert len(rows) == 2
    assert {(r.name, r.version, r.lookback_window) for r in rows} == {
        ("sma_20", 1, 20),
        ("rsi_14", 1, 14),
    }


@pytest.mark.asyncio
async def test_seed_is_idempotent(
    db_session: AsyncSession,
    tmp_path: Path,
) -> None:
    path = tmp_path / "features.json"
    path.write_text(
        json.dumps([{"name": "sma_20", "version": 1, "lookback_window": 20}]),
        encoding="utf-8",
    )

    first = await seed_feature_definitions(db_session, path)
    second = await seed_feature_definitions(db_session, path)

    assert first == 1
    assert second == 0
    repo = FeatureDefinitionRepository(db_session)
    assert len(await repo.list()) == 1


@pytest.mark.asyncio
async def test_seed_skips_existing_row_without_mutation(
    db_session: AsyncSession,
    tmp_path: Path,
) -> None:
    repo = FeatureDefinitionRepository(db_session)
    await repo.create(name="sma_20", version=1, lookback_window=99)

    path = tmp_path / "features.json"
    path.write_text(
        json.dumps([{"name": "sma_20", "version": 1, "lookback_window": 20}]),
        encoding="utf-8",
    )

    inserted = await seed_feature_definitions(db_session, path)

    assert inserted == 0
    row = await repo.get_by_name_and_version("sma_20", 1)
    assert row is not None
    assert row.lookback_window == 99


@pytest.mark.asyncio
async def test_seed_rejects_invalid_json_entry(
    db_session: AsyncSession,
    tmp_path: Path,
) -> None:
    path = tmp_path / "features.json"
    path.write_text(
        json.dumps([{"name": "sma_20", "version": 1}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid feature definition at index 0"):
        await seed_feature_definitions(db_session, path)


@pytest.mark.asyncio
async def test_seed_rejects_extra_fields(
    db_session: AsyncSession,
    tmp_path: Path,
) -> None:
    path = tmp_path / "features.json"
    path.write_text(
        json.dumps(
            [
                {
                    "name": "sma_20",
                    "version": 1,
                    "lookback_window": 20,
                    "formula": "evil",
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid feature definition at index 0"):
        await seed_feature_definitions(db_session, path)


@pytest.mark.asyncio
async def test_seed_empty_list(
    db_session: AsyncSession,
    tmp_path: Path,
) -> None:
    path = tmp_path / "features.json"
    path.write_text("[]", encoding="utf-8")

    inserted = await seed_feature_definitions(db_session, path)

    assert inserted == 0
    repo = FeatureDefinitionRepository(db_session)
    assert len(await repo.list()) == 0


@pytest.mark.asyncio
async def test_seed_missing_file_raises(
    db_session: AsyncSession,
    tmp_path: Path,
) -> None:
    path = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError, match="Feature definitions file not found"):
        await seed_feature_definitions(db_session, path)


def test_seed_entry_rejects_extra_fields_directly() -> None:
    with pytest.raises(ValidationError):
        FeatureDefinitionSeedEntry.model_validate(
            {"name": "x", "version": 1, "lookback_window": 1, "extra": True}
        )


@pytest.mark.asyncio
async def test_seed_rejects_non_array_json(
    db_session: AsyncSession,
    tmp_path: Path,
) -> None:
    path = tmp_path / "features.json"
    path.write_text(json.dumps({"name": "sma_20"}), encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a JSON array"):
        await seed_feature_definitions(db_session, path)


@pytest.mark.asyncio
async def test_seed_duplicate_entries_in_file_inserts_once(
    db_session: AsyncSession,
    tmp_path: Path,
) -> None:
    path = tmp_path / "features.json"
    entry = {"name": "sma_20", "version": 1, "lookback_window": 20}
    path.write_text(json.dumps([entry, entry]), encoding="utf-8")

    inserted = await seed_feature_definitions(db_session, path)

    assert inserted == 1
    repo = FeatureDefinitionRepository(db_session)
    assert len(await repo.list()) == 1


def test_default_feature_definitions_config_is_valid() -> None:
    assert DEFAULT_FEATURE_DEFINITIONS_PATH.is_file()
    raw = json.loads(DEFAULT_FEATURE_DEFINITIONS_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw, list)
    assert len(raw) >= 1
    for item in raw:
        FeatureDefinitionSeedEntry.model_validate(item)
