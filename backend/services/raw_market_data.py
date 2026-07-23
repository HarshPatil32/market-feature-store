"""Raw market data persistence service."""

import json
import uuid
from typing import Any

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import get_settings
from backend.storage.models import RawMarketData
from backend.storage.object_store import ObjectStoreError, get_object_store
from backend.storage.repository import RawMarketDataRepository

logger = structlog.get_logger(__name__)


async def persist_raw_fetch(
    session: AsyncSession,
    *,
    run_id: int,
    response_payload: dict[str, Any],
    symbol_id: int | None = None,
    source: str | None = None,
    request_params: dict[str, Any] | None = None,
) -> RawMarketData:
    settings = get_settings()
    store = get_object_store()
    size = len(json.dumps(response_payload).encode())
    inline_payload: dict[str, Any] | None = response_payload
    object_key: str | None = None

    if store is not None and size > settings.raw_payload_s3_threshold_bytes:
        key = f"raw/{run_id}/{uuid.uuid4()}.json"
        try:
            await store.put_json(key, response_payload)
            object_key = key
            inline_payload = None
        except ObjectStoreError:
            logger.warning(
                "raw_payload_s3_upload_failed",
                run_id=run_id,
                payload_size_bytes=size,
            )

    repo = RawMarketDataRepository(session)
    return await repo.create(
        run_id=run_id,
        symbol_id=symbol_id,
        source=source,
        request_params=request_params,
        response_payload=inline_payload,
        payload_object_key=object_key,
        payload_size_bytes=size,
    )


async def load_response_payload(raw_row: RawMarketData) -> dict[str, Any]:
    if raw_row.response_payload is not None:
        return raw_row.response_payload
    store = get_object_store()
    if store is None or raw_row.payload_object_key is None:
        raise RuntimeError(
            "raw row has no inline payload and no object store configured"
        )
    return await store.get_json(raw_row.payload_object_key)
