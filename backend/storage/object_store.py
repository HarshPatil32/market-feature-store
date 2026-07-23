"""S3-compatible object store for large raw payloads."""

import asyncio
import json
from functools import lru_cache
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from backend.config import get_settings


class ObjectStoreError(Exception):
    """Raised when object store operations fail."""


class ObjectStore:
    def __init__(
        self,
        *,
        endpoint_url: str | None,
        access_key_id: str,
        secret_access_key: str,
        region: str,
        bucket: str,
    ) -> None:
        self._bucket = bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name=region,
        )

    async def put_json(self, key: str, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        try:
            await asyncio.to_thread(
                self._client.put_object,
                Bucket=self._bucket,
                Key=key,
                Body=body,
                ContentType="application/json",
            )
        except (BotoCoreError, ClientError) as exc:
            raise ObjectStoreError(str(exc)) from exc

    async def get_json(self, key: str) -> dict[str, Any]:
        try:
            response = await asyncio.to_thread(
                self._client.get_object,
                Bucket=self._bucket,
                Key=key,
            )
            data = await asyncio.to_thread(response["Body"].read)
            parsed = json.loads(data)
            if not isinstance(parsed, dict):
                raise ObjectStoreError("stored payload is not a JSON object")
            return parsed
        except json.JSONDecodeError as exc:
            raise ObjectStoreError("stored payload is not valid JSON") from exc
        except (BotoCoreError, ClientError) as exc:
            raise ObjectStoreError(str(exc)) from exc


@lru_cache
def get_object_store() -> ObjectStore | None:
    settings = get_settings()
    if not settings.s3_enabled:
        return None
    assert settings.s3_bucket is not None
    assert settings.s3_access_key_id is not None
    assert settings.s3_secret_access_key is not None
    return ObjectStore(
        endpoint_url=settings.s3_endpoint_url,
        access_key_id=settings.s3_access_key_id.get_secret_value(),
        secret_access_key=settings.s3_secret_access_key.get_secret_value(),
        region=settings.s3_region,
        bucket=settings.s3_bucket,
    )


def clear_object_store_cache() -> None:
    get_object_store.cache_clear()
