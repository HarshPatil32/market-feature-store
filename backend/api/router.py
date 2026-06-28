"""FastAPI route definitions."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")  # type: ignore[untyped-decorator]
def health() -> dict[str, str]:
    return {"status": "ok"}
