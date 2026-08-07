from typing import Dict

from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Basic health check endpoint that confirms the service is running.
    """
    return {"status": "ok"}
