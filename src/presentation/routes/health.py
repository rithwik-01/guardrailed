from typing import Dict

from fastapi import APIRouter
from prometheus_client import Counter, Gauge, Histogram

router = APIRouter(tags=["Health"])

REQUEST_COUNT = Counter(
    "request_count", "Count of requests received", ["method", "endpoint", "status_code"]
)
REQUEST_LATENCY = Histogram(
    "request_latency_seconds", "Request latency in seconds", ["method", "endpoint"]
)
IN_PROGRESS = Gauge(
    "inprogress_requests", "Number of requests in progress", ["method", "endpoint"]
)

DB_POOL_SIZE = Gauge("db_pool_size", "Number of connections in the MongoDB pool")

MODEL_INFERENCE_TIME = Histogram(
    "model_inference_time_seconds", "Model inference time in seconds", ["model_type"]
)


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Basic health check endpoint that confirms the service is running.
    """
    return {"status": "ok"}


@router.get("/metrics")
async def get_metrics() -> Dict[str, str]:
    """
    Get application metrics.
    """
    return {
        "status": "available via /metrics endpoint in Prometheus format",
    }
