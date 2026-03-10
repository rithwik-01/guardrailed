import time

from fastapi import FastAPI, Request, Response, status
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response as StarletteResponse

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total count of requests",
    ["method", "endpoint", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "Request latency in seconds",
    ["method", "endpoint"],
)
IN_PROGRESS = Gauge(
    "http_requests_in_progress", "Requests in progress", ["method", "endpoint"]
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        method = request.method
        path = request.url.path

        if path == "/metrics":
            return await call_next(request)

        IN_PROGRESS.labels(method=method, endpoint=path).inc()

        start_time = time.time()

        try:
            response = await call_next(request)
            status_code = response.status_code
            REQUEST_COUNT.labels(
                method=method, endpoint=path, status_code=status_code
            ).inc()
        except Exception as e:
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            REQUEST_COUNT.labels(
                method=method, endpoint=path, status_code=status_code
            ).inc()
            raise e
        finally:
            duration = time.time() - start_time
            REQUEST_LATENCY.labels(method=method, endpoint=path).observe(duration)

            IN_PROGRESS.labels(method=method, endpoint=path).dec()

        return response


def metrics_endpoint(request: Request) -> StarletteResponse:
    """Endpoint that returns Prometheus metrics."""
    return StarletteResponse(
        content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST
    )


def setup_metrics(app: FastAPI) -> None:
    """Set up the metrics middleware and endpoint."""
    app.add_middleware(PrometheusMiddleware)
    app.add_route("/metrics", metrics_endpoint)
