"""FastAPI application factory: CORS, rate limiting, routers, health probes,
static image serving (local mode), user seeding, and EMF metrics.

Observability: after each /search* request the middleware emits ONE
CloudWatch-EMF-formatted JSON line to stdout. CloudWatch Logs automatically
extracts LatencyMs / Requests metrics (namespace "FashionRetrieval",
dimension Route) from those lines — that is what powers the p50/p95 latency
dashboard without any metrics SDK or sidecar.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Awaitable, Callable

import anyio.to_thread
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from api import admin_routes, auth_routes, search_routes
from api.deps import get_settings
from api.engine_holder import EngineHolder
from api.rate_limit import limiter
from api.repos import (
    UserAlreadyExistsError,
    UserRepo,
    build_parse_cache,
    build_token_repo,
    build_user_repo,
)
from api.security import hash_password
from api.settings import Settings

log = logging.getLogger(__name__)

APP_VERSION = "0.1.0"

# Bare-message logger on stdout for EMF lines (never mixed with app logging
# formatting — CloudWatch needs the raw JSON document as the full line).
_emf_logger = logging.getLogger("fashion_retrieval.emf")
if not _emf_logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _emf_logger.addHandler(_handler)
    _emf_logger.setLevel(logging.INFO)
    _emf_logger.propagate = False


def _emit_emf(route: str, latency_ms: float, status_code: int) -> None:
    """One EMF document per request — CloudWatch auto-extracts the metrics."""
    doc: dict[str, Any] = {
        "_aws": {
            "Timestamp": int(time.time() * 1000),
            "CloudWatchMetrics": [{
                "Namespace": "FashionRetrieval",
                "Dimensions": [["Route"]],
                "Metrics": [
                    {"Name": "LatencyMs", "Unit": "Milliseconds"},
                    {"Name": "Requests", "Unit": "Count"},
                ],
            }],
        },
        "Route": route,
        "LatencyMs": round(latency_ms, 3),
        "Requests": 1,
        "Status": status_code,
    }
    _emf_logger.info(json.dumps(doc, separators=(",", ":")))


async def _emf_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    start = time.perf_counter()
    status_code = 500  # if call_next raises, record the failure as a 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        # CORS preflights (OPTIONS) are excluded so Requests counts real work.
        if request.url.path.startswith("/search") and request.method != "OPTIONS":
            _emit_emf(request.url.path,
                      (time.perf_counter() - start) * 1000.0, status_code)


def _seed_users(settings: Settings, users: UserRepo) -> None:
    """Idempotently create the optional demo + admin accounts from env."""
    for email, password, role in (
        (settings.DEMO_USER_EMAIL, settings.DEMO_USER_PASSWORD, "user"),
        (settings.ADMIN_EMAIL, settings.ADMIN_PASSWORD, "admin"),
    ):
        if not email or not password:
            continue
        email = email.strip().lower()
        try:
            if users.get_by_email(email) is None:
                users.create(email, hash_password(password), role=role)
                log.info("seeded %s account %s", role, email)
        except UserAlreadyExistsError:
            pass  # concurrent replica won the race — fine
        except Exception:  # noqa: BLE001 — seeding must never block startup
            log.exception("failed to seed %s account", role)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    # SSM fetch (if configured) is blocking IO — keep it off the event loop.
    await anyio.to_thread.run_sync(settings.resolve_jwt_secret)

    app.state.settings = settings
    app.state.user_repo = build_user_repo(settings)
    app.state.token_repo = build_token_repo(settings)

    await anyio.to_thread.run_sync(_seed_users, settings, app.state.user_repo)

    holder = EngineHolder(settings, parse_cache=build_parse_cache(settings))
    app.state.engine_holder = holder
    holder.start()  # background thread — searches 503 until it flips ready

    log.info("%s v%s started (engine building in background)",
             settings.APP_NAME, APP_VERSION)
    yield
    log.info("%s shutting down", settings.APP_NAME)


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.APP_NAME, version=APP_VERSION,
                  lifespan=lifespan)

    # rate limiting (slowapi)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # CORS — credentialed (refresh cookie), so origins must be explicit.
    origins = settings.cors_origins_list or [
        "http://localhost:3000", "http://localhost:5173",
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.middleware("http")(_emf_middleware)

    app.include_router(auth_routes.router)
    app.include_router(search_routes.router)
    app.include_router(admin_routes.router)

    @app.get("/healthz")
    async def healthz(request: Request) -> dict[str, Any]:
        """Liveness: always 200; reports engine readiness + version."""
        holder = getattr(request.app.state, "engine_holder", None)
        return {
            "status": "ok",
            "ready": bool(holder is not None and holder.ready),
            "version": APP_VERSION,
        }

    @app.get("/readyz")
    async def readyz(request: Request) -> Response:
        """Readiness: 503 until the search engine has been built."""
        holder = getattr(request.app.state, "engine_holder", None)
        if holder is not None and holder.ready:
            return JSONResponse({"ready": True, "version": APP_VERSION})
        body: dict[str, Any] = {"ready": False}
        if holder is not None and holder.error:
            body["error"] = "engine initialization failed"  # details in logs only
        return JSONResponse(body, status_code=503)

    # Local mode: serve dataset images statically at /local-images/{filename}.
    if not settings.IMAGE_BASE_URL:
        images_dir = settings.images_path
        images_dir.mkdir(parents=True, exist_ok=True)
        app.mount("/local-images", StaticFiles(directory=str(images_dir)),
                  name="local-images")

    return app


app = create_app()
