"""FastAPI entrypoint — Apex Commercial Bank Knowledge Assistant."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import router
from backend.app.config import get_settings
from backend.app.observability.logging import get_logger, setup_logging
from backend.app.observability.tracing import configure_langsmith
from backend.app.retrieval.hybrid import retriever

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    configure_langsmith()
    settings = get_settings()
    logger.info("startup", bank=settings.bank_name)
    try:
        await retriever.initialize()
    except Exception as exc:  # noqa: BLE001
        logger.error("retriever_init_failed", error=str(exc))
    yield
    logger.info("shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=f"{settings.bank_name} — Knowledge Assistant API",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router, prefix="/api")
    return app


app = create_app()
