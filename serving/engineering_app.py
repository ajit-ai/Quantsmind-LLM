"""FastAPI application factory for the TAKSH engineering service."""

from __future__ import annotations

from fastapi import FastAPI

from serving.engineering_api import build_router


def create_app() -> FastAPI:
    app = FastAPI(title="TAKSH Engineering Service", version="1.0.0")
    router = build_router()
    if router is not None:
        app.include_router(router)
    return app


app = create_app()
