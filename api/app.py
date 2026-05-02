"""Minimalist FastAPI application for Gemini-Claude bridge."""

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger

from config.logging_config import configure_logging
from config.settings import get_settings
from providers.exceptions import ProviderError

from .dependencies import cleanup_provider
from .routes import router

# Configure logging
settings = get_settings()
configure_logging(settings.log_file)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Gemini-Claude Bridge...")
    yield
    logger.info("Cleaning up...")
    await cleanup_provider()
    logger.info("Shutdown complete")

def create_app() -> FastAPI:
    app = FastAPI(
        title="Gemini-Claude Bridge",
        version="2.0.0",
        lifespan=lifespan,
    )
    app.include_router(router)

    @app.exception_handler(ProviderError)
    async def provider_error_handler(request: Request, exc: ProviderError):
        return JSONResponse(status_code=exc.status_code, content=exc.to_anthropic_format())

    @app.exception_handler(Exception)
    async def general_error_handler(request: Request, exc: Exception):
        logger.error(f"Unexpected Error: {exc}")
        return JSONResponse(status_code=500, content={"type": "error", "error": {"type": "api_error", "message": str(exc)}})

    return app

app = create_app()
