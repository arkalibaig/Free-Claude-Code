"""Minimalist dependency injection for Gemini provider."""

from fastapi import Depends, HTTPException, Request
from loguru import logger

from config.settings import Settings
from config.settings import get_settings as _get_settings
from providers.base import BaseProvider, ProviderConfig
from providers.gemini import GeminiProvider
from providers.exceptions import AuthenticationError

_provider: BaseProvider | None = None

def get_settings() -> Settings:
    return _get_settings()

def get_provider() -> BaseProvider:
    """Get or create the exclusive Gemini provider."""
    global _provider
    if _provider is None:
        settings = get_settings()
        if not settings.gemini_api_key.strip():
            raise HTTPException(
                status_code=503, 
                detail="GEMINI_API_KEY is not set in .env"
            )
        
        config = ProviderConfig(
            api_key=settings.gemini_api_key,
            base_url="", # Handled by GeminiProvider
            rate_limit=15, # Gemini beta limits
            rate_window=60,
            max_concurrency=2,
            http_read_timeout=settings.http_read_timeout,
            http_write_timeout=settings.http_write_timeout,
            http_connect_timeout=settings.http_connect_timeout,
            enable_thinking=settings.enable_thinking,
        )
        _provider = GeminiProvider(config)
        logger.info("Gemini provider initialized")
    return _provider

def require_api_key(request: Request, settings: Settings = Depends(get_settings)) -> None:
    token = settings.anthropic_auth_token
    if not token:
        return

    header = request.headers.get("x-api-key") or request.headers.get("authorization")
    if not header:
        raise HTTPException(status_code=401, detail="Missing API key")

    sent_token = header.replace("Bearer ", "")
    if sent_token != token:
        raise HTTPException(status_code=401, detail="Invalid API key")

async def cleanup_provider():
    global _provider
    if _provider:
        await _provider.cleanup()
        _provider = None
