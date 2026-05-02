"""
Gemini-Claude Bridge Entry Point

Minimal entry point for the lean Gemini-exclusive pipeline.
Run with: uv run python server.py
"""

from api.app import app

if __name__ == "__main__":
    import uvicorn
    from config.settings import get_settings

    settings = get_settings()
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level="info",
        timeout_graceful_shutdown=5,
    )
