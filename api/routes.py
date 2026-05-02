"""Minimalist Gemini route handlers."""

import uuid
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from loguru import logger

from config.settings import Settings
from .dependencies import get_provider, get_settings, require_api_key
from .models.anthropic import MessagesRequest
from .request_utils import get_token_count

router = APIRouter()

def _probe_response(allow: str) -> Response:
    """Return an empty success response for compatibility probes."""
    return Response(status_code=204, headers={"Allow": allow})

@router.post("/v1/messages")
async def create_message(
    request_data: MessagesRequest,
    settings: Settings = Depends(get_settings),
    provider = Depends(get_provider),
    _auth = Depends(require_api_key),
):
    """Bridge Claude Code requests to Gemini."""
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    
    logger.info("GEMINI_BRIDGE: model={} messages={}", request_data.model, len(request_data.messages))

    input_tokens = get_token_count(request_data.messages, request_data.system, request_data.tools)
    
    return StreamingResponse(
        provider.stream_response(
            request_data,
            input_tokens=input_tokens,
            request_id=request_id,
        ),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )

@router.api_route("/v1/messages", methods=["HEAD", "OPTIONS"])
async def probe_messages():
    return _probe_response("POST, HEAD, OPTIONS")

@router.get("/health")
async def health():
    return {"status": "healthy", "engine": "gemini-only"}

@router.get("/")
async def root():
    return {"status": "ok", "message": "Claude Code to Gemini Bridge"}

@router.api_route("/", methods=["HEAD", "OPTIONS"])
async def probe_root():
    return _probe_response("GET, HEAD, OPTIONS")
