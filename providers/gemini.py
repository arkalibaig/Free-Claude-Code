from typing import Any, AsyncIterator
from providers.openai_compat import OpenAICompatibleProvider
from providers.common.message_converter import build_base_request_body

# Gemini's OpenAI-compatible endpoint
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"

class GeminiProvider(OpenAICompatibleProvider):
    """
    Lean, high-performance provider optimized specifically for Gemini.
    """
    def __init__(self, config):
        super().__init__(
            config,
            provider_name="GEMINI",
            base_url=GEMINI_BASE_URL,
            api_key=config.api_key,
        )

    def _build_request_body(self, request: Any) -> dict:
        # Gemini-specific optimizations
        body = build_base_request_body(
            request,
            include_thinking=True,
            include_reasoning_content=True
        )
        
        # Ensure model is correctly passed without our provider prefix
        if "/" in body["model"]:
            body["model"] = body["model"].split("/")[-1]
            
        return body

    def _is_thinking_enabled(self, request: Any) -> bool:
        # Force thinking for experimental Gemini thinking models
        model = getattr(request, "model", "").lower()
        return "thinking" in model or "flash-thinking" in model or super()._is_thinking_enabled(request)
