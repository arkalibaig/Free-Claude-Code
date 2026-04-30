import json
import uuid
from abc import abstractmethod
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
from loguru import logger
from openai import AsyncOpenAI

from providers.base import BaseProvider, ProviderConfig
from providers.common import (
    ContentType,
    HeuristicToolParser,
    SSEBuilder,
    ThinkTagParser,
    append_request_id,
    get_user_facing_error_message,
    map_error,
    map_stop_reason,
)
from providers.rate_limit import GlobalRateLimiter


class OpenAICompatibleProvider(BaseProvider):
    """Base class for providers using OpenAI-compatible chat completions API."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        provider_name: str,
        base_url: str,
        api_key: str,
    ):
        super().__init__(config)
        self._provider_name = provider_name
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._global_rate_limiter = GlobalRateLimiter.get_instance(
            rate_limit=config.rate_limit,
            rate_window=config.rate_window,
            max_concurrency=config.max_concurrency,
        )

        http_client = None
        if config.proxy:
            http_client = httpx.AsyncClient(
                proxy=config.proxy,
                timeout=httpx.Timeout(120.0, connect=15.0, read=120.0, write=120.0),
            )

        self._client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            max_retries=0,
            timeout=httpx.Timeout(120.0, connect=15.0, read=120.0, write=120.0),
            http_client=http_client,
        )

    async def cleanup(self) -> None:
        client = getattr(self, "_client", None)
        if client is not None:
            await client.aclose()

    @abstractmethod
    def _build_request_body(self, request: Any) -> dict:
        """Build request body. Must be implemented by subclasses."""

    def _handle_extra_reasoning(
        self, delta: Any, sse: SSEBuilder, *, thinking_enabled: bool
    ) -> Iterator[str]:
        return iter(())

    def _get_retry_request_body(self, error: Exception, body: dict) -> dict | None:
        return None

    def _is_thinking_enabled(self, request: Any) -> bool:
        if hasattr(request, "thinking") and request.thinking:
            return True
        model = getattr(request, "model", "").lower()
        return any(x in model for x in ["thinking", "o1", "o3"])

    async def _create_stream(self, body: dict) -> tuple[Any, dict]:
        try:
            stream = await self._global_rate_limiter.execute_with_retry(
                self._client.chat.completions.create, **body, stream=True
            )
            return stream, body
        except Exception as error:
            retry_body = self._get_retry_request_body(error, body)
            if retry_body is None:
                raise
            stream = await self._global_rate_limiter.execute_with_retry(
                self._client.chat.completions.create, **retry_body, stream=True
            )
            return stream, retry_body

    def _process_tool_call(self, tc: dict, sse: SSEBuilder) -> Iterator[str]:
        # FIX: Offset index by 2 to prevent collision with Thinking (Index 0) and Text (Index 1)
        raw_index = tc.get("index", 0)
        tc_index = (int(raw_index) if raw_index is not None else 0) + 2

        if tc_index < 0:
            tc_index = len(sse.blocks.tool_states)

        fn_delta = tc.get("function", {})
        incoming_name = fn_delta.get("name")
        if incoming_name:
            sse.blocks.register_tool_name(tc_index, incoming_name)

        state = sse.blocks.tool_states.get(tc_index)

        if state is None or not state.started:
            tool_id = tc.get("id") or f"tool_{uuid.uuid4()}"
            name = (state.name if state else None) or incoming_name or "tool_call"
            yield sse.start_tool_block(tc_index, tool_id, name)
            state = sse.blocks.tool_states.get(tc_index)

        args = fn_delta.get("arguments", "")
        if args:
            current_name = state.name if state else ""
            if current_name == "Task":
                parsed = sse.blocks.buffer_task_args(tc_index, args)
                if parsed is not None:
                    yield sse.emit_tool_delta(tc_index, json.dumps(parsed))
                return
            yield sse.emit_tool_delta(tc_index, args)

    async def _stream_response_impl(
        self,
        request: Any,
        input_tokens: int,
        request_id: str | None,
    ) -> AsyncIterator[str]:
        tag = self._provider_name
        message_id = f"msg_{uuid.uuid4()}"
        sse = SSEBuilder(message_id, request.model, input_tokens)

        body = self._build_request_body(request)
        req_tag = f" request_id={request_id}" if request_id else ""
        
        logger.info(
            "{}_STREAM:{} model={} msgs={} tools={}",
            tag, req_tag, body.get("model"),
            len(body.get("messages", [])),
            len(body.get("tools", []))
        )

        yield sse.message_start()

        think_parser = ThinkTagParser()
        heuristic_parser = HeuristicToolParser()
        thinking_enabled = self._is_thinking_enabled(request)

        finish_reason = None
        usage_info = None
        error_occurred = False

        async with self._global_rate_limiter.concurrency_slot():
            try:
                stream, body = await self._create_stream(body)
                async for chunk in stream:
                    if hasattr(chunk, "usage") and chunk.usage:
                        usage_info = chunk.usage

                    if not chunk.choices:
                        continue

                    choice = chunk.choices[0]
                    delta = getattr(choice, "delta", None)
                    if delta is None:
                        continue

                    if choice.finish_reason:
                        finish_reason = choice.finish_reason
                        logger.debug("{} finish_reason: {}", tag, finish_reason)

                    # --- 1. HANDLE REASONING ---
                    reasoning = getattr(delta, "reasoning_content", None)
                    if thinking_enabled and reasoning:
                        for event in sse.ensure_thinking_block(): yield event
                        yield sse.emit_thinking_delta(reasoning)

                    for event in self._handle_extra_reasoning(delta, sse, thinking_enabled=thinking_enabled):
                        yield event

                    # --- 2. HANDLE TEXT CONTENT ---
                    if getattr(delta, "content", None):
                        for part in think_parser.feed(delta.content):
                            if part.type == ContentType.THINKING:
                                if not thinking_enabled: continue
                                for event in sse.ensure_thinking_block(): yield event
                                yield sse.emit_thinking_delta(part.content)
                            else:
                                filtered_text, detected_tools = heuristic_parser.feed(part.content)
                                if filtered_text:
                                    for event in sse.ensure_text_block(): yield event
                                    yield sse.emit_text_delta(filtered_text)
                                for tool_use in detected_tools:
                                    for event in sse.close_content_blocks(): yield event
                                    block_idx = sse.blocks.allocate_index()
                                    if tool_use.get("name") == "Task" and isinstance(tool_use.get("input"), dict):
                                        tool_use["input"]["run_in_background"] = False
                                    yield sse.content_block_start(block_idx, "tool_use", id=tool_use["id"], name=tool_use["name"])
                                    yield sse.content_block_delta(block_idx, "input_json_delta", json.dumps(tool_use["input"]))
                                    yield sse.content_block_stop(block_idx)

                    # --- 3. HANDLE STRUCTURED TOOL CALLS ---
                    t_calls = getattr(delta, "tool_calls", None)
                    if t_calls:
                        for event in sse.close_content_blocks(): yield event
                        for tc in t_calls:
                            tc_func = getattr(tc, "function", None)
                            tc_info = {
                                "index": getattr(tc, "index", 0),
                                "id": getattr(tc, "id", None),
                                "function": {
                                    "name": getattr(tc_func, "name", None) if tc_func else None,
                                    "arguments": getattr(tc_func, "arguments", "") if tc_func else ""
                                }
                            }
                            for event in self._process_tool_call(tc_info, sse): yield event

                    # --- 4. SAFETY NET: Catch tool calls in final choice message ---
                    if not t_calls and hasattr(choice, "message") and getattr(choice.message, "tool_calls", None):
                        for event in sse.close_content_blocks(): yield event
                        for tc in choice.message.tool_calls:
                            tc_func = getattr(tc, "function", None)
                            tc_info = {
                                "index": 0,
                                "id": getattr(tc, "id", None),
                                "function": {
                                    "name": getattr(tc_func, "name", None) if tc_func else None,
                                    "arguments": getattr(tc_func, "arguments", "") if tc_func else ""
                                }
                            }
                            for event in self._process_tool_call(tc_info, sse): yield event

            except Exception as e:
                logger.error("{}_ERROR:{} {}: {}", tag, req_tag, type(e).__name__, e)
                mapped_e = map_error(e)
                error_occurred = True
                base_message = get_user_facing_error_message(mapped_e, read_timeout_s=120.0)
                error_message = append_request_id(base_message, request_id)
                for event in sse.close_content_blocks(): yield event
                for event in sse.emit_error(error_message): yield event

        # Finalize Parsers
        remaining = think_parser.flush()
        if remaining:
            if remaining.type == ContentType.THINKING and thinking_enabled:
                for event in sse.ensure_thinking_block(): yield event
                yield sse.emit_thinking_delta(remaining.content)
            elif remaining.type == ContentType.TEXT:
                for event in sse.ensure_text_block(): yield event
                yield sse.emit_text_delta(remaining.content)

        for tool_use in heuristic_parser.flush():
            for event in sse.close_content_blocks(): yield event
            block_idx = sse.blocks.allocate_index()
            yield sse.content_block_start(block_idx, "tool_use", id=tool_use["id"], name=tool_use["name"])
            if tool_use.get("name") == "Task" and isinstance(tool_use.get("input"), dict):
                tool_use["input"]["run_in_background"] = False
            yield sse.content_block_delta(block_idx, "input_json_delta", json.dumps(tool_use["input"]))
            yield sse.content_block_stop(block_idx)

        if not error_occurred and sse.blocks.text_index == -1 and not sse.blocks.tool_states:
            for event in sse.ensure_text_block(): yield event

        # Usage reporting
        output_tokens = int(sse.estimate_output_tokens())
        if usage_info:
            output_tokens = int(getattr(usage_info, "completion_tokens", 0) or 0)
            provider_input = int(getattr(usage_info, "prompt_tokens", 0) or 0)
            logger.debug("TOKEN_ESTIMATE: our={} provider={} diff={:+d}", input_tokens, provider_input, provider_input - input_tokens)

        yield sse.message_delta(map_stop_reason(finish_reason), output_tokens)
        yield sse.message_stop()

    async def stream_response(
        self, request: Any, input_tokens: int = 0, *, request_id: str | None = None
    ) -> AsyncIterator[str]:
        with logger.contextualize(request_id=request_id):
            async for event in self._stream_response_impl(request, input_tokens, request_id):
                yield event
