from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Iterable, Iterator, Sequence
from typing import Any, TypeVar

from pydantic import BaseModel, Field

from .agent import Agent, AsyncAgent
from .conversation import AsyncConversation, Conversation
from .transport import AsyncTransport, SyncTransport
from .types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    EmbeddingResponse,
    ModelInfo,
    StreamEvent,
    ToolCall,
    ToolCallFunction,
)

T = TypeVar("T", bound=BaseModel)


class Models:
    def __init__(self, client: "InferenceClient"):
        self._client = client

    def list(self) -> list[ModelInfo]:
        data = self._client._request("GET", "/models")
        raw = data.get("data", data if isinstance(data, list) else [])
        return [ModelInfo.model_validate(x) for x in raw]

    def current(self) -> ModelInfo:
        return ModelInfo.model_validate(self._client._request("GET", "/model"))


class AsyncModels:
    def __init__(self, client: "AsyncInferenceClient"):
        self._client = client

    async def list(self) -> list[ModelInfo]:
        data = await self._client._request("GET", "/models")
        raw = data.get("data", data if isinstance(data, list) else [])
        return [ModelInfo.model_validate(x) for x in raw]

    async def current(self) -> ModelInfo:
        return ModelInfo.model_validate(await self._client._request("GET", "/model"))


class Chat:
    def __init__(self, client: "InferenceClient"):
        self._client = client

    def create(
        self,
        messages: Sequence[ChatMessage | dict[str, Any]] | Sequence[str],
        *,
        model: str | None = None,
        stream: bool = False,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop: list[str] | None = None,
        response_format: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        **extra: Any,
    ) -> ChatResponse | Iterator[StreamEvent]:
        payload = _chat_payload(
            messages,
            stream=stream,
            model=model,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            response_format=response_format,
            metadata=metadata,
            tools=tools,
            tool_choice=tool_choice,
            **extra,
        )
        if stream:
            return self._client._chat_stream(payload)
        return ChatResponse.model_validate(
            self._client._request("POST", "/chat/completions", json=payload)
        )

    def stream(self, *args: Any, **kwargs: Any) -> Iterator[StreamEvent]:
        kwargs["stream"] = True
        return self.create(*args, **kwargs)


class AsyncChat:
    def __init__(self, client: "AsyncInferenceClient"):
        self._client = client

    async def create(
        self, messages: Sequence[ChatMessage | dict[str, Any]], **kwargs: Any
    ) -> ChatResponse | AsyncIterator[StreamEvent]:
        stream = kwargs.get("stream", False)
        payload = _chat_payload(messages, **kwargs)
        if stream:
            return self._client._chat_stream(payload)
        return ChatResponse.model_validate(
            await self._client._request("POST", "/chat/completions", json=payload)
        )

    async def stream(
        self, messages: Sequence[ChatMessage | dict[str, Any]], **kwargs: Any
    ) -> AsyncIterator[StreamEvent]:
        kwargs["stream"] = True
        return self.create(messages, **kwargs)  # type: ignore[return-value]


class Completions:
    def __init__(self, client: "InferenceClient"):
        self._client = client

    def create(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponse | Iterator[StreamEvent]:
        stream = kwargs.pop("stream", False)
        payload = {"prompt": prompt, "stream": stream, **_clean(kwargs)}
        if stream:
            return self._client._chat_stream(payload, endpoint="/completions")
        return CompletionResponse.model_validate(
            self._client._request("POST", "/completions", json=payload)
        )


class AsyncCompletions:
    def __init__(self, client: "AsyncInferenceClient"):
        self._client = client

    async def create(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponse | AsyncIterator[StreamEvent]:
        stream = kwargs.pop("stream", False)
        payload = {"prompt": prompt, "stream": stream, **_clean(kwargs)}
        if stream:
            return self._client._chat_stream(payload, endpoint="/completions")
        return CompletionResponse.model_validate(
            await self._client._request("POST", "/completions", json=payload)
        )


class Embeddings:
    def __init__(self, client: "InferenceClient"):
        self._client = client

    def create(self, input: str | list[str], **kwargs: Any) -> EmbeddingResponse:
        payload = {"input": input, **_clean(kwargs)}
        return EmbeddingResponse.model_validate(
            self._client._request("POST", "/embeddings", json=payload)
        )


class AsyncEmbeddings:
    def __init__(self, client: "AsyncInferenceClient"):
        self._client = client

    async def create(self, input: str | list[str], **kwargs: Any) -> EmbeddingResponse:
        payload = {"input": input, **_clean(kwargs)}
        return EmbeddingResponse.model_validate(
            await self._client._request("POST", "/embeddings", json=payload)
        )


class InferenceClient:
    def __init__(
        self,
        base_url: str = "http://localhost:7842/v1",
        api_key: str | None = None,
        timeout: float = 120.0,
    ):
        self._transport = SyncTransport(
            base_url, api_key or os.getenv("INFERENCE_API_KEY"), timeout
        )
        self.chat = Chat(self)
        self.completions = Completions(self)
        self.embeddings = Embeddings(self)
        self.models = Models(self)

    def close(self) -> None:
        self._transport.close()

    def __enter__(self) -> "InferenceClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        return self._transport.request(method, path, **kwargs)

    def _chat_stream(
        self, payload: dict[str, Any], endpoint: str = "/chat/completions"
    ) -> Iterator[StreamEvent]:
        tool_ids: dict[int, str] = {}
        for raw in self._transport.stream_sse("POST", endpoint, json=payload):
            yield _event_from_chunk(raw, tool_ids)

    def conversation(
        self,
        *,
        system: str | None = None,
        messages: Iterable[ChatMessage | dict[str, Any]] | None = None,
        model: str | None = None,
        **defaults: Any,
    ) -> Conversation:
        return Conversation(
            self, system=system, messages=messages, model=model, defaults=defaults
        )

    def agent(
        self,
        *,
        instructions: str | None = None,
        tools: Iterable[Any] = (),
        model: str | None = None,
        **kwargs: Any,
    ) -> Agent:
        return Agent(
            self, instructions=instructions, tools=tools, model=model, **kwargs
        )

    def structured(
        self,
        messages: Sequence[ChatMessage | dict[str, Any]],
        output_model: type[T],
        **kwargs: Any,
    ) -> T:
        response_format = kwargs.pop("response_format", {"type": "json_object"})
        response = self.chat.create(messages, response_format=response_format, **kwargs)
        assert isinstance(response, ChatResponse)
        return output_model.model_validate(json.loads(response.text))


class AsyncInferenceClient:
    def __init__(
        self,
        base_url: str = "http://localhost:7842/v1",
        api_key: str | None = None,
        timeout: float = 120.0,
    ):
        self._transport = AsyncTransport(
            base_url, api_key or os.getenv("INFERENCE_API_KEY"), timeout
        )
        self.chat = AsyncChat(self)
        self.completions = AsyncCompletions(self)
        self.embeddings = AsyncEmbeddings(self)
        self.models = AsyncModels(self)

    async def close(self) -> None:
        await self._transport.close()

    async def __aenter__(self) -> "AsyncInferenceClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        return await self._transport.request(method, path, **kwargs)

    async def _chat_stream(
        self, payload: dict[str, Any], endpoint: str = "/chat/completions"
    ) -> AsyncIterator[StreamEvent]:
        tool_ids: dict[int, str] = {}
        async for raw in self._transport.stream_sse("POST", endpoint, json=payload):
            yield _event_from_chunk(raw, tool_ids)

    def conversation(
        self,
        *,
        system: str | None = None,
        messages: Iterable[ChatMessage | dict[str, Any]] | None = None,
        model: str | None = None,
        **defaults: Any,
    ) -> AsyncConversation:
        return AsyncConversation(
            self, system=system, messages=messages, model=model, defaults=defaults
        )

    def agent(
        self,
        *,
        instructions: str | None = None,
        tools: Iterable[Any] = (),
        model: str | None = None,
        **kwargs: Any,
    ) -> AsyncAgent:
        return AsyncAgent(
            self, instructions=instructions, tools=tools, model=model, **kwargs
        )

    async def structured(
        self,
        messages: Sequence[ChatMessage | dict[str, Any]],
        output_model: type[T],
        **kwargs: Any,
    ) -> T:
        response_format = kwargs.pop("response_format", {"type": "json_object"})
        response = await self.chat.create(
            messages, response_format=response_format, **kwargs
        )
        assert isinstance(response, ChatResponse)
        return output_model.model_validate(json.loads(response.text))


def _chat_payload(
    messages: Sequence[Any], stream: bool = False, **kwargs: Any
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "messages": [_message_dict(m) for m in messages],
        "stream": stream,
    }
    ignored = {"self", "messages", "model"}
    payload.update(
        {
            k: v
            for k, v in kwargs.items()
            if k not in ignored and k != "extra" and v is not None
        }
    )
    extra = kwargs.get("extra") or {}
    payload.update(_clean(extra))
    return payload


def _message_dict(message: Any) -> dict[str, Any]:
    if isinstance(message, str):
        return ChatMessage.user(message).model_dump(exclude_none=True)
    if isinstance(message, ChatMessage):
        return message.model_dump(exclude_none=True)
    return dict(message)


def _clean(value: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in value.items() if v is not None}


def _event_from_chunk(
    raw: dict[str, Any],
    tool_ids: dict[int, str] | None = None,
) -> StreamEvent:
    """Convert an SSE payload into a normalized SDK stream event.

    Supports OpenAI-style ``choices[].delta.tool_calls`` as well as servers
    that place completed tool calls in ``message.tool_calls`` or at the
    top-level ``tool_calls`` field. Tool-call IDs are preserved across
    continuation chunks even when later chunks omit the ID.
    """
    if raw.get("__done__"):
        return StreamEvent(type="done", raw=raw)
    if "__raw__" in raw:
        return StreamEvent(type="raw", raw=raw["__raw__"])

    tool_ids = tool_ids if tool_ids is not None else {}
    choices = raw.get("choices") or []
    choice = choices[0] if choices else {}
    delta = choice.get("delta") or {}
    message = choice.get("message") or {}

    text = delta.get("content") or message.get("content") or raw.get("content") or ""
    reasoning = (
        delta.get("reasoning_content")
        or message.get("reasoning_content")
        or raw.get("reasoning_content")
        or ""
    )

    raw_tool_calls = delta.get("tool_calls")
    if raw_tool_calls is None:
        raw_tool_calls = message.get("tool_calls")
    if raw_tool_calls is None:
        raw_tool_calls = raw.get("tool_calls")
    raw_tool_calls = raw_tool_calls or []

    tool_calls: list[ToolCall] = []
    for position, item in enumerate(raw_tool_calls):
        if not isinstance(item, dict):
            continue

        index = item.get("index", position)
        try:
            index = int(index)
        except (TypeError, ValueError):
            index = position

        fn = item.get("function") or {}
        call_id = item.get("id") or tool_ids.get(index) or f"stream-tool-{index}"
        if item.get("id"):
            tool_ids[index] = call_id

        tool_calls.append(
            ToolCall(
                id=call_id,
                function=ToolCallFunction(
                    name=fn.get("name") or "",
                    arguments=fn.get("arguments") or "",
                ),
            )
        )

    finish_reason = choice.get("finish_reason")
    warning = choice.get("warning")
    if tool_calls:
        event_type = "tool_call"
    elif reasoning:
        event_type = "reasoning"
    elif text:
        event_type = "text"
    elif finish_reason:
        event_type = "completed"
    else:
        event_type = "chunk"

    return StreamEvent(
        type=event_type,
        data=raw,
        text=text,
        reasoning=reasoning,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        warning=warning,
        raw=raw,
    )
