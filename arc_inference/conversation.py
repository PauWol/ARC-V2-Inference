from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Iterator
from typing import Any

from .types import ChatMessage, ChatResponse, StreamEvent


class Conversation:
    def __init__(self, client: Any, *, system: str | None = None, messages: Iterable[Any] | None = None, model: str | None = None, defaults: dict[str, Any] | None = None):
        self.client = client
        self.model = model
        self.defaults = defaults or {}
        self.messages: list[ChatMessage] = []
        if system:
            self.messages.append(ChatMessage.system(system))
        if messages:
            self.messages.extend(_coerce_messages(messages))

    def add(self, message: ChatMessage | dict[str, Any] | str) -> "Conversation":
        self.messages.append(_coerce_message(message))
        return self

    def reset(self, *, keep_system: bool = True) -> "Conversation":
        system = self.messages[:1] if keep_system and self.messages and self.messages[0].role == "system" else []
        self.messages = list(system)
        return self

    def clone(self) -> "Conversation":
        return Conversation(self.client, messages=[m.model_copy(deep=True) for m in self.messages], model=self.model, defaults=dict(self.defaults))

    def send(self, content: str, **kwargs: Any) -> ChatResponse:
        self.messages.append(ChatMessage.user(content))
        params = {**self.defaults, **kwargs}
        response = self.client.chat.create(self.messages, model=self.model, **params)
        assert isinstance(response, ChatResponse)
        if response.message:
            self.messages.append(response.message)
        return response

    def stream(self, content: str, **kwargs: Any) -> Iterator[StreamEvent]:
        self.messages.append(ChatMessage.user(content))
        params = {**self.defaults, **kwargs, "stream": True}
        events = self.client.chat.create(self.messages, model=self.model, **params)
        assert not isinstance(events, ChatResponse)
        text = ""
        reasoning = ""
        for event in events:
            text += event.text
            reasoning += event.reasoning
            yield event
        self.messages.append(ChatMessage.assistant(text or None, reasoning_content=reasoning or None))

    def export(self) -> list[dict[str, Any]]:
        return [m.model_dump(exclude_none=True) for m in self.messages]


class AsyncConversation:
    def __init__(self, client: Any, *, system: str | None = None, messages: Iterable[Any] | None = None, model: str | None = None, defaults: dict[str, Any] | None = None):
        self.client = client
        self.model = model
        self.defaults = defaults or {}
        self.messages: list[ChatMessage] = []
        if system:
            self.messages.append(ChatMessage.system(system))
        if messages:
            self.messages.extend(_coerce_messages(messages))

    def add(self, message: ChatMessage | dict[str, Any] | str) -> "AsyncConversation":
        self.messages.append(_coerce_message(message))
        return self

    def reset(self, *, keep_system: bool = True) -> "AsyncConversation":
        system = self.messages[:1] if keep_system and self.messages and self.messages[0].role == "system" else []
        self.messages = list(system)
        return self

    async def send(self, content: str, **kwargs: Any) -> ChatResponse:
        self.messages.append(ChatMessage.user(content))
        params = {**self.defaults, **kwargs}
        response = await self.client.chat.create(self.messages, model=self.model, **params)
        assert isinstance(response, ChatResponse)
        if response.message:
            self.messages.append(response.message)
        return response

    async def stream(self, content: str, **kwargs: Any) -> AsyncIterator[StreamEvent]:
        self.messages.append(ChatMessage.user(content))
        params = {**self.defaults, **kwargs, "stream": True}
        events = await self.client.chat.create(self.messages, model=self.model, **params)
        assert not isinstance(events, ChatResponse)
        text = ""
        reasoning = ""
        async for event in events:
            text += event.text
            reasoning += event.reasoning
            yield event
        self.messages.append(ChatMessage.assistant(text or None, reasoning_content=reasoning or None))

    def export(self) -> list[dict[str, Any]]:
        return [m.model_dump(exclude_none=True) for m in self.messages]


def _coerce_message(value: Any) -> ChatMessage:
    if isinstance(value, ChatMessage):
        return value
    if isinstance(value, str):
        return ChatMessage.user(value)
    return ChatMessage.model_validate(value)


def _coerce_messages(values: Iterable[Any]) -> list[ChatMessage]:
    return [_coerce_message(v) for v in values]
