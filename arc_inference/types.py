from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


Role = Literal["system", "user", "assistant", "tool"]


class ToolCallFunction(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Role
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = None

    @classmethod
    def system(cls, content: str) -> "ChatMessage":
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> "ChatMessage":
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str | None = None, **kwargs: Any) -> "ChatMessage":
        return cls(role="assistant", content=content, **kwargs)

    @classmethod
    def tool(
        cls, content: str, tool_call_id: str, name: str | None = None
    ) -> "ChatMessage":
        return cls(role="tool", content=content, tool_call_id=tool_call_id, name=name)


class ChatResponse(BaseModel):
    """Flexible response model; unknown server fields are preserved."""

    model_config = ConfigDict(extra="allow")

    id: str | None = None
    object: str | None = None
    model: str | None = None
    choices: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] | None = None

    @property
    def message(self) -> ChatMessage | None:
        if not self.choices:
            return None
        message = self.choices[0].get("message")
        if not message:
            return None
        return ChatMessage.model_validate(message)

    @property
    def text(self) -> str:
        if not self.choices:
            return ""
        message = self.choices[0].get("message") or {}
        content = message.get("content")
        return content or ""

    @property
    def reasoning(self) -> str:
        if not self.choices:
            return ""
        message = self.choices[0].get("message") or {}
        return message.get("reasoning_content") or ""

    @property
    def finish_reason(self) -> str | None:
        if not self.choices:
            return None
        return self.choices[0].get("finish_reason")

    @property
    def warning(self) -> str | None:  # NEW
        if not self.choices:
            return None
        return self.choices[0].get("warning")

    @property
    def tool_calls(self) -> list[ToolCall]:
        message = self.message
        return message.tool_calls or [] if message else []


class CompletionResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str | None = None
    object: str | None = None
    model: str | None = None
    choices: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, Any] | None = None

    @property
    def text(self) -> str:
        if not self.choices:
            return ""
        return self.choices[0].get("text") or ""


class EmbeddingResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    data: list[dict[str, Any]] = Field(default_factory=list)
    model: str | None = None
    usage: dict[str, Any] | None = None


class ModelInfo(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str | None = None
    object: str | None = None
    owned_by: str | None = None


@dataclass(slots=True)
class StreamEvent:
    type: str
    data: dict[str, Any] = field(default_factory=dict)
    text: str = ""
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    warning: str | None = None
    raw: Any = None


@dataclass(slots=True)
class ToolResult:
    tool_call_id: str
    name: str
    content: str
    success: bool = True
    error: str | None = None
