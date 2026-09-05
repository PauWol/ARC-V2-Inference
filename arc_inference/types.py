from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant", "tool"]
ReasoningEffort = Literal["auto", "none", "on"]
ToolChoiceMode = Literal["auto", "none", "required"]


class ToolCallFunction(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction


class ToolChoiceFunction(BaseModel):
    name: str


class NamedToolChoice(BaseModel):
    type: Literal["function"] = "function"
    function: ToolChoiceFunction


ToolChoice = ToolChoiceMode | NamedToolChoice


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
    def assistant(
        cls,
        content: str | None = None,
        *,
        reasoning_content: str | None = None,
        tool_calls: list[ToolCall] | None = None,
        **kwargs: Any,
    ) -> "ChatMessage":
        return cls(
            role="assistant",
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            **kwargs,
        )

    @classmethod
    def tool(
        cls,
        content: str,
        tool_call_id: str,
        name: str | None = None,
    ) -> "ChatMessage":
        return cls(
            role="tool",
            content=content,
            tool_call_id=tool_call_id,
            name=name,
        )


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str | None = None
    object: str | None = None
    model: str | None = None
    choices: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, Any] | None = None

    @property
    def message(self) -> ChatMessage | None:
        if not self.choices:
            return None
        message = self.choices[0].get("message")
        return ChatMessage.model_validate(message) if message else None

    @property
    def text(self) -> str:
        message = self.message
        return message.content or "" if message else ""

    @property
    def reasoning(self) -> str:
        message = self.message
        return message.reasoning_content or "" if message else ""

    @property
    def finish_reason(self) -> str | None:
        if not self.choices:
            return None
        return self.choices[0].get("finish_reason")

    @property
    def warning(self) -> str | None:
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

    # ARC runtime capabilities
    tool_protocol: str | None = None
    supports_tools: bool | None = None
    supports_reasoning: bool | None = None
    reasoning_modes: list[ReasoningEffort] = Field(default_factory=list)


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

    @property
    def is_text(self) -> bool:
        return self.type == "text" and bool(self.text)

    @property
    def is_reasoning(self) -> bool:
        return self.type == "reasoning" and bool(self.reasoning)

    @property
    def is_tool_call(self) -> bool:
        return self.type == "tool_call" and bool(self.tool_calls)

    @property
    def is_done(self) -> bool:
        return self.type == "done"

    @property
    def has_finish_reason(self) -> bool:
        return self.finish_reason is not None


@dataclass(slots=True)
class ToolResult:
    tool_call_id: str
    name: str
    content: str
    success: bool = True
    error: str | None = None
