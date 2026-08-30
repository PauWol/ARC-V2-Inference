from __future__ import annotations


class InferenceError(Exception):
    """Base exception for the inference SDK."""


class APIError(InferenceError):
    def __init__(self, status_code: int, message: str, detail=None):
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code
        self.message = message
        self.detail = detail


class ToolExecutionError(InferenceError):
    def __init__(self, tool_name: str, message: str):
        super().__init__(f"Tool {tool_name!r} failed: {message}")
        self.tool_name = tool_name
        self.message = message


class ToolCallIncompleteError(InferenceError):
    """Raised when the model starts a tool call but the server truncates it."""

    def __init__(self, *, finish_reason: str | None, max_output_tokens: int, tool_name: str | None = None):
        name = f" for tool {tool_name!r}" if tool_name else ""
        reason = finish_reason or "unknown"
        super().__init__(
            "The model started a tool call"
            f"{name} but the response ended with finish_reason={reason!r}. "
            f"Increase max_output_tokens (current: {max_output_tokens}) and verify the "
            "inference server's tool-call tag configuration matches the model."
        )
        self.finish_reason = finish_reason
        self.max_output_tokens = max_output_tokens
        self.tool_name = tool_name
