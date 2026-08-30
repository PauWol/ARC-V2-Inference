from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from typing import Any, Callable, get_type_hints

from .errors import ToolExecutionError

try:
    from pydantic import TypeAdapter
except ImportError:  # pragma: no cover
    TypeAdapter = None


@dataclass(slots=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]

    @classmethod
    def from_function(cls, fn: Callable[..., Any], *, name: str | None = None, description: str | None = None) -> "Tool":
        schema = _function_schema(fn)
        return cls(
            name=name or fn.__name__,
            description=description if description is not None else (inspect.getdoc(fn) or ""),
            parameters=schema,
            handler=fn,
        )

    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description or None,
                "parameters": self.parameters,
            },
        }

    def call(self, arguments: str | dict[str, Any]) -> Any:
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
        except json.JSONDecodeError as exc:
            raise ToolExecutionError(self.name, f"invalid JSON arguments: {exc}") from exc
        if not isinstance(args, dict):
            raise ToolExecutionError(self.name, "arguments must decode to a JSON object")
        try:
            return self.handler(**args)
        except Exception as exc:
            raise ToolExecutionError(self.name, str(exc)) from exc


def tool(fn: Callable[..., Any] | None = None, *, name: str | None = None, description: str | None = None):
    """Turn a Python function into an inference tool."""
    def decorator(func: Callable[..., Any]) -> Tool:
        return Tool.from_function(func, name=name, description=description)

    return decorator(fn) if fn is not None else decorator


def ensure_tool(value: Tool | Callable[..., Any]) -> Tool:
    return value if isinstance(value, Tool) else Tool.from_function(value)


def _function_schema(fn: Callable[..., Any]) -> dict[str, Any]:
    # Pydantic gives us robust JSON schema for Python annotations.
    if TypeAdapter is None:
        raise RuntimeError("pydantic is required for automatic tool schema generation")

    signature = inspect.signature(fn)
    hints = get_type_hints(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for parameter in signature.parameters.values():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            raise TypeError(f"Tool {fn.__name__!r} cannot use *args or **kwargs")
        annotation = hints.get(parameter.name, Any)
        schema = TypeAdapter(annotation).json_schema()
        properties[parameter.name] = schema
        if parameter.default is inspect.Parameter.empty:
            required.append(parameter.name)

    result: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        result["required"] = required
    return result
