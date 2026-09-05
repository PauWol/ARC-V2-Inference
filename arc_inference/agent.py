from __future__ import annotations

import inspect
import json
import time
from collections.abc import AsyncIterator, Iterable, Iterator
from typing import Any, TypeVar

from pydantic import BaseModel

from .errors import ToolCallIncompleteError, ToolExecutionError
from .tools import Tool, ensure_tool
from .tracing import AgentRunResult, AgentTrace, TraceCallback, TraceEvent
from .types import (
    ChatMessage,
    ChatResponse,
    StreamEvent,
    ToolCall,
    ToolCallFunction,
    ToolResult,
)

T = TypeVar("T", bound=BaseModel)


class Agent:
    def __init__(
        self,
        client: Any,
        *,
        instructions: str | None = None,
        tools: Iterable[Any] = (),
        model: str | None = None,
        max_iterations: int = 10,
        tool_choice: str = "auto",
        max_output_tokens: int | None = 1024,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        metadata: dict[str, Any] | None = None,
        tool_error_mode: str = "message",
        retry_incomplete_tool_calls: int = 1,
        trace_callback: TraceCallback | None = None,
    ):
        self.client = client
        self.model = model
        self.instructions = instructions
        self.max_iterations = max_iterations
        self.tool_choice = tool_choice
        self.defaults = {
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "metadata": metadata,
        }
        self.tool_error_mode = tool_error_mode
        self.retry_incomplete_tool_calls = max(0, retry_incomplete_tool_calls)
        self.trace_callback = trace_callback
        self.last_trace: AgentTrace | None = None
        self._recent_tool_calls: list[tuple[str, str]] = []
        self._tool_result_cache: dict[tuple[str, str], ToolResult] = {}
        self.max_same_tool_calls = 1
        self.tools: dict[str, Tool] = {}
        self.messages: list[ChatMessage] = []
        if instructions:
            self.messages.append(ChatMessage.system(instructions))
        self.add_tools(tools)

    def add_tools(self, tools: Iterable[Any]) -> "Agent":
        for value in tools:
            item = ensure_tool(value)
            self.tools[item.name] = item
        return self

    def remove_tool(self, name: str) -> "Agent":
        self.tools.pop(name, None)
        return self

    def reset(self) -> "Agent":
        self.messages = (
            [ChatMessage.system(self.instructions)] if self.instructions else []
        )
        self.last_trace = None
        self._recent_tool_calls.clear()
        return self

    def run(
        self,
        prompt: str,
        *,
        response_model: type[T] | None = None,
        trace: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Run the agent. Use trace=True to return AgentRunResult with full execution trace."""
        result = self._run(prompt, response_model=response_model, **kwargs)
        return result if trace else result.response

    def run_with_trace(
        self, prompt: str, *, response_model: type[T] | None = None, **kwargs: Any
    ) -> AgentRunResult:
        return self._run(prompt, response_model=response_model, **kwargs)

    def _emit_trace(
        self, trace: AgentTrace, event_type: str, **kwargs: Any
    ) -> TraceEvent:
        event = trace.add(event_type, **kwargs)
        if self.trace_callback:
            self.trace_callback(event)
        return event

    def _run(
        self, prompt: str, *, response_model: type[T] | None = None, **kwargs: Any
    ) -> AgentRunResult:
        trace = AgentTrace()
        self.last_trace = trace
        self._recent_tool_calls.clear()
        self._tool_result_cache.clear()
        self.messages.append(ChatMessage.user(prompt))
        self._emit_trace(trace, "agent_started", text=prompt)
        retries_left = self.retry_incomplete_tool_calls
        current_max_tokens = (
            kwargs.get("max_output_tokens", self.defaults.get("max_output_tokens"))
            or 1024
        )
        force_no_tools = False

        force_no_tools = False
        try:
            for iteration in range(1, self.max_iterations + 1):
                self._emit_trace(trace, "iteration_started", iteration=iteration)
                request_kwargs = {
                    k: v for k, v in self.defaults.items() if v is not None
                }
                request_kwargs.update(kwargs)
                request_kwargs["max_output_tokens"] = current_max_tokens
                response = self.client.chat.create(
                    self.messages,
                    model=self.model,
                    tools=[]
                    if force_no_tools
                    else [item.definition() for item in self.tools.values()],
                    tool_choice="none"
                    if force_no_tools
                    else (self.tool_choice if self.tools else "none"),
                    **request_kwargs,
                )
                assert isinstance(response, ChatResponse)
                self._emit_trace(
                    trace,
                    "model_response",
                    iteration=iteration,
                    data={
                        "finish_reason": response.finish_reason,
                        "usage": response.usage,
                    },
                    text=response.text,
                )
                message = response.message
                if message is None:
                    self._emit_trace(trace, "agent_completed", iteration=iteration)
                    return AgentRunResult(response, trace.finish())

                if message.tool_calls:
                    invalid = next(
                        (
                            call
                            for call in message.tool_calls
                            if not _valid_json_object(call.function.arguments)
                        ),
                        None,
                    )
                    if invalid is not None and response.finish_reason == "length":
                        if retries_left > 0:
                            retries_left -= 1
                            current_max_tokens *= 2
                            self._emit_trace(
                                trace,
                                "tool_call_retry",
                                iteration=iteration,
                                name=invalid.function.name if invalid else None,
                                call_id=invalid.id if invalid else None,
                                data={
                                    "max_output_tokens": current_max_tokens,
                                    "warning": response.warning,
                                },
                            )
                            continue
                        raise ToolCallIncompleteError(
                            finish_reason=response.finish_reason,
                            max_output_tokens=current_max_tokens,
                            tool_name=invalid.function.name if invalid else None,
                            warning=response.warning,
                        )

                    self.messages.append(message)
                    for call in message.tool_calls:
                        result = self._execute_traced(
                            trace,
                            iteration,
                            call.id,
                            call.function.name,
                            call.function.arguments,
                        )
                        if (
                            _tool_signature(call.function.name, call.function.arguments)
                            in self._recent_tool_calls[:-1]
                        ):
                            force_no_tools = True
                        self.messages.append(
                            ChatMessage.tool(
                                result.content, result.tool_call_id, name=result.name
                            )
                        )
                    continue

                self.messages.append(message)
                output = _parse_output(response, response_model)
                self._emit_trace(
                    trace, "agent_completed", iteration=iteration, text=response.text
                )
                return AgentRunResult(output, trace.finish())

            raise RuntimeError(f"Agent exceeded max_iterations={self.max_iterations}")
        except Exception as exc:
            self._emit_trace(trace, "error", error=str(exc), success=False)
            trace.finish()
            raise

    def stream(
        self, prompt: str, *, trace: bool = True, **kwargs: Any
    ) -> Iterator[StreamEvent]:
        """Stream model output and agent execution events.

        trace=True emits SDK-generated events such as iteration_started,
        tool_call_started, tool_call_completed and agent_completed alongside
        the server's normal text/reasoning/tool events.
        """
        self._recent_tool_calls.clear()
        self._tool_result_cache.clear()
        self.messages.append(ChatMessage.user(prompt))
        trace_log = AgentTrace()
        self.last_trace = trace_log
        if trace:
            started_event = self._emit_trace(
                trace_log,
                "agent_started",
                text=prompt,
            )
            yield self._trace_stream_event(
                trace_log,
                "agent_started",
                trace_event=started_event,
            )

        force_no_tools = False

        try:
            for iteration in range(1, self.max_iterations + 1):
                if trace:
                    self._emit_trace(
                        trace_log, "iteration_started", iteration=iteration
                    )
                    yield self._trace_stream_event(
                        trace_log, "iteration_started", iteration=iteration
                    )

                request_kwargs = {
                    k: v for k, v in self.defaults.items() if v is not None
                }
                request_kwargs.update(kwargs)
                events = self.client.chat.create(
                    self.messages,
                    model=self.model,
                    stream=True,
                    tools=[]
                    if force_no_tools
                    else [item.definition() for item in self.tools.values()],
                    tool_choice="none"
                    if force_no_tools
                    else (self.tool_choice if self.tools else "none"),
                    **request_kwargs,
                )
                assert not isinstance(events, ChatResponse)
                text = ""
                reasoning = ""
                tool_acc: dict[str, dict[str, str]] = {}
                finish = None
                warning = None
                for event in events:
                    text += event.text
                    reasoning += event.reasoning
                    for call in event.tool_calls:
                        key = call.id or f"tool-{len(tool_acc)}"
                        item = tool_acc.setdefault(
                            key,
                            {"id": call.id, "name": "", "arguments": ""},
                        )
                        if call.id and not item.get("id"):
                            item["id"] = call.id
                        if call.function.name:
                            item["name"] += call.function.name
                        item["arguments"] += call.function.arguments
                    finish = (event.data.get("choices") or [{}])[0].get(
                        "finish_reason"
                    ) or finish
                    warning = event.warning or warning
                    yield event

                if trace:
                    self._emit_trace(
                        trace_log,
                        "model_response",
                        iteration=iteration,
                        text=text,
                        data={"finish_reason": finish, "warning": warning},
                    )

                if tool_acc and finish == "length":
                    bad = next(
                        (
                            item
                            for item in tool_acc.values()
                            if not _valid_json_object(item["arguments"])
                        ),
                        None,
                    )
                    if warning is not None or bad is not None:
                        raise ToolCallIncompleteError(
                            finish_reason=finish,
                            max_output_tokens=kwargs.get(
                                "max_output_tokens",
                                self.defaults.get("max_output_tokens"),
                            )
                            or 1024,
                            tool_name=bad["name"] if bad else None,
                            warning=warning,
                        )

                assistant = ChatMessage.assistant(
                    text or None, reasoning_content=reasoning or None
                )
                if tool_acc:
                    assistant.tool_calls = [
                        ToolCall(
                            id=item.get("id") or call_id,
                            function=ToolCallFunction(
                                name=item["name"],
                                arguments=item["arguments"],
                            ),
                        )
                        for call_id, item in tool_acc.items()
                    ]
                    self.messages.append(assistant)
                    for call in assistant.tool_calls:
                        if trace:
                            start_event = self._emit_trace(
                                trace_log,
                                "tool_call_started",
                                iteration=iteration,
                                name=call.function.name,
                                call_id=call.id,
                                arguments=_parse_arguments(call.function.arguments),
                            )
                            yield self._trace_stream_event(
                                trace_log,
                                "tool_call_started",
                                iteration=iteration,
                                trace_event=start_event,
                            )
                        signature = _tool_signature(
                            call.function.name, call.function.arguments
                        )
                        repeats = self._recent_tool_calls.count(signature)
                        if repeats >= self.max_same_tool_calls:
                            cached = self._tool_result_cache.get(signature)
                            blocked_message = (
                                f"Duplicate tool call detected for {call.function.name}. "
                                "Use the existing result and continue without calling this tool again."
                            )
                            result = ToolResult(
                                call.id,
                                call.function.name,
                                cached.content if cached else blocked_message,
                                success=cached.success if cached else False,
                                error=cached.error if cached else "duplicate_tool_call",
                            )
                            force_no_tools = True
                            if trace:
                                blocked = self._emit_trace(
                                    trace_log,
                                    "tool_call_blocked",
                                    iteration=iteration,
                                    name=call.function.name,
                                    call_id=call.id,
                                    arguments=_parse_arguments(call.function.arguments),
                                    result=result.content,
                                    success=False,
                                    error="duplicate_tool_call",
                                    data={
                                        "cached": cached is not None,
                                        "force_no_tools": True,
                                    },
                                )
                                yield self._trace_stream_event(
                                    trace_log, "tool_call_blocked", trace_event=blocked
                                )
                            duration_ms = 0.0
                        else:
                            self._recent_tool_calls.append(signature)
                            started = time.perf_counter()
                            result = self._execute(
                                call.id, call.function.name, call.function.arguments
                            )
                            duration_ms = (time.perf_counter() - started) * 1000
                            self._tool_result_cache[signature] = result
                        self.messages.append(
                            ChatMessage.tool(
                                result.content, result.tool_call_id, name=result.name
                            )
                        )
                        if trace:
                            completed = self._emit_trace(
                                trace_log,
                                "tool_call_completed",
                                iteration=iteration,
                                name=call.function.name,
                                call_id=call.id,
                                arguments=_parse_arguments(call.function.arguments),
                                result=_parse_result(result.content),
                                duration_ms=duration_ms,
                                success=result.success,
                                error=result.error,
                            )
                            yield self._trace_stream_event(
                                trace_log,
                                "tool_call_completed",
                                iteration=iteration,
                                trace_event=completed,
                            )
                    continue

                self.messages.append(assistant)
                if trace:
                    completed = self._emit_trace(
                        trace_log, "agent_completed", iteration=iteration, text=text
                    )
                    yield self._trace_stream_event(
                        trace_log,
                        "agent_completed",
                        iteration=iteration,
                        trace_event=completed,
                    )
                    trace_log.finish()
                return
            raise RuntimeError(f"Agent exceeded max_iterations={self.max_iterations}")
        except Exception as exc:
            if trace:
                error = self._emit_trace(
                    trace_log, "error", error=str(exc), success=False
                )
                yield self._trace_stream_event(trace_log, "error", trace_event=error)
                trace_log.finish()
            raise

    def _trace_stream_event(
        self,
        trace: AgentTrace,
        event_type: str,
        *,
        iteration: int | None = None,
        trace_event: TraceEvent | None = None,
        text: str = "",
    ) -> StreamEvent:
        event = trace_event or trace.events[-1]
        return StreamEvent(
            type=event_type,
            text=text or event.text,
            data={
                "trace": True,
                "timestamp": event.timestamp,
                "iteration": event.iteration
                if event.iteration is not None
                else iteration,
                "name": event.name,
                "call_id": event.call_id,
                "arguments": event.arguments,
                "result": event.result,
                "duration_ms": event.duration_ms,
                "success": event.success,
                "error": event.error,
                **event.data,
            },
        )

    def _execute_traced(
        self, trace: AgentTrace, iteration: int, call_id: str, name: str, arguments: str
    ) -> ToolResult:
        self._emit_trace(
            trace,
            "tool_call_started",
            iteration=iteration,
            name=name,
            call_id=call_id,
            arguments=_parse_arguments(arguments),
        )
        signature = _tool_signature(name, arguments)
        repeats = self._recent_tool_calls.count(signature)
        if repeats >= self.max_same_tool_calls:
            cached = self._tool_result_cache.get(signature)
            message = (
                f"Duplicate tool call detected for {name}. "
                "The same call was already executed. Use the existing result and continue without calling this tool again."
            )
            self._emit_trace(
                trace,
                "tool_call_blocked",
                iteration=iteration,
                name=name,
                call_id=call_id,
                arguments=_parse_arguments(arguments),
                result=cached.content if cached else message,
                success=False,
                error="duplicate_tool_call",
                data={"cached": cached is not None, "force_no_tools": True},
            )
            if cached:
                return ToolResult(
                    call_id,
                    name,
                    cached.content,
                    success=cached.success,
                    error=cached.error,
                )
            return ToolResult(
                call_id, name, message, success=False, error="duplicate_tool_call"
            )
        self._recent_tool_calls.append(signature)
        started = time.perf_counter()
        result = self._execute(call_id, name, arguments)
        duration_ms = (time.perf_counter() - started) * 1000
        self._tool_result_cache[signature] = result
        self._emit_trace(
            trace,
            "tool_call_completed",
            iteration=iteration,
            name=name,
            call_id=call_id,
            arguments=_parse_arguments(arguments),
            result=_parse_result(result.content),
            duration_ms=duration_ms,
            success=result.success,
            error=result.error,
        )
        return result

    def _execute(self, call_id: str, name: str, arguments: str) -> ToolResult:
        tool = self.tools.get(name)
        if tool is None:
            return ToolResult(
                call_id,
                name,
                f"Unknown tool: {name}",
                success=False,
                error="unknown_tool",
            )
        try:
            value = tool.call(arguments)
            return ToolResult(call_id, name, _serialize_result(value))
        except ToolExecutionError as exc:
            if self.tool_error_mode == "raise":
                raise
            return ToolResult(
                call_id,
                name,
                f"Tool error: {exc.message}",
                success=False,
                error=exc.message,
            )
        except Exception as exc:
            if self.tool_error_mode == "raise":
                raise
            return ToolResult(
                call_id, name, f"Tool error: {exc}", success=False, error=str(exc)
            )

    def conversation(self):
        from .conversation import Conversation

        return Conversation(self.client, messages=self.messages, model=self.model)


class AsyncAgent(Agent):
    async def run(
        self,
        prompt: str,
        *,
        response_model: type[T] | None = None,
        trace: bool = False,
        **kwargs: Any,
    ) -> Any:
        result = await self._run_async(prompt, response_model=response_model, **kwargs)
        return result if trace else result.response

    async def run_with_trace(
        self, prompt: str, *, response_model: type[T] | None = None, **kwargs: Any
    ) -> AgentRunResult:
        return await self._run_async(prompt, response_model=response_model, **kwargs)

    async def _run_async(
        self, prompt: str, *, response_model: type[T] | None = None, **kwargs: Any
    ) -> AgentRunResult:
        trace = AgentTrace()
        self.last_trace = trace
        self._recent_tool_calls.clear()
        self._tool_result_cache.clear()
        self.messages.append(ChatMessage.user(prompt))
        self._emit_trace(trace, "agent_started", text=prompt)
        retries_left = self.retry_incomplete_tool_calls
        current_max_tokens = (
            kwargs.get("max_output_tokens", self.defaults.get("max_output_tokens"))
            or 1024
        )
        force_no_tools = False

        try:
            for iteration in range(1, self.max_iterations + 1):
                self._emit_trace(trace, "iteration_started", iteration=iteration)
                request_kwargs = {
                    k: v for k, v in self.defaults.items() if v is not None
                }
                request_kwargs.update(kwargs)
                request_kwargs["max_output_tokens"] = current_max_tokens
                response = await self.client.chat.create(
                    self.messages,
                    model=self.model,
                    tools=[item.definition() for item in self.tools.values()],
                    tool_choice=self.tool_choice if self.tools else "none",
                    **request_kwargs,
                )
                assert isinstance(response, ChatResponse)
                self._emit_trace(
                    trace,
                    "model_response",
                    iteration=iteration,
                    data={
                        "finish_reason": response.finish_reason,
                        "usage": response.usage,
                    },
                    text=response.text,
                )
                message = response.message
                if message is None:
                    self._emit_trace(trace, "agent_completed", iteration=iteration)
                    return AgentRunResult(response, trace.finish())
                if message.tool_calls:
                    invalid = next(
                        (
                            call
                            for call in message.tool_calls
                            if not _valid_json_object(call.function.arguments)
                        ),
                        None,
                    )
                    if invalid is not None and response.finish_reason == "length":
                        if retries_left > 0:
                            retries_left -= 1
                            current_max_tokens *= 2
                            self._emit_trace(
                                trace,
                                "tool_call_retry",
                                iteration=iteration,
                                name=invalid.function.name,
                                call_id=invalid.id,
                                data={"max_output_tokens": current_max_tokens},
                            )
                            continue
                        raise ToolCallIncompleteError(
                            finish_reason=response.finish_reason,
                            max_output_tokens=current_max_tokens,
                            tool_name=invalid.function.name or None,
                        )
                    self.messages.append(message)
                    for call in message.tool_calls:
                        result = await self._execute_async_traced(
                            trace,
                            iteration,
                            call.id,
                            call.function.name,
                            call.function.arguments,
                        )
                        self.messages.append(
                            ChatMessage.tool(
                                result.content, result.tool_call_id, name=result.name
                            )
                        )
                    continue
                self.messages.append(message)
                output = _parse_output(response, response_model)
                self._emit_trace(
                    trace, "agent_completed", iteration=iteration, text=response.text
                )
                return AgentRunResult(output, trace.finish())
            raise RuntimeError(f"Agent exceeded max_iterations={self.max_iterations}")
        except Exception as exc:
            self._emit_trace(trace, "error", error=str(exc), success=False)
            trace.finish()
            raise

    async def stream(
        self, prompt: str, *, trace: bool = True, **kwargs: Any
    ) -> AsyncIterator[StreamEvent]:
        self.messages.append(ChatMessage.user(prompt))
        trace_log = AgentTrace()
        self.last_trace = trace_log
        if trace:
            started_event = self._emit_trace(
                trace_log,
                "agent_started",
                text=prompt,
            )
            yield self._trace_stream_event(
                trace_log,
                "agent_started",
                trace_event=started_event,
            )
        try:
            for iteration in range(1, self.max_iterations + 1):
                if trace:
                    self._emit_trace(
                        trace_log, "iteration_started", iteration=iteration
                    )
                    yield self._trace_stream_event(
                        trace_log, "iteration_started", iteration=iteration
                    )
                request_kwargs = {
                    k: v for k, v in self.defaults.items() if v is not None
                }
                request_kwargs.update(kwargs)
                events = await self.client.chat.create(
                    self.messages,
                    model=self.model,
                    stream=True,
                    tools=[]
                    if force_no_tools
                    else [item.definition() for item in self.tools.values()],
                    tool_choice="none"
                    if force_no_tools
                    else (self.tool_choice if self.tools else "none"),
                    **request_kwargs,
                )
                assert not isinstance(events, ChatResponse)
                text = ""
                reasoning = ""
                tool_acc: dict[str, dict[str, str]] = {}
                finish = None
                async for event in events:
                    text += event.text
                    reasoning += event.reasoning
                    for call in event.tool_calls:
                        key = call.id or f"tool-{len(tool_acc)}"
                        item = tool_acc.setdefault(
                            key,
                            {"id": call.id, "name": "", "arguments": ""},
                        )
                        if call.id and not item.get("id"):
                            item["id"] = call.id
                        if call.function.name:
                            item["name"] += call.function.name
                        item["arguments"] += call.function.arguments
                    finish = (event.data.get("choices") or [{}])[0].get(
                        "finish_reason"
                    ) or finish
                    yield event
                if trace:
                    self._emit_trace(
                        trace_log,
                        "model_response",
                        iteration=iteration,
                        text=text,
                        data={"finish_reason": finish},
                    )
                assistant = ChatMessage.assistant(
                    text or None, reasoning_content=reasoning or None
                )
                if tool_acc:
                    assistant.tool_calls = [
                        ToolCall(
                            id=i,
                            function=ToolCallFunction(
                                name=v["name"], arguments=v["arguments"]
                            ),
                        )
                        for i, v in tool_acc.items()
                    ]
                    self.messages.append(assistant)
                    for call in assistant.tool_calls:
                        if trace:
                            started_event = self._emit_trace(
                                trace_log,
                                "tool_call_started",
                                iteration=iteration,
                                name=call.function.name,
                                call_id=call.id,
                                arguments=_parse_arguments(call.function.arguments),
                            )
                            yield self._trace_stream_event(
                                trace_log,
                                "tool_call_started",
                                iteration=iteration,
                                trace_event=started_event,
                            )
                        started = time.perf_counter()
                        result = await self._execute_async(
                            call.id, call.function.name, call.function.arguments
                        )
                        duration_ms = (time.perf_counter() - started) * 1000
                        self.messages.append(
                            ChatMessage.tool(
                                result.content, result.tool_call_id, name=result.name
                            )
                        )
                        if trace:
                            completed = self._emit_trace(
                                trace_log,
                                "tool_call_completed",
                                iteration=iteration,
                                name=call.function.name,
                                call_id=call.id,
                                arguments=_parse_arguments(call.function.arguments),
                                result=_parse_result(result.content),
                                duration_ms=duration_ms,
                                success=result.success,
                                error=result.error,
                            )
                            yield self._trace_stream_event(
                                trace_log,
                                "tool_call_completed",
                                iteration=iteration,
                                trace_event=completed,
                            )
                    continue
                self.messages.append(assistant)
                if trace:
                    completed = self._emit_trace(
                        trace_log, "agent_completed", iteration=iteration, text=text
                    )
                    yield self._trace_stream_event(
                        trace_log,
                        "agent_completed",
                        iteration=iteration,
                        trace_event=completed,
                    )
                    trace_log.finish()
                return
            raise RuntimeError(f"Agent exceeded max_iterations={self.max_iterations}")
        except Exception as exc:
            if trace:
                error = self._emit_trace(
                    trace_log, "error", error=str(exc), success=False
                )
                yield self._trace_stream_event(trace_log, "error", trace_event=error)
                trace_log.finish()
            raise

    async def _execute_async(
        self, call_id: str, name: str, arguments: str
    ) -> ToolResult:
        tool = self.tools.get(name)
        if tool is None:
            return ToolResult(
                call_id,
                name,
                f"Unknown tool: {name}",
                success=False,
                error="unknown_tool",
            )
        try:
            args = json.loads(arguments)
            value = tool.handler(**args)
            if inspect.isawaitable(value):
                value = await value
            return ToolResult(call_id, name, _serialize_result(value))
        except Exception as exc:
            if self.tool_error_mode == "raise":
                raise
            return ToolResult(
                call_id, name, f"Tool error: {exc}", success=False, error=str(exc)
            )

    async def _execute_async_traced(
        self, trace: AgentTrace, iteration: int, call_id: str, name: str, arguments: str
    ) -> ToolResult:
        self._emit_trace(
            trace,
            "tool_call_started",
            iteration=iteration,
            name=name,
            call_id=call_id,
            arguments=_parse_arguments(arguments),
        )
        started = time.perf_counter()
        result = await self._execute_async(call_id, name, arguments)
        duration_ms = (time.perf_counter() - started) * 1000
        self._emit_trace(
            trace,
            "tool_call_completed",
            iteration=iteration,
            name=name,
            call_id=call_id,
            arguments=_parse_arguments(arguments),
            result=_parse_result(result.content),
            duration_ms=duration_ms,
            success=result.success,
            error=result.error,
        )
        return result


def _tool_signature(name: str, arguments: str) -> tuple[str, str]:
    try:
        normalized = json.dumps(
            json.loads(arguments),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    except Exception:
        normalized = arguments.strip()
    return name, normalized


def _valid_json_object(arguments: str) -> bool:
    try:
        value = json.loads(arguments)
    except (TypeError, json.JSONDecodeError):
        return False
    return isinstance(value, dict)


def _parse_arguments(arguments: str) -> Any:
    try:
        return json.loads(arguments)
    except Exception:
        return arguments


def _parse_result(content: str) -> Any:
    try:
        return json.loads(content)
    except Exception:
        return content


def _serialize_result(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


def _parse_output(response: ChatResponse, response_model: type[T] | None) -> Any:
    if response_model is None:
        return response
    return response_model.model_validate(json.loads(response.text))
