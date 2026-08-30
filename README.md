# ARC Inference Client

A small Python SDK for the FastAPI inference API:

- `/v1/chat/completions`
- `/v1/completions`
- `/v1/embeddings`
- `/v1/models`
- `/v1/model`

It provides a low-level client plus stateful conversations and automatic agent tool loops.

## Install

```bash
uv pip install -e .
```

## Basic chat

```python
from arc_inference import InferenceClient

client = InferenceClient("http://localhost:7842/v1")

response = client.chat.create([
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello"},
], temperature=0.7)

print(response.text)
client.close()
```

## Conversation

```python
conversation = client.conversation(
    system="You are a concise assistant.",
)

print(conversation.send("Hello").text)
print(conversation.send("What did I just say?").text)
```

## Streaming

```python
for event in conversation.stream("Explain agents in one paragraph"):
    if event.type == "text":
        print(event.text, end="", flush=True)
```

## Tools

```python
from arc_inference import tool

@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

agent = client.agent(
    instructions="You are a helpful assistant.",
    tools=[add],
    max_iterations=8,
)

result = agent.run("What is 12 + 30?")
print(result.text)
```

The SDK automatically handles model → tool call → tool result → model iterations.

## Async

```python
from arc_inference import AsyncInferenceClient

async with AsyncInferenceClient("http://localhost:7842/v1") as client:
    conversation = client.conversation(system="You are helpful.")
    response = await conversation.send("Hello")
    print(response.text)
```

Async tools may be normal or `async def` functions.

## Structured output

```python
from pydantic import BaseModel

class Answer(BaseModel):
    answer: str
    confidence: float

result = client.agent().run(
    "Return an answer about Python",
    response_model=Answer,
)
print(result.answer)
```

This uses your API's `response_format` field and expects JSON output.

## Tool-calling reliability

For agent runs, the SDK defaults to `max_output_tokens=1024`. If the server reports `finish_reason="length"` while a tool call's JSON arguments are incomplete, the agent automatically retries once with double the output budget instead of feeding malformed arguments back into the model.

You can tune this explicitly:

```python
agent = client.agent(
    tools=[get_weather],
    max_output_tokens=2048,
    retry_incomplete_tool_calls=2,
)
```

If retries are exhausted, the SDK raises `ToolCallIncompleteError` with a clear message. A persistent XML/tool-parser mismatch must still be fixed in the inference engine configuration; the client cannot infer or change the model's server-side tool-call tags.

## Agent transparency

Agent execution can be inspected without changing the normal `run()` API:

```python
result = agent.run("Plan my trip", trace=True)
print(result.text)

for event in result.trace.events:
    print(event.type, event.name, event.duration_ms)
```

Or use the explicit form:

```python
result = agent.run_with_trace("Plan my trip")
print(result.trace.duration_ms)
for tool in result.trace.tool_summary():
    print(tool.name, tool.arguments, tool.result, tool.duration_ms)
```

For interactive agents, `agent.stream()` emits SDK trace events alongside model events:

```python
for event in agent.stream("Plan my trip"):
    if event.type == "tool_call_started":
        print(f"[tool] {event.data['name']}({event.data['arguments']})")
    elif event.type == "tool_call_completed":
        print(f"[result] {event.data['result']} ({event.data['duration_ms']:.1f} ms)")
    elif event.type == "text":
        print(event.text, end="")
```

You can also receive every trace event through `trace_callback=` when creating an agent.

The default ARC inference endpoint is `http://localhost:7842/v1`.
