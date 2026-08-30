"""ARC Inference Client multi-tool agent example."""

from __future__ import annotations

import math
from datetime import datetime

from arc_inference import InferenceClient, tool


@tool
def get_weather(city: str) -> dict:
    """Get the current weather for a city."""
    # Demo data. Replace with a real weather API in production.
    data = {
        "Berlin": {"temperature_c": 23, "condition": "sunny", "humidity": 48},
        "Munich": {"temperature_c": 21, "condition": "partly cloudy", "humidity": 55},
        "Hamburg": {"temperature_c": 19, "condition": "cloudy", "humidity": 68},
        "London": {"temperature_c": 17, "condition": "light rain", "humidity": 72},
    }
    return data.get(city, {"temperature_c": 20, "condition": "unknown", "humidity": 50})


@tool
def calculate(expression: str) -> float:
    """Calculate a basic arithmetic expression using numbers and + - * / parentheses."""
    allowed = set("0123456789.+-*/() ")
    if not expression or any(char not in allowed for char in expression):
        raise ValueError("Only basic arithmetic expressions are allowed.")
    return float(eval(expression, {"__builtins__": {}}, {}))


@tool
def convert_temperature(celsius: float, unit: str) -> float:
    """Convert a Celsius temperature to Fahrenheit or Kelvin."""
    unit = unit.lower()
    if unit in {"f", "fahrenheit"}:
        return round(celsius * 9 / 5 + 32, 2)
    if unit in {"k", "kelvin"}:
        return round(celsius + 273.15, 2)
    raise ValueError("unit must be Fahrenheit/F or Kelvin/K")


@tool
def get_time(city: str) -> str:
    """Return a demo local time for a city."""
    # Demo only; replace with timezone-aware logic for production.
    return f"{city}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (demo local time)"


@tool
def calculate_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two coordinates in kilometers."""
    radius = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return round(2 * radius * math.asin(math.sqrt(a)), 2)


@tool
def lookup_city(city: str) -> dict:
    """Return demo coordinates and country information for a city."""
    cities = {
        "Berlin": {"country": "Germany", "latitude": 52.5200, "longitude": 13.4050},
        "Munich": {"country": "Germany", "latitude": 48.1351, "longitude": 11.5820},
        "Hamburg": {"country": "Germany", "latitude": 53.5511, "longitude": 9.9937},
        "London": {"country": "United Kingdom", "latitude": 51.5074, "longitude": -0.1278},
    }
    return cities.get(city, {"country": "Unknown", "latitude": 0.0, "longitude": 0.0})


@tool
def get_currency_rate(base: str, quote: str) -> float:
    """Return a demo exchange rate from base currency to quote currency."""
    rates = {
        ("EUR", "USD"): 1.17,
        ("USD", "EUR"): 0.85,
        ("EUR", "GBP"): 0.87,
        ("GBP", "EUR"): 1.15,
    }
    try:
        return rates[(base.upper(), quote.upper())]
    except KeyError as exc:
        raise ValueError(f"No demo rate for {base.upper()}/{quote.upper()}") from exc


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


TOOLS = [
    get_weather,
    calculate,
    convert_temperature,
    get_time,
    calculate_distance_km,
    lookup_city,
    get_currency_rate,
    multiply,
]


with InferenceClient("http://localhost:7842/v1") as client:
    agent = client.agent(
        instructions=(
            "You are a helpful general-purpose assistant. "
            "Use tools when they provide useful information. "
            "For multi-part questions, use every relevant tool and combine the results. "
            "Never invent tool results."
        ),
        tools=TOOLS,
        max_output_tokens=2048,
        temperature=0.0,
        max_iterations=10,
        retry_incomplete_tool_calls=2,
    )

    question = (
        "I am planning a trip from Berlin to London. "
        "Tell me Berlin's current demo weather, convert its temperature to Fahrenheit, "
        "give me the demo EUR to GBP exchange rate, "
        "calculate the distance from Berlin to London using their city coordinates, "
        "and tell me the current demo time in London. "
        "Then summarize all of that in a compact travel snapshot."
    )

    print("USER:", question)
    print("\nASSISTANT:")
    result = agent.run_with_trace(question)
    print(result.text)

    print("\nEXECUTION TRACE:")
    for event in result.trace.events:
        if event.type == "tool_call_started":
            print(f"[TOOL START] {event.name}({event.arguments})")
        elif event.type == "tool_call_completed":
            status = "ok" if event.success else "failed"
            print(f"[TOOL {status.upper()}] {event.name} -> {event.result} ({event.duration_ms:.2f} ms)")
        elif event.type == "tool_call_retry":
            print(f"[TOOL RETRY] {event.name} with max_output_tokens={event.data['max_output_tokens']}")

    print(f"\nTRACE: {len(result.trace.events)} events, {len(result.trace.tool_calls)} tool calls, {result.trace.duration_ms:.2f} ms")

    print("\nTOOLS AVAILABLE:")
    for item in TOOLS:
        print(f"- {item.name}")
