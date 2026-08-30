"""ARC Inference Client — streaming multi-tool agent example."""

from __future__ import annotations

import math
from datetime import datetime

from arc_inference import InferenceClient, tool


@tool
def get_weather(city: str) -> dict:
    """Get the current demo weather for a city."""
    data = {
        "Berlin": {"temperature_c": 23, "condition": "sunny", "humidity": 48},
        "Munich": {"temperature_c": 21, "condition": "partly cloudy", "humidity": 55},
        "Hamburg": {"temperature_c": 19, "condition": "cloudy", "humidity": 68},
        "London": {"temperature_c": 17, "condition": "light rain", "humidity": 72},
    }
    return data.get(
        city,
        {"temperature_c": 20, "condition": "unknown", "humidity": 50},
    )


@tool
def convert_temperature(celsius: float, unit: str) -> float:
    """Convert Celsius to Fahrenheit or Kelvin."""
    unit = unit.lower()

    if unit in {"f", "fahrenheit"}:
        return round(celsius * 9 / 5 + 32, 2)

    if unit in {"k", "kelvin"}:
        return round(celsius + 273.15, 2)

    raise ValueError("unit must be Fahrenheit/F or Kelvin/K")


@tool
def get_time(city: str) -> str:
    """Return the current demo local time for a city."""
    return f"{city}: {datetime.now():%Y-%m-%d %H:%M:%S} (demo local time)"


@tool
def calculate_distance_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Calculate great-circle distance between coordinates in kilometers."""
    radius = 6371.0

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)

    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2

    return round(
        2 * radius * math.asin(math.sqrt(a)),
        2,
    )


@tool
def lookup_city(city: str) -> dict:
    """Return demo coordinates for a city."""
    cities = {
        "Berlin": {
            "country": "Germany",
            "latitude": 52.5200,
            "longitude": 13.4050,
        },
        "London": {
            "country": "United Kingdom",
            "latitude": 51.5074,
            "longitude": -0.1278,
        },
        "Munich": {
            "country": "Germany",
            "latitude": 48.1351,
            "longitude": 11.5820,
        },
        "Hamburg": {
            "country": "Germany",
            "latitude": 53.5511,
            "longitude": 9.9937,
        },
    }

    return cities.get(
        city,
        {
            "country": "Unknown",
            "latitude": 0.0,
            "longitude": 0.0,
        },
    )


@tool
def get_currency_rate(base: str, quote: str) -> float:
    """Return a demo exchange rate."""
    rates = {
        ("EUR", "USD"): 1.17,
        ("USD", "EUR"): 0.85,
        ("EUR", "GBP"): 0.87,
        ("GBP", "EUR"): 1.15,
    }

    key = (base.upper(), quote.upper())

    if key not in rates:
        raise ValueError(f"No demo rate for {base}/{quote}")

    return rates[key]


TOOLS = [
    get_weather,
    convert_temperature,
    get_time,
    calculate_distance_km,
    lookup_city,
    get_currency_rate,
]


def main() -> None:
    question = (
        "I am planning a trip from Berlin to London. "
        "Tell me Berlin's current demo weather, convert its temperature "
        "to Fahrenheit, give me the demo EUR to GBP exchange rate, "
        "calculate the distance from Berlin to London using their city "
        "coordinates, and tell me the current demo time in London. "
        "Then summarize everything in a compact travel snapshot."
    )

    with InferenceClient() as client:
        agent = client.agent(
            instructions=(
                "You are a helpful assistant. "
                "Use tools whenever they provide the required information. "
                "For multi-part questions, use all relevant tools. "
                "Never invent tool results."
            ),
            tools=TOOLS,
            max_output_tokens=2048,
            temperature=0.0,
            max_iterations=10,
            tool_choice="auto",
            retry_incomplete_tool_calls=2,
        )

        print("USER:")
        print(question)

        print("\nASSISTANT:\n")

        for event in agent.stream(question):
            # Normal model text
            if event.type == "text":
                print(event.text, end="", flush=True)

            # Optional reasoning stream
            elif event.type == "reasoning":
                # Uncomment if you want reasoning content displayed.
                # print(event.reasoning, end="", flush=True)
                pass

            # SDK transparency: agent started
            elif event.type == "agent_started":
                print("\n[AGENT START]")

            # SDK transparency: new model iteration
            elif event.type == "iteration_started":
                print(f"\n[ITERATION {event.data.get('iteration')}]")

            # SDK transparency: tool execution started
            elif event.type == "tool_call_started":
                print(
                    f"\n[TOOL START] "
                    f"{event.data.get('name')}("
                    f"{event.data.get('arguments')})"
                )

            # SDK transparency: tool execution completed
            elif event.type == "tool_call_completed":
                status = "OK" if event.data.get("success") else "FAILED"

                print(
                    f"[TOOL {status}] "
                    f"{event.data.get('name')} -> "
                    f"{event.data.get('result')} "
                    f"({event.data.get('duration_ms', 0):.2f} ms)"
                )

            # SDK transparency: incomplete tool-call retry
            elif event.type == "tool_call_retry":
                print(
                    f"\n[TOOL RETRY] "
                    f"{event.data.get('name')} "
                    f"(max_output_tokens="
                    f"{event.data.get('max_output_tokens')})"
                )

            # SDK transparency: finished
            elif event.type == "agent_completed":
                print(
                    f"\n\n[AGENT COMPLETED]\nIterations: {event.data.get('iteration')}"
                )

            # SDK transparency: error
            elif event.type == "error":
                print(f"\n[ERROR] {event.data.get('error')}")

        print("\n")


if __name__ == "__main__":
    main()
