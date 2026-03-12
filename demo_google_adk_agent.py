"""
=============================================================================
DEMO: Google Agent Development Kit (ADK)
=============================================================================
Pattern  : Gemini-backed agent with Python-function tools + sub-agents (A2A)
Framework: Google ADK (google-adk)
LLM      : Gemini 2.0 Flash (via Google AI or Vertex AI)

Install deps:
    pip install google-adk google-generativeai

Set env var (Google AI Studio key — free tier available):
    export GOOGLE_API_KEY="AIza..."

OR for Vertex AI (enterprise):
    export GOOGLE_CLOUD_PROJECT="your-project-id"
    export GOOGLE_CLOUD_LOCATION="us-central1"
    gcloud auth application-default login
=============================================================================
"""

import asyncio
import os
from google.adk.agents import Agent
from google.adk.tools import google_search
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import AgentTool
from google.genai import types


# ─── 1. Define tools as plain Python functions ───────────────────────────────
#   ADK introspects the type hints and docstring to describe the tool to the LLM.

def get_weather(city: str) -> dict:
    """
    Return current weather conditions for the specified city.

    Args:
        city: Name of the city (e.g. 'Hyderabad', 'Mumbai')

    Returns:
        A dict with temperature, condition, and humidity.
    """
    # --- Replace with a real weather API call in production ---
    mock_data = {
        "Hyderabad": {"temperature": "32°C", "condition": "Partly Cloudy", "humidity": "68%"},
        "Mumbai":    {"temperature": "29°C", "condition": "Humid",         "humidity": "82%"},
        "Bangalore": {"temperature": "24°C", "condition": "Mild",          "humidity": "55%"},
    }
    return mock_data.get(city, {"temperature": "Unknown", "condition": "N/A", "humidity": "N/A"})


def check_calendar(date: str) -> dict:
    """
    Retrieve scheduled events for a given date.

    Args:
        date: Date string in YYYY-MM-DD format.

    Returns:
        A dict with a list of events for that day.
    """
    # --- Replace with Google Calendar API call in production ---
    mock_events = {
        "2024-12-10": ["09:00 — Team standup (30 min)", "14:00 — Client demo call (1 hr)"],
        "2024-12-11": ["10:00 — Architecture review", "16:00 — 1:1 with manager"],
    }
    return {"date": date, "events": mock_events.get(date, ["No events scheduled"])}


def get_news_headlines(topic: str, count: int = 3) -> list:
    """
    Fetch the latest news headlines for a given topic.

    Args:
        topic: The subject to search news for (e.g. 'AI agents', 'LLM').
        count: Number of headlines to return (default 3).

    Returns:
        A list of headline strings.
    """
    # --- Replace with a real news API (NewsAPI, GNews, etc.) ---
    return [
        f"[Mock headline {i+1}] Latest development in {topic} — 2024"
        for i in range(count)
    ]


# ─── 2. Build a specialist sub-agent (weather domain) ────────────────────────

weather_agent = Agent(
    name="weather_specialist",
    model="gemini-2.0-flash-exp",
    description="Handles all weather-related queries for Indian cities.",
    instruction=(
        "You are a weather specialist. Only answer questions about weather. "
        "Always call get_weather tool with the city name provided."
    ),
    tools=[get_weather],
)


# ─── 3. Build the root orchestrator agent ────────────────────────────────────

root_agent = Agent(
    name="personal_assistant",
    model="gemini-2.0-flash-exp",
    description="A proactive personal assistant for planning and information.",
    instruction=(
        "You are a helpful personal assistant. "
        "For weather queries, delegate to the weather_specialist sub-agent. "
        "For calendar questions, use check_calendar. "
        "For current news, use get_news_headlines. "
        "For general knowledge, use google_search. "
        "Always be concise and structured in your responses."
    ),
    tools=[
        AgentTool(agent=weather_agent),  # sub-agent exposed as a tool (A2A)
        check_calendar,
        get_news_headlines,
        google_search,                   # built-in ADK Google Search tool
    ],
)


# ─── 4. Session service + Runner ─────────────────────────────────────────────

APP_NAME   = "assistant_demo"
session_svc = InMemorySessionService()
# For production swap InMemorySessionService with:
#   from google.adk.sessions import VertexAiSessionService

runner = Runner(
    agent=root_agent,
    app_name=APP_NAME,
    session_service=session_svc,
)


# ─── 5. Helper: send a message and print the response ────────────────────────

async def chat(user_id: str, session_id: str, message: str) -> str:
    """Send a message to the agent and return the text response."""
    content = types.Content(
        role="user",
        parts=[types.Part(text=message)]
    )
    final_response = ""
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content,
    ):
        if event.is_final_response():
            final_response = event.content.parts[0].text
    return final_response


# ─── 6. Run demos ────────────────────────────────────────────────────────────

async def main():
    USER_ID = "demo_user_1"

    # Create a session (persistent state per user)
    session = await session_svc.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        state={
            "user_city": "Hyderabad",
            "preferences": {"language": "English", "units": "metric"},
        }
    )
    SID = session.id
    print(f"\nSession created: {SID}\n")

    # --- DEMO 1: Weather via sub-agent (A2A delegation) ---
    print("=" * 60)
    print("DEMO 1 — Weather query (delegated to sub-agent)")
    print("=" * 60)
    resp = await chat(USER_ID, SID, "What is the weather like in Hyderabad today?")
    print("[RESPONSE]", resp)

    # --- DEMO 2: Calendar lookup ---
    print("\n" + "=" * 60)
    print("DEMO 2 — Calendar query")
    print("=" * 60)
    resp = await chat(USER_ID, SID, "What meetings do I have on 2024-12-10?")
    print("[RESPONSE]", resp)

    # --- DEMO 3: Combined planning query ---
    print("\n" + "=" * 60)
    print("DEMO 3 — Multi-tool: plan my day (weather + calendar)")
    print("=" * 60)
    resp = await chat(
        USER_ID, SID,
        "Help me plan my day on 2024-12-10 in Hyderabad — "
        "what's the weather and what meetings do I have?"
    )
    print("[RESPONSE]", resp)

    # --- DEMO 4: State persistence (same session, new question) ---
    print("\n" + "=" * 60)
    print("DEMO 4 — State persistence: follow-up question in same session")
    print("=" * 60)
    resp = await chat(USER_ID, SID, "Based on the weather, should I carry an umbrella?")
    print("[RESPONSE]", resp)

    # --- DEMO 5: News query (uses google_search or get_news_headlines) ---
    print("\n" + "=" * 60)
    print("DEMO 5 — News query")
    print("=" * 60)
    resp = await chat(USER_ID, SID, "Give me 3 recent headlines about AI agents.")
    print("[RESPONSE]", resp)


if __name__ == "__main__":
    asyncio.run(main())


# ─── Key concepts recap ──────────────────────────────────────────────────────
"""
─── KEY CONCEPTS ────────────────────────────────────────────────────
• Agent(tools=[fn])     — any typed Python function becomes a tool automatically
• AgentTool(agent=x)    — expose a sub-agent as a tool (A2A protocol)
• InMemorySessionService— per-user session state; swap for VertexAI in prod
• Runner.run_async()    — async event stream; listen for is_final_response()
• session.state         — shared dict readable/writable by the agent
• google_search         — built-in ADK tool for live Google Search
• model="gemini-2.0-flash-exp" — latest fast Gemini model
─────────────────────────────────────────────────────────────────────
"""
