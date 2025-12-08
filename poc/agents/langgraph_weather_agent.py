import os
import requests
from typing import TypedDict, List

from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition


# ---------------------------------------------------------------------
# Environment Setup
# ---------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------
# Tool Setup
# ---------------------------------------------------------------------
@tool
def get_weather_tool(city: str) -> dict:
    """Tool: Get current weather for a specified city (OpenWeather Current Weather API)."""
    api_key = os.getenv("WEATHER_API_KEY")
    weather_url = os.getenv("WEATHER_URL")

    if not api_key:
        return {"error": "WEATHER_API_KEY not found in environment variables."}
    if not weather_url.startswith("https://"):
        return {"error": "WEATHER_URL not found or invalid in environment variables."}

    params = {
        "q": city,
        "appid": api_key,
        "units": "metric",
        "lang": "en",
    }

    try:
        resp = requests.get(weather_url, params=params, timeout=15)
    except requests.RequestException as e:
        return {"error": f"Network error while fetching weather for {city}: {e}"}

    if resp.status_code != 200:
        # Surfacing OpenWeather error payload helps debugging
        try:
            err = resp.json()
        except Exception:
            err = {"message": resp.text[:200]}
        return {
            "error": "OpenWeather non-200",
            "code": resp.status_code,
            "details": err
        }

    try:
        data = resp.json()
        name = data.get("name", city)
        desc = data["weather"][0]["description"]
        temp_c = data["main"]["temp"]
        return {
            "city": name,
            "desc": desc,
            "temp_celsius": temp_c
        }
    except (KeyError, IndexError, TypeError, ValueError) as e:
        return {"error": f"Error processing weather data for {city}: {e}"}


# ---------------------------------------------------------------------
# LLM Setup
# ---------------------------------------------------------------------
model_name = os.getenv("LLM_MODEL")
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found. Set it in your environment or .env file.")

llm = ChatGroq(
    api_key=groq_api_key,
    model=model_name,
    temperature=0.2,
)

# Bind tools so the model can emit tool-calls
llm_with_tools = llm.bind_tools([get_weather_tool])

# ---------------------------------------------------------------------
# LangGraph state & nodes
# ---------------------------------------------------------------------
class GraphState(TypedDict):
    messages: List[BaseMessage]

def llm_node(state: GraphState) -> GraphState:
    """Calls the LLM; it may either answer or request a tool call."""
    result = llm_with_tools.invoke(state["messages"])
    return {"messages": state["messages"] + [result]}

tool_node = ToolNode([get_weather_tool])

# ---------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------
builder = StateGraph(GraphState)

builder.add_node("llm", llm_node)
builder.add_node("tools", tool_node)

# If LLM output contains tool-calls -> go to tools; otherwise end.
builder.add_conditional_edges(
    "llm",
    tools_condition,
    {
        "tools": "tools",
        "end": END,
    }
)

builder.add_edge(START, "llm")

graph = builder.compile()

# ---------------------------------------------------------------------
# CLI runner (add system prompt; ask user for city)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    sys_prompt = os.getenv("SYSTEM_WEATHER_PROMPT")
    city = input("Enter city name for weather information: ")

    initial_messages = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=f"Get the current weather for {city}."),
    ]

    result = graph.invoke({"messages": initial_messages})
    final_msg = result["messages"][-1]
    print("\nWeather Agent Response:")
    print(final_msg.content)
