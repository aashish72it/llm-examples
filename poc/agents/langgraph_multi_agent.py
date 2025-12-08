import os
import requests
from dotenv import load_dotenv
from typing import TypedDict, List

from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage, ToolMessage

from langchain_core.tools import tool
from langchain_groq import ChatGroq

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition



load_dotenv()



def _get_weather(city: str) -> dict:
    """Tool to get the current weather for a specified city."""
    try:
        api_key = os.getenv("WEATHER_API_KEY")
        weather_url = os.getenv("WEATHER_URL")
        units = "metric"
        if not api_key:
            raise ValueError("WEATHER_API_KEY not found in environment variables.")
        if not weather_url.startswith("https://"):
            raise ValueError("WEATHER_URL not found or invalid in environment variables.")


        params = {
            "q": city,         # Tip: include country code, e.g., "Toronto,CA"
            "appid": api_key,
            "units": units,
        }
        response = requests.get(weather_url, params=params, timeout=15)

    except requests.RequestException as e:
        return f"Network error while fetching weather for {city}: {e}"

    if response.status_code != 200:
        return f"Could not retrieve weather data for {city}. " \
               f"HTTP {response.status_code} body={response.text[:200]}"

    try:
        data = response.json()
        city = data['name']
        weather_desc = data['weather'][0]['description']
        temp = data['main']['temp']

        return {"city": city, "description": weather_desc, "temperature": temp}

    except (ValueError, KeyError) as e:
        return f"Error processing weather data for {city}: {e}"

def _serp_search(query: str) -> dict:
    """Search the web using SerpAPI (Google Search)."""
    try:
        api_key = os.getenv("SERP_API_KEY")
        search_engine = os.getenv("SERPAPI_SEARCH_ENGINE", "google")
        if not api_key:
            return {"error": "SERP_API_KEY not found in environment variables."}

        url = "https://serpapi.com/search"
        params = {
            "q": query,
            "api_key": api_key,
            "engine": search_engine,
            "num": 2
        }

        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()

        organic = []
        for item in data.get("organic_results", [])[:5]:
            organic.append({
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet"),
            })

        if resp.status_code != 200:
            return {"error": f"Search failed: {resp.status_code}", "details": resp.text}
    except requests.RequestException as e:
        return {"error": f"Network error during search: {e}"}
    
    
    return {"query": query, "results": organic}
    

@tool
def web_search(query: str) -> dict:
    """Tool: Perform a web search using SerpAPI."""
    return _serp_search(query)


@tool(description="Get current weather for a specified city.",name_or_callable="weather_search")
def weather_search(query: str) -> dict:
    """Tool: Perform a weather search using the weather API."""
    return _get_weather(query)

# LLM Setup & binding both tools to LLM

model_name = os.getenv("LLM_MODEL")
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found. Set it in your environment or .env file.")

llm = ChatGroq(
    api_key=groq_api_key,
    model=model_name,
    temperature=0.2,
)

llm_with_tools = llm.bind_tools([web_search, weather_search])

# ---------------------------------------------------------------------
# LangGraph state & nodes
# ---------------------------------------------------------------------
class GraphState(TypedDict):
    messages: List[BaseMessage]

def llm_node(state: GraphState) -> GraphState:
    """Calls the LLM; it may either answer or request a tool call."""
    result = llm_with_tools.invoke(state["messages"])
    return {"messages": state["messages"] + [result]}

tool_node = ToolNode([web_search, weather_search])


## Build the graph (entrypoint + routes)

builder = StateGraph(GraphState)
builder.add_node("llm", llm_node)
builder.add_node("tools", tool_node)

builder.add_conditional_edges(
    "llm",
    tools_condition,
    {"tools": "tools", "end": END},
)

builder.add_edge(START, "llm")

graph = builder.compile()


# CLI Runner

if __name__ == "__main__":
###    sys_prompt = os.getenv("SYSTEM_MULTI_AGENT_PROMPT") or (
###        "You are a helpful data & weather assistant.\n"
###        "- When asked complex questions, plan steps explicitly.\n"
###        "- Use `web_search` to gather facts. Use multiple searches if needed.\n"
###        "- After you know the exact city, call `weather_search(city)` to get temperature in °C.\n"
###        "- In the final answer, clearly state the city, weather description and the temperature.\n"
###        "- If sources are available from web_search, include the top link."
###        "- After completing the task, summarize the steps you took:"
###        "    1) Which province you found"
###        "    2) Which city you found"
###        "    3) Weather details"
###        "- Include the reasoning and one source link in the final answer."
###    )

    sys_prompt = os.getenv("SYSTEM_MULTI_AGENT_PROMPT") or (           
        "You are a data & weather assistant."
        "Required steps:"
        "1) When a question is asked then try to break it in chains."
        "2) Use `web_search` to find the correct city for the query "
        "for ex: largest city in Quebec. (Answer: Montreal)"
        "or where the city is not directly mentioned like capital of France (Answer: Paris)."
        "3) and then in the end MUST call `weather_search(city)` for the city found in step 2 "
        "to get the temperature in °C."
        "Final response: return ONLY a JSON object:"
    )

    user_query = input("Enter your query: ")

    initial_messages = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=user_query),
    ]

    # Allow multiple tool/LLM turns
    result = graph.invoke({"messages": initial_messages}, config={"recursion_limit": 8})

    final_msg = result["messages"][-1]
    print("Assistant:", final_msg.content)

###Prompt1: Give me the weather detail for largest city in the largest province of Canada?
###Prompt2: Give me the weather detail for capital of France?
###Prompt3: Give me the weather detail for the city where first president of America was born?
###Prompt4: Give me the weather detail for the biggest city in Australia?
