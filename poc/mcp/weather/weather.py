from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP

import os
from dotenv import load_dotenv
import requests

load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("weather")

API_KEY = os.getenv("WEATHER_API_KEY")
WEATHER_URL = os.getenv("WEATHER_URL")

def _get_weather(city: str) -> dict:
    """Tool to get the current weather for a specified city."""
    try:
        if not API_KEY:
            raise ValueError("WEATHER_API_KEY not found in environment variables.")
        if not WEATHER_URL.startswith("https://"):
            raise ValueError("WEATHER_URL not found or invalid in environment variables.")

        params = {
            "q": city,         # Tip: include country code, e.g., "Toronto,CA"
            "appid": API_KEY,
            "units": "metric",
        }
        response = requests.get(WEATHER_URL, params=params, timeout=15)

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

@mcp.tool()
def weather_search(city: str) -> dict:
    return _get_weather(city)

if __name__ == "__main__":
    mcp.run(transport="stdio")
