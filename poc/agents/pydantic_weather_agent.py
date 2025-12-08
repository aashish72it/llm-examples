import os
from langsmith import unit
import requests
from dotenv import load_dotenv

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.settings import ModelSettings

class WeatherRequest(BaseModel):
    city: str
    desc: str = "Get the current weather for the specified city."
    temp_celsius: float = None


load_dotenv()

weather_agent = Agent(
    model = os.getenv("WEATHER_LLM_MODEL"),
    model_settings = ModelSettings(temperature=0.2),
    output_type=str,
    system_prompt=os.getenv("SYSTEM_WEATHER_PROMPT")
)


weather_agent.tool
def get_weather_tool(ctx: RunContext, city: str) -> WeatherRequest:
    """Tool to get the current weather for a specified city."""
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
    try:
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

        #return f"The current weather in {city} is {weather_desc} with a temperature of {temp}°C."
        return WeatherRequest(
            city=city,
            desc=weather_desc,
            temp_celsius=temp
        )
    except (ValueError, KeyError) as e:
        return f"Error processing weather data for {city}: {e}"
    
city = input("Enter city name for weather information: ")
weather=weather_agent.run_sync(city)
print(f"Weather Agent Response: {weather.output}")