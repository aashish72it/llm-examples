import os
from groq import Groq
from dotenv import load_dotenv

import json
import duckdb
import pandas as pd
from pydantic import BaseModel, Field

###Arize Phoenix imports
import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.groq import GroqInstrumentor
from opentelemetry.trace import StatusCode
import logging, warnings


warnings.filterwarnings('ignore')

###############environment variables#################
load_dotenv()
TRANSACTION_DATA_FILE_PATH = os.getenv("TRANSACTION_DATA_FILE_PATH")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)
MODEL = os.getenv("LLM_MODEL")
PHOENIX_API_KEY=os.getenv("PHOENIX_API_KEY")
PROJECT_NAME=os.getenv("PROJECT_NAME")
PHOENIX_COLLECTOR_ENDPOINT=os.getenv("PHOENIX_COLLECTOR_ENDPOINT","https://localhost:6006/")
PHOENIX_WORKING_DIR=os.getenv("PHOENIX_WORKING_DIR")
PHOENIX_SQL_DATABASE_URL=os.getenv("PHOENIX_SQL_DATABASE_URL")
####################observability init##############


for name in ("phoenix", "phoenix.otel", "openinference", "opentelemetry", "opentelemetry.sdk", "opentelemetry.exporter"):
    logging.getLogger(name).setLevel(logging.ERROR)

tracer_provider = register(
  project_name=PROJECT_NAME,
  auto_instrument=True)

tracer = tracer_provider.get_tracer(__name__)

session = px.launch_app(use_temp_dir=False)

#GroqInstrumentor().instrument(tracer_provider=tracer_provider)

####################Prompts##########################
# prompt template for tool 1
SQL_GENERATION_PROMPT = """
Generate an SQL query based on a prompt. Do not reply with anything besides the SQL query.
The prompt is: {prompt}

The available columns are: {columns}
The table name is: {table_name}
"""

# prompt template for data analysis tool
DATA_ANALYSIS_PROMPT = """
Analyze the following data: {data}
Your job is to answer the following question: {prompt}
"""


# prompt template for step 1 of tool 3
CHART_CONFIGURATION_PROMPT = """
Generate a chart configuration based on this data: {data}
The goal is to show: {visualization_goal}
"""

# prompt template for step 2 of tool 3

CREATE_CHART_PROMPT = """
Return ONLY valid Python code (no markdown fences, no explanations, no comments).
Requirements:
- Use pandas and matplotlib.pyplot as plt (import both if not already imported).
- Read the tab-separated 'data' string via io.StringIO.
- Use the provided config dict keys: title, x_axis, y_axis, chart_type.
- Do not print; just build the chart and call plt.show().

data (TSV): {config[data]}
config: {config}
"""

SYSTEM_PROMPT = """

You are a helpful assistant for Store Sales Price Elasticity Promotions data.

Tool-use rules:
- Use the built-in tool calling mechanism only (structured tool calls), not custom text or markup.
- Never write function calls, XML tags, or any tool invocation inside the assistant message content.
- Only call tools that are available in the current request.
- Call one tool at a time; after receiving a tool result (role="tool"), use that string/result in the next step.
- All tool arguments must be valid JSON that matches the declared schema.
"""


def get_phoenix_endpoint():
    return PHOENIX_COLLECTOR_ENDPOINT


########################Tools##########################

# code for tool 1
@tracer.tool()
def lookup_sales_data(prompt: str) -> str:
    """Implementation of sales data lookup from parquet file using SQL"""
    try:

        # define the table name
        table_name = "sales"
        
        # step 1: read the parquet file into a DuckDB table
        df = pd.read_parquet(TRANSACTION_DATA_FILE_PATH)
        duckdb.sql(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM df")

        # step 2: generate the SQL code
        sql_query = generate_sql_query(prompt, df.columns, table_name)
        # clean the response to make sure it only includes the SQL code
        sql_query = sql_query.strip()
        sql_query = sql_query.replace("```sql", "").replace("```", "")
        
        # step 3: execute the SQL query
        result = duckdb.sql(sql_query).df()
        
                
        MAX_ROWS = 500
        if len(result) > MAX_ROWS:
            result_sample = result.head(MAX_ROWS)
        else:
            result_sample = result
            
        return result_sample.to_csv(sep="\t", index=False)
        #return result.to_string()
    
    except Exception as e:
        return f"Error accessing data: {str(e)}"
    

# code for step 2 of tool 1
def generate_sql_query(prompt: str, columns: list, table_name: str) -> str:
    """Generate an SQL query based on a prompt"""
    formatted_prompt = SQL_GENERATION_PROMPT.format(prompt=prompt, 
                                                    columns=columns, 
                                                    table_name=table_name)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": formatted_prompt}],
        tool_choice="none"
    )
    
    return response.choices[0].message.content


# code for tool 2
@tracer.tool()
def analyze_sales_data(prompt: str, data: str) -> str:
    """Implementation of AI-powered sales data analysis"""
    formatted_prompt = DATA_ANALYSIS_PROMPT.format(data=data, prompt=prompt)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": formatted_prompt}],
        tool_choice="none"
    )
    
    analysis = response.choices[0].message.content
    return analysis if analysis else "No analysis could be generated"


# class defining the response format of step 1 of tool 3
class VisualizationConfig(BaseModel):
    chart_type: str = Field(..., description="Type of chart to generate (line|bar)")
    x_axis: str = Field(..., description="Name of the x-axis column (e.g., 'Sold_Date', 'SKU_Coded')")
    y_axis: str = Field(..., description="Name of the y-axis column (e.g., 'Total_Sale_Value', 'Qty_Sold')")
    title: str = Field(..., description="Title of the chart")


# code for step 1 of tool 3
@tracer.chain()
def extract_chart_config(data: str, visualization_goal: str) -> dict:
    """Generate chart visualization configuration
    
    Args:
        data: String containing the data to visualize
        visualization_goal: Description of what the visualization should show
        
    Returns:
        Dictionary containing line chart configuration
    """
    formatted_prompt = CHART_CONFIGURATION_PROMPT.format(data=data,
                                visualization_goal=visualization_goal)
    

    # Ask the model to return a JSON object with required keys
    json_instruction = (
        "Return a JSON object with keys: chart_type, x_axis, y_axis, title."
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": f"{formatted_prompt}\n{json_instruction}"}],
        # JSON Object Mode works even when strict schema mode isn't available
        response_format={"type": "json_object"},
        temperature=0.0,
        tool_choice="none"
    )
    
    
    raw = response.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw)
        return {
            "chart_type": parsed.get("chart_type", "line"),
            "x_axis": parsed.get("x_axis", "date"),
            "y_axis": parsed.get("y_axis", "value"),
            "title": parsed.get("title", visualization_goal),
            "data": data,
        }
    except Exception:
        return {
            "chart_type": "line",
            "x_axis": "date",
            "y_axis": "value",
            "title": visualization_goal,
            "data": data,
        }


# code for step 2 of tool 3
@tracer.chain()
def create_chart(config: dict) -> str:
    """Create a chart based on the configuration"""
    formatted_prompt = CREATE_CHART_PROMPT.format(config=config)
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": formatted_prompt}],
        tool_choice="none"
    )
    
    code = response.choices[0].message.content
    code = code.replace("```python", "").replace("```", "")
    code = code.strip()
    
    return code

# code for tool 3
@tracer.tool()
def generate_visualization(data: str, visualization_goal: str) -> str:
    """Generate a visualization based on the data and goal"""
    config = extract_chart_config(data, visualization_goal)
    code = create_chart(config)
    return code


######################Schema##########################

# Define tools/functions that can be called by the model
tools = [
    {
        "type": "function",
        "function": {
            "name": "lookup_sales_data",
            "description": "Look up data from Store Sales Price Elasticity Promotions dataset",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "The unchanged prompt that the user provided."}
                },
                "required": ["prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_sales_data", 
            "description": "Analyze sales data to extract insights",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {"type": "string", "description": "The lookup_sales_data tool's output."},
                    "prompt": {"type": "string", "description": "The unchanged prompt that the user provided."}
                },
                "required": ["data", "prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_visualization",
            "description": "Generate Python code to create data visualizations",
            "parameters": {
                "type": "object", 
                "properties": {
                    "data": {"type": "string", "description": "The lookup_sales_data tool's output."},
                    "visualization_goal": {"type": "string", "description": "The goal of the visualization."}
                },
                "required": ["data", "visualization_goal"]
            }
        }
    }
]

# Dictionary mapping function names to their implementations
tool_implementations = {
    "lookup_sales_data": lookup_sales_data,
    "analyze_sales_data": analyze_sales_data, 
    "generate_visualization": generate_visualization
}


# code for executing the tools returned in the model's response
@tracer.chain()
def handle_tool_calls(tool_calls, messages):
    
    for tool_call in tool_calls:
        name = tool_call.function.name
        try: 
            args = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        function = tool_implementations.get(name)

        if not function:
            result = {"error": f"Tool {name} not implemented"}
        else:
            try:
                result = function(**args)
            except Exception as e:
                result = {"error": f"Execution error in {name}: {e}"}

        messages.append({"role": "tool", "tool_call_id": tool_call.id,
                         "name": name, "content": json.dumps(result)})
    return messages


def run_agent(messages):
    print("Running agent with messages:", messages)

    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
        
    # Check and add system prompt if needed
    if not any(
            isinstance(message, dict) and message.get("role") == "system" for message in messages
        ):
            system_prompt = {"role": "system", "content": SYSTEM_PROMPT}
            messages.insert(0,system_prompt)

    while True:
        print("Starting router call span")
        with tracer.start_as_current_span(
            "Router_Chain", openinference_span_kind="chain",
        ) as span:
            span.set_input(value=messages)

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        ###adding message, role & tools                
        assistant_msg = response.choices[0].message
        messages.append({
            "role": "assistant",
            "content": assistant_msg.content or "",
            "tool_calls": assistant_msg.tool_calls or []
        })

        tool_calls = response.choices[0].message.tool_calls
        print("Received response with tool calls:", bool(tool_calls))
        span.set_status(StatusCode.OK)

        # if the model decides to call function(s), call handle_tool_calls
        if tool_calls:
            print("Starting tool calls span")
            messages = handle_tool_calls(tool_calls, messages)
            span.set_output(value=tool_calls)
        else:
            print("No tool calls, returning final response")
            span.set_output(value=response.choices[0].message.content)
            return response.choices[0].message.content
        

def start_main_span(messages):
    print("Starting main span with messages:", messages)
    
    with tracer.start_as_current_span(
        "Agent_Trace", openinference_span_kind="agent"
    ) as span:
        span.set_input(value=messages)
        ret = run_agent(messages)
        print("Main span completed with return value:", ret)
        span.set_output(value=ret)
        span.set_status(StatusCode.OK)
        return ret

def prepare_chart_code(code: str) -> str:
    code = code.replace("```python", "").replace("```", "").strip()
    # Prepend imports if the code uses StringIO or pandas/plt
    import_lines = []
    if "StringIO(" in code or ".read_csv(" in code:
        import_lines.append("import io")
    if "pd." in code:
        import_lines.append("import pandas as pd")
    if "plt." in code:
        import_lines.append("import matplotlib.pyplot as plt")
    # Ensure imports exist
    for line in import_lines:
        if line not in code:
            code = line + "\n" + code
    return code


#########################Main#########################

# if __name__ == "__main__":
#     # example usage of tool 1

#     result = start_main_span([{"role": "user", 
#                            "content": "Which stores did the best in 2021?"}])
#     print(result)
    
#     result = start_main_span([{"role": "user", 
#                                "content": "Show me the code for graph of sales by store in Nov 2021,"
#                                 "and tell me what trends you see?"}])


    #print("="*80)
    #print("step - 1: Generate Sample data")
    #print("="*80)
    #example_data = lookup_sales_data("Show me all the sales for store 1320 on November 1st, 2021")
    # print(example_data)

    # print("="*80)
    # print("step - 2: Generate Analysis")
    # print("="*80)
    # # example usage of tool 2
    # print(analyze_sales_data(prompt="what trends do you see in this data",
    #                      data=example_data))
    
    # print("="*80)
    # print("step - 3: Generate Visualization")
    # print("="*80)
    # code = generate_visualization(example_data, 
    # "A bar chart of sales by product SKU. Put the product SKU on the x-axis and the sales on the y-axis.")
    # #print(code)
    # code = prepare_chart_code(code)
    # #print("-----------------Sanitized code------------------")
    # #print(code)
    # exec(code)

    # print("="*80)
    # print("step - 4: Run Agent")
    # print("="*80)
    # result = run_agent('Show me the code for graph of sales by store in Nov 2021, and tell me what trends you see.')
    # print(result)
