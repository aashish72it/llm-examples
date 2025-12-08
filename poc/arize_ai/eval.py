import os
from groq import Groq
from dotenv import load_dotenv
from tqdm import tqdm
import json

###Arize Phoenix imports
import phoenix as px
from phoenix.evals import (
    TOOL_CALLING_PROMPT_TEMPLATE, 
    llm_classify,
    OpenAIModel
)
from phoenix.trace import SpanEvaluations
from openinference.instrumentation import suppress_tracing
from phoenix.otel import register
from openinference.instrumentation.groq import GroqInstrumentor
from opentelemetry.trace import StatusCode
from phoenix.client.types.spans import SpanQuery
from phoenix.client import Client


import logging, warnings
warnings.filterwarnings('ignore')


import nest_asyncio
nest_asyncio.apply()

from arize_ai_agent import run_agent, start_main_span, tools, get_phoenix_endpoint

import pandas as pd

###############environment variables#################
load_dotenv()
TRANSACTION_DATA_FILE_PATH = os.getenv("TRANSACTION_DATA_FILE_PATH")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)
MODEL = os.getenv("LLM_MODEL")
PROJECT_NAME = os.getenv("PROJECT_NAME")

agent_questions = [
    "What was the most popular product SKU?",
    "What was the total revenue across all stores?",
    "Which store had the highest sales volume?",
    "Create a bar chart showing total sales by store",
    "What percentage of items were sold on promotion?",
    "What was the average transaction value?"
    "Show me the code for graph of sales by store in Nov 2021 and tell me what trends you see?"
]


client = Client()

for question in tqdm(agent_questions, desc="Processing questions"):
    try:
        ret = start_main_span([{"role": "user", "content": question}])
    except Exception as e:
        print(f"Error processing question: {question}")
        print(e)
        continue

print("TOOL_CALLING_PROMPT_TEMPLATE: ", TOOL_CALLING_PROMPT_TEMPLATE)

query = SpanQuery().where(
    # Filter for the `LLM` span kind.
    # The filter condition is a string of valid Python boolean expression.
    "span_kind == 'LLM'",
).select(
    question="input.value",
    tool_call="llm.tools"
)

# ------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------
def df_to_annotations(
    df: pd.DataFrame,
    *,
    annotation_name: str,
    span_id_col: str = "span_id",
    label_col: str | None = "label",
    score_col: str | None = "score",
    explanation_col: str | None = "explanation",
    metadata_col: str | None = None,
    annotator_kind: str = "LLM",  # "LLM" | "CODE" | "HUMAN"
) -> list[dict]:
    """
    Build payload for client.spans.log_span_annotations(...).

    Phoenix requires at least one of {label, score, explanation} inside "result".
    """
    annotations: list[dict] = []
    for _, r in df.iterrows():
        result: dict = {}
        if score_col and score_col in df.columns and pd.notna(r.get(score_col)):
            result["score"] = float(r[score_col])
        if label_col and label_col in df.columns and pd.notna(r.get(label_col)):
            result["label"] = str(r[label_col])
        if explanation_col and explanation_col in df.columns and pd.notna(r.get(explanation_col)):
            result["explanation"] = str(r[explanation_col])

        payload: dict = {
            "span_id": r[span_id_col],
            "name": annotation_name,
            "annotator_kind": annotator_kind,
            "result": result or {"explanation": "no label/score provided"},
        }
        if metadata_col and metadata_col in df.columns and pd.notna(r.get(metadata_col)):
            payload["metadata"] = r[metadata_col]
        annotations.append(payload)
    return annotations


def code_is_runnable(output: str) -> bool:
    """
    Checks if generated Python code is runnable.
    NOTE: This uses exec() — sandbox appropriately in production.
    """
    output = (output or "").strip().replace("```python", "").replace("```", "")
    try:
        exec(output, {})  # run in isolated globals; avoid side effects in production
        return True
    except Exception:
        return False


# ------------------------------------------------------------------------------------
# 1) Tool Calling Eval
# ------------------------------------------------------------------------------------
# Query LLM spans; select user question + tool calls
tool_query = SpanQuery().where("span_kind == 'LLM'").select(
    question="input.value",
    tool_call="llm.tools",
)

tool_calls_df = client.spans.get_spans_dataframe(
    query=tool_query, project_identifier=PROJECT_NAME
).dropna(subset=["tool_call"])


# Classify with LLM-as-judge
tool_call_eval = llm_classify(
    dataframe=tool_calls_df,
    template=TOOL_CALLING_PROMPT_TEMPLATE.replace("{tool_definitions}", json.dumps(tools)),
    rails=["correct", "incorrect"],
    model=MODEL,
    provide_explanation=True,
)
tool_call_eval["score"] = (tool_call_eval["label"] == "correct").astype(int)

# Log as span annotations
client.spans.log_span_annotations(
    annotations=df_to_annotations(
        tool_call_eval,
        annotation_name="Tool Calling Eval",
        span_id_col="span_id",
        label_col="label",
        score_col="score",
        explanation_col="explanation",
        annotator_kind="LLM",
    ),
    project_identifier=PROJECT_NAME,
    sync=False,
)

print("✅ Tool Calling Eval logged:", len(tool_call_eval))


# ------------------------------------------------------------------------------------
# 2) Runnable Code Eval (from generate_visualization tool)
# ------------------------------------------------------------------------------------
code_query = SpanQuery().where("name == 'generate_visualization'").select(
    generated_code="output.value"
)

code_gen_df = client.spans.get_spans_dataframe(
    query=code_query, project_identifier=PROJECT_NAME
)
code_gen_df["label"] = code_gen_df["generated_code"].apply(code_is_runnable).map({True: "runnable", False: "not_runnable"})
code_gen_df["score"] = code_gen_df["label"].map({"runnable": 1, "not_runnable": 0})
code_gen_df["explanation"] = ""  # optional free-text

client.spans.log_span_annotations(
    annotations=df_to_annotations(
        code_gen_df,
        annotation_name="Runnable Code Eval",
        span_id_col="span_id",
        label_col="label",
        score_col="score",
        explanation_col="explanation",
        annotator_kind="CODE",
    ),
    project_identifier=PROJECT_NAME,
    sync=False,
)

print("✅ Runnable Code Eval logged:", len(code_gen_df))


# ------------------------------------------------------------------------------------
# 3) Response Clarity Eval (AGENT spans)
# ------------------------------------------------------------------------------------
clarity_prompt = """
Your response must be single word, either "clear" or "unclear",
and should not contain any other text beyond that word.

- "clear" = the answer is precise, coherent, and directly addresses the query.
- "unclear" = the answer is vague, disorganized, or difficult to understand.

[BEGIN DATA]
Query: {query}
Answer: {response}
[END DATA]
"""

clarity_query = SpanQuery().where("span_kind == 'AGENT'").select(
    response="output.value",
    query="input.value",
)

clarity_df = client.spans.get_spans_dataframe(
    query=clarity_query, project_identifier=PROJECT_NAME
)

clarity_eval = llm_classify(
    dataframe=clarity_df,
    template=clarity_prompt,
    rails=["clear", "unclear"],
    model=MODEL,
    provide_explanation=True,
)
clarity_eval["score"] = (clarity_eval["label"] == "clear").astype(int)

client.spans.log_span_annotations(
    annotations=df_to_annotations(
        clarity_eval,
        annotation_name="Response Clarity",
        span_id_col="span_id",
        label_col="label",
        score_col="score",
        explanation_col="explanation",
        annotator_kind="LLM",
    ),
    project_identifier=PROJECT_NAME,
    sync=False,
)

print("✅ Response Clarity Eval logged:", len(clarity_eval))


# ------------------------------------------------------------------------------------
# 4) SQL Generation Eval (LLM spans filtered by your SQL-gen prompt)
# ------------------------------------------------------------------------------------
SQL_EVAL_GEN_PROMPT = """
Your response must be single word, either "correct" or "incorrect".
Assume the database exists and columns are appropriately named.
Use both the instruction and the query text to decide.

- "correct" = the SQL correctly solves the instruction.
- "incorrect" = it does not.

[BEGIN DATA]
Instruction: {question}
Reference Query: {query_gen}
[END DATA]
"""

sql_query = SpanQuery().where("span_kind == 'LLM'").select(
    query_gen="llm.output_messages",
    question="input.value",
)

sql_df = client.spans.get_spans_dataframe(
    query=sql_query, project_identifier=PROJECT_NAME
)

# Keep only the SQL-generation instructions (adapt the filter to your exact wording)
mask = sql_df["question"].str.contains("Generate an SQL query based on a prompt.", na=False)
sql_df = sql_df[mask].copy()

sql_gen_eval = llm_classify(
    dataframe=sql_df,
    template=SQL_EVAL_GEN_PROMPT,
    rails=["correct", "incorrect"],
    model=MODEL,
    provide_explanation=True,
)
sql_gen_eval["score"] = (sql_gen_eval["label"] == "correct").astype(int)

client.spans.log_span_annotations(
    annotations=df_to_annotations(
        sql_gen_eval,
        annotation_name="SQL Gen Eval",
        span_id_col="span_id",
        label_col="label",
        score_col="score",
        explanation_col="explanation",
        annotator_kind="LLM",
    ),
    project_identifier=PROJECT_NAME,
    sync=False,
)

print("✅ SQL Gen Eval logged:", len(sql_gen_eval))


print("\nAll evals completed.")
