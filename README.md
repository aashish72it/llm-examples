# Snowflake RAG Framework

This project implements a Retrieval-Augmented Generation (RAG) pipeline with evaluation capabilities using **LangChain**, **ChromaDB**, **Groq LLM**, and **MLflow**.  
It provides a Streamlit UI for interactive Q&A and supports ingestion of custom documents into the knowledge base.

---

## 🚀 Features
- Document ingestion into ChromaDB vector store
- Query answering with Groq LLM
- Evaluation metrics (faithfulness, relevancy, precision, recall) via MLflow
- Streamlit app for interactive Q&A
- Configurable modes: `eval` (with evaluation) or `prod` (answer only)

---

## ⚙️ Setup Instructions

### 1. Create `.env` file in project root directory
Add the following environment variables:

```env
# Get your API key from https://console.groq.com
GROQ_API_KEY=<groq-token>
LLM_MODEL=llama-3.1-8b-instant
MAX_TOKENS_HISTORY=4000

UPLOAD_DIR=<upload-dir>
SESSION_LOG=<session-log-file>
CHROMA_DIR=<chromadb-dir>
SOURCE_DIR=<source-dir>
STORAGE_DIR=<storage-dir>

MLFLOW_EXPERIMENT=snowflake_rag_workflow
MLFLOW_TRACKING_URI=snowflake_mlruns

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
STREAMLIT_APP="Snowflake Q&A with Evaluation"
TOP_K=5
SYSTEM_PROMPT="You are a helpful Q&A assistant to answer the question related to snowflake datawarehouse.
Use the provided context to answer the user's question. If there is no context or the question
is not related to snowflake warehouse just reply `Sorry, I cannot answer this as it is not a relevant question.`"
FEEDBACK_DIR=<feedback-dir>

# Set to "eval" during development, "prod" for production
APP_MODE=prod

# set to cpu if working on local machine
EMBEDDING_DEVICE=cpu
```

### 2. Create virtual environment & install dependencies
Create a Python virtual environment and install the required packages. Run the below from commands from **project root directory**

```bash
python -m venv .venv
source .venv/bin/activate   # On Linux/Mac
.venv\Scripts\activate      # On Windows

pip install -r rag_framework/requirements.txt
```

### 3. Test the ingestion to make sure source file(s) ingested as expected in vector database.

```bash
cd rag_framework
python -m rag_framework.tests.ingest
```

### 4. Test the evaluation. 
Check all the important metrics like faithfulness, answer_relevancy, context_precision, context_recall.
Make sure the APP_MODE is set to eval to test the evaluation in chatbot. It will be off for **APP_MODE = prod**

```bash
python -m rag_framework.app
```

### 5. Run the streamlit app

```bash
cd ..

$env:PYTHONPATH = "<root-dir-of-this-project>"
streamlit run rag_framework/streamlit_app.py
```

### Project Structure

<pre>
rag_framework/
├── __init__.py
├── app.py                        # CLI evaluation runner
├── streamlit_app.py              # Streamlit UI
├── feedback_logger.py            # Logs human & mlflow feedback
├── config.py                     # Loads env, validates, defaults
├── core/
│   ├── embeddings.py             # Embedder abstraction (SentenceTransformers)
│   ├── vector_store.py           # ChromaDB wrapper
│   ├── retriever.py              # Retrieval and reranking
│   ├── llm_adapter.py            # Adapter to expose LLMClient as LangChain-compatible
│   ├── chain.py                  # RAGChain orchestrating retrieval + answer
│   ├── llm.py                    # Groq LLM client
│   └── evaluator.py              # Ragas + MLflow evaluation
├── ingest/
│   ├── document_loader.py        # PDF loader -> chunks text
│   └── pipeline.py               # Ingestor: embeds + stores chunks
├── utils/
│   └── errors.py                 # Exceptions
├── tests/
│   └── ingest.py                 # Smoke test: ingestion + query
└── requirements.txt
<pre>
