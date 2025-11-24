# MultiModal RAG Framework

This project implements a Retrieval-Augmented Generation (RAG) pipeline with evaluation capabilities using **LangChain**, **ChromaDB**, **Groq LLM**, and **MLflow**.  
It provides a Streamlit UI for interactive Q&A and supports ingestion of custom documents into the knowledge base.

---

## 🚀 Features
- Ingesting documents, images, videos & audios into ChromaDB vector store
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
MAX_TOKENS_HISTORY = 4000
UPLOAD_DIR=".\\uploads"
SESSION_LOG=".\\session_log.txt"
CHROMA_DIR=".\\chromadb"
STORAGE_DIR=".\\store_input_docs"
MLFLOW_EXPERIMENT=multimodal_finance_assistant
MLFLOW_TRACKING_URI=sqlite:///multimodal_mlruns.db
STREAMLIT_APP="Multimodal Finance Assistant"
TOP_K=3
APP_MODE=prod
EMBEDDING_DEVICE=cpu
LOG_DIR=".\\logs"
# --- Multimodal Embeddings ---
TEXT_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
IMAGE_EMBEDDING_MODEL=openai/clip-vit-base-patch32
AUDIO_TRANSCRIBE_MODEL=base  # tiny/base: faster but less accurate, small/medium/large: more accurate but slower
VIDEO_EMBEDDING_MODEL=openai/clip-vit-base-patch32

# --- Modalities Storage ---
TEXT_DIR=".\\text"
AUDIO_DIR=".\\audio"
IMAGE_DIR=".\\images"
VIDEO_DIR=".\\videos"
FEEDBACK_DIR=".\\feedback"
STORAGE_DIR=".\\store_input_docs"
# --- Vector DB Collections ---
TEXT_COLLECTION=text_docs
IMAGE_COLLECTION=image_docs
AUDIO_COLLECTION=audio_docs
VIDEO_COLLECTION=video_docs

# --- Retrieval Settings ---
TOP_K_TEXT=5
TOP_K_IMAGE=3
TOP_K_AUDIO=3
TOP_K_VIDEO=3

# --- System Prompts ---
SYSTEM_PROMPT_TEXT="You are a helpful assistant for text queries."
SYSTEM_PROMPT_MULTIMODAL="You are a multimodal finance assistant. Use the provided text, audio transcripts, image captions, and video frame descriptions to answer the user's question. If the answer is not in the context, say you don't know."
```

### 2. Create virtual environment & install dependencies
Create a Python virtual environment and install the required packages. Run the below from commands from **project root directory**

```bash
python -m venv .venv
source .venv/bin/activate   # On Linux/Mac
.venv\Scripts\activate      # On Windows

pip install -r multimodal/requirements.txt
```

### 3. Test the ingestion to make sure text file(s) ingested as expected in vector database.

```bash
python -m multimodal.tests.ingest
```

### 4. Test the ingestion to make sure audio, video file(s) & images are ingested as expected in vector database.

```bash
python -m multimodal.tests.multimodal_query
```

### 5. Load all the source knowledge base in vector database. In case you want to add new files, add them in respective folder and run below commands here after.

```bash
python -m multimodal.ingest.pipeline --all
```


### 6. Test the evaluation. 
Check all the important metrics like faithfulness, answer_relevancy, context_precision, context_recall.
Make sure the APP_MODE is set to eval to test the evaluation in chatbot. It will be off for **APP_MODE = prod**

```bash
python -m multimodal.app
```

### 7. Run the streamlit app

```bash
cd ..

$env:PYTHONPATH = "<root-dir-of-this-project>"
streamlit run multimodal/streamlit_app.py
```


### Project Structure

<pre>

multimodal/
├── __init__.py
├── app.py                        # CLI runner for multimodal queries
├── streamlit_app.py              # Streamlit UI for multimodal chatbot
├── feedback_logger.py            # Logs feedback (human + MLflow)
├── config.py                     # Loads env, validates, defaults
├── core/
│   ├── embeddings.py             # Abstraction: text, audio, image, video embeddings
│   ├── vector_store.py           # Vector DB wrapper (ChromaDB/FAISS/Pinecone)
│   ├── retriever.py              # Retrieval + reranking across modalities
│   ├── llm_adapter.py            # Adapter exposing Groq LLM as LangChain-compatible
│   ├── chain.py                  # Multimodal RAGChain orchestrating retrieval + answer
│   ├── llm.py                    # Groq LLM client
│   └── evaluator.py              # Evaluation (Ragas + MLflow)
├── ingest/
│   ├── text_loader.py            # PDF/doc loader -> chunks text
│   ├── audio_loader.py           # Audio loader -> ASR transcript chunks
│   ├── image_loader.py           # Image loader -> CLIP embeddings
│   ├── video_loader.py           # Video loader -> frame/caption embeddings
│   └── pipeline.py               # Unified ingestion pipeline for all modalities
├── utils/
│   └── errors.py                 # Exceptions
│   └── logger.py                 # Logging
├── tests/
│   ├── ingest.py                 # Smoke test: ingestion + query
│   └── multimodal_query.py       # Test multimodal retrieval
└── requirements.txt


<pre>