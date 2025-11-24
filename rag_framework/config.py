import os
from dotenv import load_dotenv
from rag_framework.utils.errors import ConfigError

class Config:
    def __init__(self):
        load_dotenv()
        try:
            # LLM
            self.groq_api_key = os.getenv("GROQ_API_KEY")
            self.llm_model = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
            self.system_prompt = os.getenv("SYSTEM_PROMPT")
            # Embeddings
            self.embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            self.embedding_device = os.getenv("EMBEDDING_DEVICE", "cpu")  # cpu|cuda

            # Vector store
            self.chroma_dir = os.getenv("CHROMA_DIR", "./chroma_db")
            self.collection_name = os.getenv("COLLECTION_NAME", "documents")

            # Storage and sources
            self.source_dir = os.getenv("SOURCE_DIR", "./data")
            self.storage_dir = os.getenv("STORAGE_DIR", "./storage")
            self.feedback_dir = os.getenv("FEEDBACK_DIR")
            # MLflow
            self.mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
            self.mlflow_experiment = os.getenv("MLFLOW_EXPERIMENT", "rag_experiment")

            # App
            self.streamlit_app = os.getenv("STREAMLIT_APP", "Snowflake Q&A")
            self.top_k_docs = int(os.getenv("TOP_K", "5"))
            self.app_mode = os.getenv("APP_MODE", "eval")  # eval|prod

            if not self.groq_api_key:
                raise ConfigError("GROQ_API_KEY missing in environment.")

        except Exception as e:
            raise ConfigError(f"Failed to load configuration: {e}")
