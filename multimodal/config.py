import os
from dotenv import load_dotenv
from multimodal.utils.errors import ConfigError

class Config:
    def __init__(self):
        load_dotenv()
        try:
            # --- LLM ---
            self.groq_api_key = os.getenv("GROQ_API_KEY")
            self.llm_model = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
            self.max_tokens_history = int(os.getenv("MAX_TOKENS_HISTORY", "4000"))
            self.system_prompt_text = os.getenv("SYSTEM_PROMPT_TEXT")
            self.system_prompt_multimodal = os.getenv("SYSTEM_PROMPT_MULTIMODAL")

            # --- Embeddings ---
            self.text_embedding_model = os.getenv("TEXT_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            self.image_embedding_model = os.getenv("IMAGE_EMBEDDING_MODEL", "openai/clip-vit-base-patch32")
            self.audio_transcribe_model = os.getenv("AUDIO_TRANSCRIBE_MODEL", "base")
            self.video_embedding_model = os.getenv("VIDEO_EMBEDDING_MODEL", "openai/clip-vit-base-patch32")
            self.embedding_device = os.getenv("EMBEDDING_DEVICE", "cpu")  # cpu|cuda

            # --- Vector store ---
            self.chroma_dir = os.getenv("CHROMA_DIR", "./chroma_db")
            self.text_collection = os.getenv("TEXT_COLLECTION", "text_docs")
            self.image_collection = os.getenv("IMAGE_COLLECTION", "image_docs")
            self.audio_collection = os.getenv("AUDIO_COLLECTION", "audio_docs")
            self.video_collection = os.getenv("VIDEO_COLLECTION", "video_docs")

            # --- Storage ---
            self.upload_dir = os.getenv("UPLOAD_DIR", "./uploads")
            self.session_log = os.getenv("SESSION_LOG", "./session_log.txt")
            self.storage_dir = os.getenv("STORAGE_DIR", "./storage")
            self.text_dir = os.getenv("TEXT_DIR", "./text")
            self.audio_dir = os.getenv("AUDIO_DIR", "./audio")
            self.image_dir = os.getenv("IMAGE_DIR", "./images")
            self.video_dir = os.getenv("VIDEO_DIR", "./videos")
            self.log_dir = os.getenv("LOG_DIR", "./logs")
            self.feedback_dir = os.getenv("FEEDBACK_DIR", "./feedback")
            self.storage_dir = os.getenv("STORAGE_DIR", "./store_input_docs")

            # --- MLflow ---
            self.mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
            self.mlflow_experiment = os.getenv("MLFLOW_EXPERIMENT", "multimodal_experiment")

            # --- App ---
            self.streamlit_app = os.getenv("STREAMLIT_APP", "Multimodal Finance Assistant")
            self.top_k = int(os.getenv("TOP_K", "5"))
            self.top_k_text = int(os.getenv("TOP_K_TEXT", "5"))
            self.top_k_image = int(os.getenv("TOP_K_IMAGE", "3"))
            self.top_k_audio = int(os.getenv("TOP_K_AUDIO", "3"))
            self.top_k_video = int(os.getenv("TOP_K_VIDEO", "3"))
            self.app_mode = os.getenv("APP_MODE", "eval")  # eval|prod

            # --- Validation ---
            if not self.groq_api_key:
                raise ConfigError("GROQ_API_KEY missing in environment.")

        except Exception as e:
            raise ConfigError(f"Failed to load configuration: {e}")
