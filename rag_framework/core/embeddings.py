import os
from typing import List
from rag_framework.utils.errors import RAGError
from rag_framework.config import Config

cfg = Config()

class Embedder:
    def __init__(self, model_name: str, device: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
            device = cfg.embedding_device
            self.model = SentenceTransformer(model_name, device=device)
        except Exception as e:
            raise RAGError(f"Failed to initialize embedder: {e}")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        try:
            return [vec.tolist() for vec in self.model.encode(texts, convert_to_numpy=True)]
        except Exception as e:
            raise RAGError(f"Embedding failed: {e}")

    def embed_query(self, query: str) -> List[float]:
        try:
            return self.model.encode([query], convert_to_numpy=True)[0].tolist()
        except Exception as e:
            raise RAGError(f"Query embedding failed: {e}")
