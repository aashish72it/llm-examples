import uuid
from typing import Iterable
from rag_framework.utils.errors import RAGError

class Ingestor:
    def __init__(self, embedder, vector_store):
        self.embedder = embedder
        self.vector_store = vector_store

    def ingest_texts(self, texts: Iterable[str], metadatas: Iterable[dict] | None = None):
        try:
            texts = list(texts)
            ids = [str(uuid.uuid4()) for _ in texts]
            embeddings = self.embedder.embed_texts(texts)
            self.vector_store.add_texts(
                ids=ids,
                texts=texts,
                embeddings=embeddings,
                metadatas=list(metadatas) if metadatas else None
            )
            return ids
        except Exception as e:
            raise RAGError(f"Ingestion failed: {e}")
