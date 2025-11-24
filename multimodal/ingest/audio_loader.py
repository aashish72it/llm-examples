import os
from typing import List, Dict
from multimodal.utils.errors import IngestError
from multimodal.config import Config
from multimodal.core.embeddings import Embedder
from multimodal.core.vector_store import VectorStore
from multimodal.utils.logger import get_logger

cfg = Config()
logger = get_logger(__name__)

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks for embedding."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

class AudioLoader:
    def __init__(self, embedder: Embedder, vector_store: VectorStore):
        """
        embedder: multimodal Embedder instance
        vector_store: VectorStore instance pointing to AUDIO_COLLECTION
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.audio_dir = cfg.audio_dir

    def load(self, file_path: str) -> List[Dict]:
        try:
            result = self.embedder.audio_model.transcribe(file_path)
            transcript = result["text"]

            chunks = chunk_text(transcript)
            embeddings = self.embedder.embed_texts(chunks)

            results = []
            for i, chunk in enumerate(chunks):
                embedding = embeddings[i]
                metadata = {"source": file_path, "chunk_index": i}
                chunk_id = f"{os.path.basename(file_path)}_chunk{i}"

                self.vector_store.add_texts(
                    ids=[chunk_id],
                    texts=[chunk],
                    embeddings=[embedding],
                    metadatas=[metadata],
                )

                results.append({"id": chunk_id, "metadata": metadata, "transcript": chunk})

            logger.info(f"Ingested audio file with {len(chunks)} chunks: {file_path}")
            return results
        except Exception as e:
            raise IngestError(f"Failed to ingest audio {file_path}: {e}")


    def load_all(self) -> List[Dict]:
        """Scan AUDIO_DIR and ingest all supported audio files."""
        results = []
        for f in os.listdir(self.audio_dir):
            if f.lower().endswith((".mp3", ".wav", ".m4a")):
                path = os.path.join(self.audio_dir, f)
                logger.info(f"Ingesting {path}")
                results.extend(self.load(path))
        return results
