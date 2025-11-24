import os
import cv2
from typing import List
from multimodal.utils.errors import IngestError
from multimodal.config import Config
from multimodal.utils.logger import get_logger

cfg = Config()
logger = get_logger(__name__)

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

class VideoLoader:
    def __init__(self, embedder, vector_store, frame_interval: int = 150):
        """
        embedder: multimodal Embedder instance
        vector_store: VectorStore instance pointing to VIDEO_COLLECTION
        frame_interval: number of frames to skip between samples (default ~5s at 30fps)
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.frame_interval = frame_interval
        self.video_dir = cfg.video_dir

    def load(self, file_path: str) -> List[dict]:
        """Ingest a single video file by sampling frames + transcript chunking."""
        try:
            if not os.path.exists(file_path):
                raise IngestError(f"File not found: {file_path}")

            # --- Step 1: Transcribe audio track ---
            transcript = self.embedder.audio_model.transcribe(file_path)["text"]

            # --- Step 2: Chunk transcript ---
            chunks = chunk_text(transcript, chunk_size=500, overlap=100)
            embeddings = self.embedder.embed_texts(chunks)

            ids, texts, metas, results = [], [], [], []
            for i, chunk in enumerate(chunks):
                emb = embeddings[i]   # ✅ use [i] here
                chunk_id = f"video_{os.path.basename(file_path)}_chunk{i}"
                ids.append(chunk_id)
                texts.append(chunk)
                metas.append({"source": file_path, "chunk_index": i, "modality": "video"})
                results.append({"id": chunk_id, "content": chunk, "modality": "video"})

            # --- Step 3: Store in vector DB ---
            self.vector_store.add_texts(
                ids=ids,
                texts=texts,
                embeddings=embeddings,
                metadatas=metas,
            )

            logger.info(f"Ingested video file with {len(chunks)} transcript chunks: {file_path}")
            return results

        except Exception as e:
            raise IngestError(f"Video ingestion failed: {e}")

    def load_all(self) -> List[dict]:
        """Scan VIDEO_DIR and ingest all supported video files."""
        results = []
        for f in os.listdir(self.video_dir):
            if f.lower().endswith((".mp4", ".mov")):
                path = os.path.join(self.video_dir, f)
                logger.info(f"Ingesting {path}")
                results.extend(self.load(path))
        return results
