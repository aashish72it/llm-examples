import os
from typing import List
from PIL import Image
from multimodal.utils.errors import IngestError
from multimodal.config import Config
from multimodal.utils.logger import get_logger

cfg = Config()
logger = get_logger(__name__)

class ImageLoader:
    def __init__(self, embedder, vector_store):
        """
        embedder: multimodal Embedder instance
        vector_store: VectorStore instance pointing to IMAGE_COLLECTION
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.image_dir = cfg.image_dir

    def load(self, file_path: str) -> List[dict]:
        """Ingest a single image file (embed + store)."""
        try:
            if not os.path.exists(file_path):
                raise IngestError(f"File not found: {file_path}")

            # --- Step 1: Load image ---
            image = Image.open(file_path).convert("RGB")

            # --- Step 2: Embed image ---
            embedding = self.embedder.embed_image(image)

            # --- Step 3: Store in vector DB ---
            self.vector_store.add_texts(
                ids=[f"image_{os.path.basename(file_path)}"],
                texts=[f"Image embedding for {file_path}"],
                embeddings=[embedding],
                metadatas=[{"source": file_path, "modality": "image"}]
            )

            return [{
                "id": f"image_{os.path.basename(file_path)}",
                "content": f"Image {file_path}",
                "modality": "image"
            }]

        except Exception as e:
            raise IngestError(f"Image ingestion failed: {e}")

    def load_all(self) -> List[dict]:
        """Scan IMAGE_DIR and ingest all supported image files."""
        results = []
        for f in os.listdir(self.image_dir):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(self.image_dir, f)
                logger.info(f"Ingesting {path}")
                results.extend(self.load(path))
        return results
