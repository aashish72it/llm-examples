import os
from typing import List
from multimodal.utils.errors import IngestError
from multimodal.config import Config
from langchain.text_splitter import RecursiveCharacterTextSplitter
from multimodal.utils.logger import get_logger

cfg = Config()
logger = get_logger(__name__)

class TextLoader:
    def __init__(self, embedder, vector_store):
        """
        embedder: multimodal Embedder instance
        vector_store: VectorStore instance pointing to TEXT_COLLECTION
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.text_dir = cfg.text_dir 

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks using LangChain's RecursiveCharacterTextSplitter."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " "]
        )
        return splitter.split_text(text)

    def load(self, file_path: str) -> List[dict]:
        """Ingest a single text file (PDF, DOCX, TXT)."""
        try:
            if not os.path.exists(file_path):
                raise IngestError(f"File not found: {file_path}")

            # --- Step 1: Read text ---
            if file_path.endswith(".pdf"):
                from PyPDF2 import PdfReader
                reader = PdfReader(file_path)
                text = " ".join([page.extract_text() or "" for page in reader.pages])
            elif file_path.endswith(".docx"):
                import docx
                doc = docx.Document(file_path)
                text = " ".join([para.text for para in doc.paragraphs])
            else:  # .txt
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()

            if not text.strip():
                raise IngestError("Empty text document")

            # --- Step 2: Chunk text ---
            chunks = self.chunk_text(text, chunk_size=500, overlap=100)
            ids = [f"text_{i}" for i in range(len(chunks))]

            # --- Step 3: Embed chunks ---
            embeddings = self.embedder.embed_texts(chunks)

            # --- Step 4: Store in vector DB ---
            self.vector_store.add_texts(
                ids=ids,
                texts=chunks,
                embeddings=embeddings,
                metadatas=[{"source": file_path, "modality": "text"} for _ in chunks]
            )

            return [{"id": ids[i], "content": chunks[i], "modality": "text"} for i in range(len(chunks))]

        except Exception as e:
            raise IngestError(f"Text ingestion failed: {e}")

    def load_all(self) -> List[dict]:
        """Scan TEXT_DIR and ingest all supported files."""
        results = []
        for f in os.listdir(self.text_dir):
            if f.lower().endswith((".pdf", ".docx", ".txt")):
                path = os.path.join(self.text_dir, f)
                logger.info(f"Ingesting {path}")
                results.extend(self.load(path))
        return results
