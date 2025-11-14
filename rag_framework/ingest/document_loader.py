import fitz
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag_framework.core.embeddings import Embedder
from rag_framework.core.vector_store import VectorStore
from rag_framework.utils.errors import RAGError

class PDFLoader:
    def __init__(self, embedder: Embedder, vstore: VectorStore):
        self.embedder = embedder
        self.vstore = vstore

    def ingest_pdf(self, file_path: str):
        try:
            with open(file_path, "rb") as f:
                pdf_bytes = f.read()
            doc = fitz.open("pdf", pdf_bytes)
            text = "".join([page.get_text() for page in doc])

            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            docs = splitter.create_documents([text])
            chunks = [d.page_content for d in docs]
            metadatas = [{"source": file_path, "chunk_id": i} for i in range(len(chunks))]

            embeddings = self.embedder.embed_texts(chunks)
            ids = [str(uuid.uuid4()) for _ in chunks]
            self.vstore.add_texts(ids=ids, texts=chunks, embeddings=embeddings, metadatas=metadatas)

            print(f"Ingested {len(chunks)} chunks from {file_path}")
            return chunks
        except Exception as e:
            raise RAGError(f"Failed to ingest PDF {file_path}: {e}")
