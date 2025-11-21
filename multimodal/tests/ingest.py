import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from multimodal.config import Config
from multimodal.core.embeddings import Embedder
from multimodal.core.vector_store import VectorStore
from multimodal.utils.logger import get_logger

def main():
    cfg = Config()
    logger = get_logger(name="ingest", level="INFO")
    source_dir = cfg.text_dir

    logger.info(f"source dir: {source_dir}")
    pdf_files = [f for f in os.listdir(source_dir) if f.lower().endswith(".pdf")]
    logger.info(f"pdf files: {pdf_files}")

    if not pdf_files:
        logger.warning("No PDFs found in source dir")
        return

    file_path = os.path.join(source_dir, pdf_files[0])
    logger.info(f"Processing: {file_path}")

    reader = PdfReader(file_path)
    text = " ".join([page.extract_text() or "" for page in reader.pages])

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.create_documents([text])
    chunks = [d.page_content for d in docs]
    metadatas = [{"source": file_path, "chunk_id": i} for i in range(len(chunks))]

    embedder = Embedder()
    embeddings = embedder.embed_texts(chunks)

    vstore = VectorStore(persist_dir=cfg.chroma_dir, collection_name=cfg.text_collection)
    ids = [f"{os.path.basename(file_path)}_chunk_{i}" for i in range(len(chunks))]
    vstore.add_texts(ids=ids, texts=chunks, embeddings=embeddings, metadatas=metadatas)

    logger.info(f"Collection size: {vstore.count()}")
    logger.info(f"Ingested {len(chunks)} chunks into ChromaDB")

    query = "What does this PDF mention about Rule of 72?"
    q_emb = embedder.embed_query_text(query)
    results = vstore.query(q_emb, k=3)
    logger.info(f"Query results: {results}")

if __name__ == "__main__":
    main()
