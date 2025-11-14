import os
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag_framework.config import Config
from rag_framework.core.embeddings import Embedder
from rag_framework.core.vector_store import VectorStore

def main():
    cfg = Config()
    source_dir = cfg.source_dir

    print(f"source dir: {source_dir}")
    pdf_files = [f for f in os.listdir(source_dir) if f.lower().endswith(".pdf")]
    print(f"pdf files: {pdf_files}")

    if not pdf_files:
        print("No PDFs found in source dir")
        return

    file_path = os.path.join(source_dir, pdf_files[0])
    print(f"Processing: {file_path}")

    with open(file_path, "rb") as f:
        pdf_bytes = f.read()
    doc = fitz.open("pdf", pdf_bytes)
    text = "".join([page.get_text() for page in doc])

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.create_documents([text])
    chunks = [d.page_content for d in docs]
    metadatas = [{"source": file_path, "chunk_id": i} for i in range(len(chunks))]

    embedder = Embedder(cfg.embedding_model, device=cfg.embedding_device)
    embeddings = embedder.embed_texts(chunks)

    vstore = VectorStore(persist_dir=cfg.chroma_dir, collection_name=cfg.collection_name)
    ids = [f"{os.path.basename(file_path)}_chunk_{i}" for i in range(len(chunks))]
    vstore.add_texts(ids=ids, texts=chunks, embeddings=embeddings, metadatas=metadatas)

    print("Collection size:", vstore.count())
    print(f"Ingested {len(chunks)} chunks into ChromaDB")

    query = "What does this PDF mention about Snowflake datawarehouse?"
    q_emb = embedder.embed_query(query)
    results = vstore.query(q_emb, k=3)
    print("Query results:", results)

if __name__ == "__main__":
    main()
