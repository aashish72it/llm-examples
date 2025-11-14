import os
from rag_framework.config import Config
from rag_framework.core.llm import LLMClient
from rag_framework.core.embeddings import Embedder
from rag_framework.core.vector_store import VectorStore
from rag_framework.core.retriever import Retriever
from rag_framework.core.chain import RAGChain
from rag_framework.core.evaluator import Evaluator
from rag_framework.ingest.pipeline import Ingestor
from rag_framework.ingest.document_loader import PDFLoader

def main():
    cfg = Config()

    llm_client = LLMClient(api_key=cfg.groq_api_key, model=cfg.llm_model)
    embedder = Embedder(model_name=cfg.embedding_model, device=cfg.embedding_device)
    vstore = VectorStore(persist_dir=cfg.chroma_dir, collection_name=cfg.collection_name)
    retriever = Retriever(vector_store=vstore, embedder=embedder, k=cfg.top_k_docs, relevance_bias=0.1)
    chain = RAGChain(llm_client=llm_client, retriever=retriever)
    evaluator = Evaluator(tracking_uri=cfg.mlflow_tracking_uri, experiment_name=cfg.mlflow_experiment)
    ingestor = Ingestor(embedder=embedder, vector_store=vstore)
    loader = PDFLoader(embedder=embedder, vstore=vstore)

    # Ingest PDFs
    source_dir = cfg.source_dir
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"SOURCE_DIR not found: {source_dir}")

    for fname in os.listdir(source_dir):
        if fname.lower().endswith(".pdf"):
            file_path = os.path.join(source_dir, fname)
            loader.ingest_pdf(file_path)

    # Query
    query = "Explain how Retrieval-Augmented Generation improves LLM performance."
    answer = chain.run(query)

    # Evaluate
    contexts = [d["content"] for d in retriever.retrieve(query)]
    scores = evaluator.evaluate(query=query, context_docs=contexts, answer=answer)

    print("\n=== RAG Pipeline Output ===")
    print("Query:", query)
    print("Answer:", answer)
    print("Evaluation Scores:", scores)

if __name__ == "__main__":
    main()
