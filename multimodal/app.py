import os
from multimodal.config import Config
from multimodal.config import Config
from multimodal.config import Config
from multimodal.core.llm import LLMClient
from multimodal.core.embeddings import Embedder
from multimodal.core.vector_store import VectorStore
from multimodal.core.retriever import Retriever
from multimodal.core.chain import RAGChain
from multimodal.core.evaluator import Evaluator
from multimodal.ingest.pipeline import Ingestor
from multimodal.utils.logger import get_logger
from multimodal.ingest.document_loader import PDFLoader


def main():
    cfg = Config()
    logger = get_logger(name="multimodal_app", level="INFO")

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

    logger.info("\n=== RAG Pipeline Output ===")
    logger.info(f"Query: {query}")
    logger.info(f"Answer: {answer}")
    logger.info(f"Evaluation Scores: {scores}")

if __name__ == "__main__":
    main()
