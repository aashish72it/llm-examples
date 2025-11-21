import os
from multimodal.config import Config
from multimodal.core.embeddings import Embedder
from multimodal.core.vector_store import VectorStore
from multimodal.core.retriever import Retriever
from multimodal.utils import logger
from multimodal.utils.logger import get_logger

def main():
    cfg = Config()

    # --- Initialize embedder ---
    embedder = Embedder()

    # --- Initialize vector stores per modality ---
    text_store = VectorStore(persist_dir=cfg.chroma_dir, collection_name=cfg.text_collection)
    image_store = VectorStore(persist_dir=cfg.chroma_dir, collection_name=cfg.image_collection)
    audio_store = VectorStore(persist_dir=cfg.chroma_dir, collection_name=cfg.audio_collection)
    video_store = VectorStore(persist_dir=cfg.chroma_dir, collection_name=cfg.video_collection)

    # --- Build vector stores dict ---
    vector_stores={
        "text": text_store,
        "image": image_store,
        "audio": audio_store,
        "video": video_store,
    }

    # --- Build embedders dict ---
    embedders = {
        "text": embedder,
        "image": embedder,
        "audio": embedder,
        "video": embedder
    }

    # --- Build k-values dict ---
    k_values = {
        "text": cfg.top_k_text,
        "image": cfg.top_k_image,
        "audio": cfg.top_k_audio,
        "video": cfg.top_k_video
    }

    # --- Build retriever ---
    retriever = Retriever(
        vector_stores=vector_stores,  # default store, but retriever will query all
        embedders=embedders,
        k_values=k_values
    )

    # --- Run multimodal query ---
    query = "What insights are mentioned about exchange stablization fund in text, audio, images, or video?"
    results = retriever.retrieve(query)

    logger.info(f"Query: {query}")
    logger.info(f"Results: {results}")

    merged = retriever._rerank(results)
    for r in merged:
        logger.info(f"[{r['modality'].upper()}] {r['document']} (score={r['score']})")

if __name__ == "__main__":
    main()
