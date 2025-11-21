import argparse
import os
from multimodal.config import Config
from multimodal.ingest.text_loader import TextLoader
from multimodal.ingest.audio_loader import AudioLoader
from multimodal.ingest.image_loader import ImageLoader
from multimodal.ingest.video_loader import VideoLoader
from multimodal.core.embeddings import Embedder
from multimodal.core.vector_store import VectorStore
from multimodal.utils.logger import get_logger

cfg = Config()
logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Multimodal ingestion pipeline")
    parser.add_argument("--text", action="store_true", help="Ingest all text files")
    parser.add_argument("--audio", action="store_true", help="Ingest all audio files")
    parser.add_argument("--image", action="store_true", help="Ingest all image files")
    parser.add_argument("--video", action="store_true", help="Ingest all video files")
    parser.add_argument("--all", action="store_true", help="Ingest all modalities")
    args = parser.parse_args()

    embedder = Embedder()

    # Build vector stores for each modality
    text_store = VectorStore(cfg.chroma_dir, cfg.text_collection)
    audio_store = VectorStore(cfg.chroma_dir, cfg.audio_collection)
    image_store = VectorStore(cfg.chroma_dir, cfg.image_collection)
    video_store = VectorStore(cfg.chroma_dir, cfg.video_collection)

    if args.text or args.all:
        logger.info("Text ingestion started...")
        loader = TextLoader(embedder, text_store)
        loader.load_all()
        logger.info(f"Text ingestion complete. Collection size: {text_store.count()}")

    if args.audio or args.all:
        logger.info("Audio ingestion started...")
        loader = AudioLoader(embedder, audio_store)
        loader.load_all()
        logger.info(f"Audio ingestion complete. Collection size: {audio_store.count()}")

    if args.image or args.all:
        logger.info("Image ingestion started...")
        loader = ImageLoader(embedder, image_store)
        loader.load_all()
        logger.info(f"Image ingestion complete. Collection size: {image_store.count()}")

    if args.video or args.all:
        logger.info("Video ingestion started...")
        loader = VideoLoader(embedder, video_store)
        loader.load_all()
        logger.info(f"Video ingestion complete. Collection size: {video_store.count()}")

if __name__ == "__main__":
    main()
