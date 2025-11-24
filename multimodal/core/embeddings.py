import os
from typing import List, Union
from multimodal.utils.errors import RAGError
from multimodal.config import Config

cfg = Config()

class Embedder:
    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
            from transformers import CLIPProcessor, CLIPModel
            import whisper

            # --- Text ---
            self.text_model = SentenceTransformer(cfg.text_embedding_model, device=cfg.embedding_device)

            # --- Image ---
            self.clip_model = CLIPModel.from_pretrained(cfg.image_embedding_model)
            self.clip_processor = CLIPProcessor.from_pretrained(cfg.image_embedding_model)

            # --- Audio ---
            self.audio_model = whisper.load_model(cfg.audio_transcribe_model)

            # --- Video (reuse CLIP for frames) ---
            self.video_model = self.clip_model
            self.video_processor = self.clip_processor

        except Exception as e:
            raise RAGError(f"Failed to initialize multimodal embedders: {e}")


    # --- Text ---
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        try:
            return [vec.tolist() for vec in self.text_model.encode(texts, convert_to_numpy=True)]
        except Exception as e:
            raise RAGError(f"Text embedding failed: {e}")

    def embed_query_text(self, query: str) -> List[float]:
        try:
            return self.text_model.encode([query], convert_to_numpy=True)[0].tolist()
        except Exception as e:
            raise RAGError(f"Text query embedding failed: {e}")

    # --- Audio ---
    def embed_audio(self, audio_path: str) -> List[float]:
        try:
            result = self.audio_model.transcribe(audio_path)
            transcript = result["text"]
            return self.embed_query_text(transcript)
        except Exception as e:
            raise RAGError(f"Audio embedding failed: {e}")

    # --- Image ---
    def embed_image(self, image) -> List[float]:
        try:
            inputs = self.clip_processor(images=image, return_tensors="pt")
            features = self.clip_model.get_image_features(**inputs)
            return features.detach().cpu().numpy()[0].tolist()
        except Exception as e:
            raise RAGError(f"Image embedding failed: {e}")

    # --- Video ---
    def embed_video_frame(self, frame) -> List[float]:
        try:
            inputs = self.video_processor(images=frame, return_tensors="pt")
            features = self.video_model.get_image_features(**inputs)
            return features.detach().cpu().numpy()[0].tolist()
        except Exception as e:
            raise RAGError(f"Video frame embedding failed: {e}")
