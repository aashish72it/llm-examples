from typing import List, Dict
from multimodal.utils.errors import ChainError
from multimodal.config import Config

cfg = Config()

SYSTEM_PROMPT_MULTIMODAL = cfg.system_prompt_multimodal

class RAGChain:
    def __init__(self, llm_client, retriever, system_prompt: str = SYSTEM_PROMPT_MULTIMODAL, max_context_docs: int = 5):
        self.llm_client = llm_client
        self.retriever = retriever
        self.system_prompt = system_prompt
        self.max_context_docs = max_context_docs

    def build_context(self, docs: List[Dict]) -> str:
        """
        Build a unified context string from multimodal docs.
        Each doc should have: content, metadata, modality, score.
        """
        try:
            selected = docs[: self.max_context_docs]
            ctx_parts = []
            for i, d in enumerate(selected):
                modality = d.get("modality", "text")
                meta = d.get("metadata", {})
                ctx_parts.append(
                    f"[Doc {i+1} | {modality.upper()}] {d['content']} (meta: {meta})"
                )
            return "\n\n".join(ctx_parts)
        except Exception as e:
            raise ChainError(f"Failed to build multimodal context: {e}")

    def run(self, query: str) -> str:
        """
        Run multimodal RAG chain: retrieve docs, build context, query LLM.
        """
        try:
            docs = self.retriever.retrieve(query)
            context = self.build_context(docs)
            prompt = f"""{self.system_prompt}

Context:
{context}

Question:
{query}
"""
            messages = [{"role": "user", "content": prompt}]
            return self.llm_client.ask_llm(messages)
        except Exception as e:
            raise ChainError(f"Multimodal RAG chain failed: {e}")
