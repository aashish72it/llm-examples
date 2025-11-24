from typing import List, Dict
from rag_framework.utils.errors import ChainError

SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the provided context to answer the user's question. "
    "If the answer is not in the context, say you don't know."
)

class RAGChain:
    def __init__(self, llm_client, retriever, system_prompt: str = SYSTEM_PROMPT, max_context_docs: int = 5):
        self.llm_client = llm_client
        self.retriever = retriever
        self.system_prompt = system_prompt
        self.max_context_docs = max_context_docs

    def build_context(self, docs: List[Dict]) -> str:
        try:
            selected = docs[: self.max_context_docs]
            ctx = "\n\n".join([f"[Doc {i+1}] {d['content']}" for i, d in enumerate(selected)])
            return ctx
        except Exception as e:
            raise ChainError(f"Failed to build context: {e}")

    def run(self, query: str) -> str:
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
            raise ChainError(f"RAG chain failed: {e}")
