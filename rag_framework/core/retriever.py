from typing import List, Dict
from rag_framework.utils.errors import RetrieverError

class Retriever:
    def __init__(self, vector_store, embedder, k: int = 5, relevance_bias: float = 0.0):
        self.vector_store = vector_store
        self.embedder = embedder
        self.k = k
        self.relevance_bias = relevance_bias

    def _apply_relevance_bias(self, results: Dict) -> List[Dict]:
        try:
            docs = []
            for i, doc in enumerate(results.get("documents", [[]])[0]):
                meta = results.get("metadatas", [[]])[0][i] if results.get("metadatas") else {}
                score = results.get("distances", [[]])[0][i] if results.get("distances") else None
                adjusted_score = (score if score is not None else 0.0) - self.relevance_bias
                docs.append({"content": doc, "metadata": meta, "score": adjusted_score})
            return docs
        except Exception as e:
            raise RetrieverError(f"Failed to apply relevance bias: {e}")

    def _rerank(self, query: str, docs: List[Dict]) -> List[Dict]:
        try:
            return sorted(docs, key=lambda d: d.get("score", 0.0))
        except Exception as e:
            raise RetrieverError(f"Reranking failed: {e}")

    def retrieve(self, query: str) -> List[Dict]:
        try:
            q_emb = self.embedder.embed_query(query)
            results = self.vector_store.query(q_emb, k=self.k)
            biased = self._apply_relevance_bias(results)
            reranked = self._rerank(query, biased)
            return reranked
        except Exception as e:
            raise RetrieverError(f"Retrieve failed: {e}")
