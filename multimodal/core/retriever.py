from typing import Dict, Any

class Retriever:
    def __init__(
        self,
        vector_stores: Dict[str, Any],
        embedders: Dict[str, Any],
        k_values: Dict[str, int],
        relevance_bias: float = 0.0,
    ):
        """
        vector_stores: dict of modality -> VectorStore
        embedders: dict of modality -> Embedder
        k_values: dict of modality -> top-k values
        relevance_bias: optional bias factor for reranking
        """
        self.vector_stores = vector_stores
        self.embedders = embedders
        self.k_values = k_values
        self.relevance_bias = relevance_bias

    def retrieve(self, query: str) -> Dict[str, Any]:
        """
        Retrieve results across all modalities.
        Returns dict: modality -> raw query results from vector store.
        """
        results = {}
        for modality, store in self.vector_stores.items():
            embedder = self.embedders[modality]
            q_emb = embedder.embed_query_text(query)
            k = self.k_values.get(modality, 5)
            results[modality] = store.query(q_emb, k=k)
        return results

    def _rerank(self, results: Dict[str, Any]) -> list:
        merged = []
        for modality, res in results.items():
            if not res or "documents" not in res:
                continue
            docs = res.get("documents", [[]])[0]
            scores = res.get("distances", [[]])[0]
            metas = res.get("metadatas", [[]])[0]

            for doc, score, meta in zip(docs, scores, metas):
                merged.append({
                    "modality": modality,
                    "document": doc,
                    "score": score + self.relevance_bias,
                    "metadata": meta,
                })

        merged.sort(key=lambda x: x["score"])
        return merged

