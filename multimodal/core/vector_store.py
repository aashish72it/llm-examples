import chromadb
from multimodal.utils.errors import VectorStoreError

class VectorStore:
    def __init__(self, persist_dir: str, collection_name: str):
        try:
            self.client = chromadb.PersistentClient(path=persist_dir)
            self.collection = self.client.get_or_create_collection(name=collection_name)
        except Exception as e:
            raise VectorStoreError(f"Failed to init vector store: {e}")

    def add_texts(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict] | None = None
    ):
        try:
            self.collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas or [{}] * len(texts)
            )
        except Exception as e:
            raise VectorStoreError(f"Failed to add to vector store: {e}")

    def query(self, query_embedding: list[float], k: int = 5):
        try:
            return self.collection.query(query_embeddings=[query_embedding], n_results=k)
        except Exception as e:
            raise VectorStoreError(f"Vector store query failed: {e}")

    def count(self) -> int:
        return self.collection.count()
