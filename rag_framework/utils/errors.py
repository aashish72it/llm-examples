class RAGError(Exception):
    """Base exception for RAG framework."""
    pass

class ConfigError(RAGError):
    pass

class LLMError(RAGError):
    pass

class VectorStoreError(RAGError):
    pass

class RetrieverError(RAGError):
    pass

class ChainError(RAGError):
    pass

class EvaluationError(RAGError):
    pass
