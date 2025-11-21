class RAGError(Exception):
    """Base exception for RAG framework."""
    pass

class ConfigError(RAGError):
    """Raised when configuration setup fails."""
    pass

class LLMError(RAGError):
    """Raised when LLM client or adapter fails."""
    pass

class VectorStoreError(RAGError):
    """Raised when vector store operations fail."""
    pass

class RetrieverError(RAGError):
    """Raised when retrieval logic fails."""
    pass

class ChainError(RAGError):
    """Raised when RAG chain execution fails."""
    pass

class EvaluationError(RAGError):
    """Raised when evaluation pipeline fails."""
    pass

class IngestError(RAGError):
    """Raised when ingestion of files (text, audio, image, video) fails."""
    pass

class FeedbackError(RAGError):
    """Raised when feedback logging fails."""
    pass
