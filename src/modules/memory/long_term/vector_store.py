"""Legacy compatibility module forwarding to provider.qdrant and provider.base."""
from src.modules.memory.long_term.factory import get_vector_store
from src.modules.memory.long_term.interface import Memory
from src.modules.memory.long_term.provider.qdrant import QdrantVectorStore as VectorStore

__all__ = [
    "Memory",
    "VectorStore",
    "get_vector_store",
]
