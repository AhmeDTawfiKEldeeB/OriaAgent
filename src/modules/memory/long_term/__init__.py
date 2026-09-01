from src.modules.memory.long_term.factory import (
    MemoryFactory,
    VectorStoreFactory,
    get_vector_store,
)
from src.modules.memory.long_term.interface import (
    BaseVectorStore,
    Memory,
)
from src.modules.memory.long_term.provider.mongodb import MongoDBVectorStore
from src.modules.memory.long_term.provider.qdrant import QdrantVectorStore

# Alias for backwards compatibility
VectorStore = QdrantVectorStore

__all__ = [
    "BaseVectorStore",
    "Memory",
    "MemoryFactory",
    "MongoDBVectorStore",
    "QdrantVectorStore",
    "VectorStore",
    "VectorStoreFactory",
    "get_vector_store",
]
