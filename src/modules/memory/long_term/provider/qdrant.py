import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
except Exception:
    QdrantClient = None

    class Distance:  # type: ignore[no-redef]
        COSINE = "Cosine"
        DOT = "Dot"
        EUCLID = "Euclid"

    class PointStruct:  # type: ignore[no-redef]
        def __init__(self, id=None, vector=None, payload=None):
            self.id = id
            self.vector = vector
            self.payload = payload or {}

    class VectorParams:  # type: ignore[no-redef]
        def __init__(self, size=None, distance=None):
            self.size = size
            self.distance = distance

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

from src.config.settings import QdrantSettings, get_settings
from src.modules.memory.long_term.interface import BaseVectorStore, Memory


class QdrantVectorStore(BaseVectorStore):
    """Vector storage provider using Qdrant for long term memory.

    Supports:
    - Remote / Cloud: QDRANT_URL="https://xxx.qdrant.tech" with QDRANT_API_KEY
    - Local Server: QDRANT_URL="http://localhost:6333"
    - In-Memory (No server required): QDRANT_URL=":memory:"
    - Local Directory (No server required): QDRANT_URL="./qdrant_storage"
    """

    def __init__(self, settings: Optional[QdrantSettings] = None) -> None:
        self._settings = settings or get_settings().qdrant
        self._validate_config()
        if QdrantClient is None:
            raise ImportError(
                "qdrant-client is required for QdrantVectorStore. "
                "Please ensure qdrant-client is properly installed."
            )
        if SentenceTransformer is None:
            raise ImportError(
                "SentenceTransformer is required for Qdrant vector embeddings. "
                "Please ensure sentence-transformers and its dependencies are properly installed."
            )
        self.model = SentenceTransformer(self._settings.embedding_model)

        url = self._settings.url.strip()
        if url == ":memory:":
            self.client = QdrantClient(location=":memory:")
        elif url.startswith("./") or url.startswith(".\\") or url.startswith("path:"):
            path = url.replace("path:", "").strip()
            self.client = QdrantClient(path=path)
        else:
            client_kwargs: Dict[str, Any] = {"url": url}
            if self._settings.api_key:
                client_kwargs["api_key"] = self._settings.api_key
            self.client = QdrantClient(**client_kwargs)

        self.collection_name = self._settings.collection_name
        self.similarity_threshold = self._settings.similarity_threshold

    @property
    def provider_name(self) -> str:
        return "qdrant"

    def _validate_config(self) -> None:
        """Validate that all required configuration settings are set."""
        if not self._settings.url:
            raise ValueError("Missing required Qdrant URL configuration (QDRANT_URL).")

    def _collection_exists(self) -> bool:
        """Check if the memory collection exists."""
        collections = self.client.get_collections().collections
        return any(col.name == self.collection_name for col in collections)

    def _create_collection(self) -> None:
        """Create a new collection for storing memories."""
        sample_embedding = self.model.encode("sample text")
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=len(sample_embedding),
                distance=Distance.COSINE,
            ),
        )

    def find_similar_memory(self, text: str) -> Optional[Memory]:
        """Find if a similar memory already exists above threshold."""
        results = self.search_memories(text, k=1)
        if results and results[0].score is not None and results[0].score >= self.similarity_threshold:
            return results[0]
        return None

    def store_memory(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store a new memory in Qdrant or update if a similar one exists."""
        meta = dict(metadata) if metadata else {}

        if "timestamp" not in meta:
            meta["timestamp"] = datetime.now(timezone.utc).isoformat()

        if not self._collection_exists():
            self._create_collection()

        # Check if similar memory exists
        similar_memory = self.find_similar_memory(text)
        if similar_memory and similar_memory.id:
            memory_id = str(similar_memory.id)
            meta["id"] = memory_id
        else:
            memory_id = str(meta.get("id") or uuid.uuid5(uuid.NAMESPACE_DNS, text))
            meta["id"] = memory_id

        embedding = self.model.encode(text)
        point = PointStruct(
            id=memory_id,
            vector=embedding.tolist(),
            payload={
                "text": text,
                **meta,
            },
        )

        self.client.upsert(
            collection_name=self.collection_name,
            points=[point],
        )
        return memory_id

    def search_memories(self, query: str, k: int = 5) -> List[Memory]:
        """Search for similar memories in the Qdrant vector store."""
        if not self._collection_exists():
            return []

        query_embedding = self.model.encode(query)

        if hasattr(self.client, "search"):
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=k,
            )
            return [
                Memory(
                    text=hit.payload.get("text", "") if hit.payload else "",
                    metadata={key: val for key, val in hit.payload.items() if key != "text"} if hit.payload else {},
                    score=hit.score,
                )
                for hit in results
            ]
        else:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding.tolist(),
                limit=k,
            )
            return [
                Memory(
                    text=hit.payload.get("text", "") if hit.payload else "",
                    metadata={key: val for key, val in hit.payload.items() if key != "text"} if hit.payload else {},
                    score=hit.score,
                )
                for hit in response.points
            ]
