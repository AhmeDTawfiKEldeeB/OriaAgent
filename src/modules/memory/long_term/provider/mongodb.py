import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import certifi
import numpy as np
from pymongo import MongoClient
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

from src.config.settings import MongoDBSettings, get_settings
from src.modules.memory.long_term.interface import BaseVectorStore, Memory


class MongoDBVectorStore(BaseVectorStore):
    """Vector storage provider using MongoDB for long term memory."""

    def __init__(self, settings: Optional[MongoDBSettings] = None) -> None:
        self._settings = settings or get_settings().mongodb
        self._validate_config()
        if SentenceTransformer is None:
            raise ImportError(
                "SentenceTransformer is required for MongoDB vector embeddings. "
                "Please ensure sentence-transformers and its dependencies are properly installed."
            )
        self.model = SentenceTransformer(self._settings.embedding_model)

        # Configure MongoClient with TLS CA certificate for Atlas / cloud compatibility
        client_kwargs: Dict[str, Any] = {
            "serverSelectionTimeoutMS": 5000,
        }
        if "mongodb+srv://" in self._settings.uri or "ssl=true" in self._settings.uri.lower() or "tls=true" in self._settings.uri.lower():
            client_kwargs["tlsCAFile"] = certifi.where()

        self.client: MongoClient = MongoClient(self._settings.uri, **client_kwargs)
        self.db = self.client[self._settings.database]
        self.collection_name = self._settings.collection_name
        self.collection = self.db[self.collection_name]
        self.similarity_threshold = self._settings.similarity_threshold

        # Ensure collection exists
        self._ensure_collection()

    @property
    def provider_name(self) -> str:
        return "mongodb"

    def _validate_config(self) -> None:
        """Validate that required MongoDB configuration is provided."""
        if not self._settings.uri:
            raise ValueError("Missing required MongoDB URI configuration (MONGODB_URI).")

    def _ensure_collection(self) -> None:
        """Ensure the target collection exists in the database."""
        try:
            if self.collection_name not in self.db.list_collection_names():
                self.db.create_collection(self.collection_name)
        except Exception:
            # Collection may already exist or will be lazily created on first document insert
            pass

    @staticmethod
    def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a = np.array(vec1, dtype=np.float32)
        b = np.array(vec2, dtype=np.float32)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def find_similar_memory(self, text: str) -> Optional[Memory]:
        """Find if a similar memory already exists above threshold."""
        results = self.search_memories(text, k=1)
        if results and results[0].score is not None and results[0].score >= self.similarity_threshold:
            return results[0]
        return None

    def store_memory(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store a new memory in MongoDB or update if a similar one exists."""
        self._ensure_collection()
        meta = dict(metadata) if metadata else {}

        if "timestamp" not in meta:
            meta["timestamp"] = datetime.now(timezone.utc).isoformat()

        # Check if similar memory exists
        similar_memory = self.find_similar_memory(text)
        if similar_memory and similar_memory.id:
            memory_id = str(similar_memory.id)
            meta["id"] = memory_id
        else:
            memory_id = str(meta.get("id") or uuid.uuid5(uuid.NAMESPACE_DNS, text))
            meta["id"] = memory_id

        embedding = self.model.encode(text).tolist()

        doc = {
            "_id": memory_id,
            "text": text,
            "vector": embedding,
            "metadata": meta,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        self.collection.replace_one(
            {"_id": memory_id},
            doc,
            upsert=True,
        )
        return memory_id

    def search_memories(self, query: str, k: int = 5) -> List[Memory]:
        """Search for similar memories in MongoDB using cosine similarity."""
        query_embedding = self.model.encode(query).tolist()

        # Attempt Atlas Vector Search aggregation if vector index exists
        try:
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_index",
                        "path": "vector",
                        "queryVector": query_embedding,
                        "numCandidates": max(k * 10, 20),
                        "limit": k,
                    }
                },
                {
                    "$project": {
                        "_id": 1,
                        "text": 1,
                        "metadata": 1,
                        "score": {"$meta": "vectorSearchScore"},
                    }
                },
            ]
            cursor = self.collection.aggregate(pipeline)
            results = list(cursor)
            if results:
                return [
                    Memory(
                        text=doc.get("text", ""),
                        metadata={**doc.get("metadata", {}), "id": str(doc["_id"])},
                        score=float(doc.get("score", 0.0)),
                    )
                    for doc in results
                ]
        except Exception:
            # Fall back to in-database vector scanning and Python cosine similarity computation
            pass

        # In-memory cosine similarity fallback (works on any MongoDB standalone / replica)
        cursor = self.collection.find({}, {"_id": 1, "text": 1, "vector": 1, "metadata": 1})
        scored_memories: List[tuple[float, dict]] = []

        for doc in cursor:
            doc_vec = doc.get("vector")
            if doc_vec:
                score = self._cosine_similarity(query_embedding, doc_vec)
                scored_memories.append((score, doc))

        # Sort by similarity score descending
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        top_k = scored_memories[:k]

        return [
            Memory(
                text=doc.get("text", ""),
                metadata={**doc.get("metadata", {}), "id": str(doc["_id"])},
                score=score,
            )
            for score, doc in top_k
        ]
