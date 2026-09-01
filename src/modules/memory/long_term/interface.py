from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class Memory:
    """Represents a memory entry in the vector store."""

    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None

    @property
    def id(self) -> Optional[str]:
        """Unique ID of the memory entry."""
        return self.metadata.get("id")

    @property
    def timestamp(self) -> Optional[datetime]:
        """Timestamp of the memory creation/update."""
        ts = self.metadata.get("timestamp")
        if not ts:
            return None
        try:
            return datetime.fromisoformat(ts)
        except Exception:
            return None


class BaseVectorStore(ABC):
    """Abstract Base Class for Vector Store Memory Providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the vector store provider."""
        pass

    @abstractmethod
    def find_similar_memory(self, text: str) -> Optional[Memory]:
        """Find if a similar memory already exists above threshold.

        Args:
            text: The text to search for.

        Returns:
            Optional Memory if found, else None.
        """
        pass

    @abstractmethod
    def store_memory(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store a new memory in the vector store or update if similar exists.

        Args:
            text: The text content of the memory.
            metadata: Additional information about the memory (timestamp, tags, etc.).

        Returns:
            The memory ID (string).
        """
        pass

    @abstractmethod
    def search_memories(self, query: str, k: int = 5) -> List[Memory]:
        """Search for similar memories in the vector store.

        Args:
            query: Text query to search for.
            k: Number of results to return.

        Returns:
            List of Memory objects sorted by relevance.
        """
        pass
