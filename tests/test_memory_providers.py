import os
import sys
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config.settings import Settings
from src.modules.memory.long_term.factory import MemoryFactory
from src.modules.memory.long_term.interface import Memory
from src.modules.memory.long_term.provider.mongodb import MongoDBVectorStore
from src.modules.memory.long_term.provider.qdrant import QdrantVectorStore


def test_qdrant_provider_unit():
    with patch("src.modules.memory.long_term.provider.qdrant.QdrantClient") as mock_qdrant_cls, \
         patch("src.modules.memory.long_term.provider.qdrant.SentenceTransformer") as mock_st_cls:

        mock_qdrant = MagicMock()
        mock_qdrant_cls.return_value = mock_qdrant
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock(tolist=lambda: [0.1, 0.2, 0.3])
        mock_st_cls.return_value = mock_model

        s = Settings(MEMORY_PROVIDER="qdrant", QDRANT_URL="http://localhost:6333")
        store = MemoryFactory.create(settings=s, force_new=True)

        assert isinstance(store, QdrantVectorStore)
        assert store.provider_name == "qdrant"

        # Mock collection check
        mock_col = MagicMock()
        mock_col.name = "long_term_memory"
        mock_qdrant.get_collections.return_value.collections = [mock_col]

        # Test store
        mem_id = store.store_memory("Test text", metadata={"user": "ahmed"})
        assert mem_id is not None
        mock_qdrant.upsert.assert_called_once()
        print("Qdrant unit test passed!")


def test_mongodb_provider_unit():
    with patch("src.modules.memory.long_term.provider.mongodb.MongoClient") as mock_mongo_cls, \
         patch("src.modules.memory.long_term.provider.mongodb.SentenceTransformer") as mock_st_cls:

        mock_client = MagicMock()
        mock_mongo_cls.return_value = mock_client
        mock_db = MagicMock()
        mock_client.__getitem__.return_value = mock_db
        mock_col = MagicMock()
        mock_db.__getitem__.return_value = mock_col

        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock(tolist=lambda: [0.1, 0.2, 0.3])
        mock_st_cls.return_value = mock_model

        s = Settings(MEMORY_PROVIDER="mongodb", MONGODB_URI="mongodb://localhost:27017")
        store = MemoryFactory.create(settings=s, force_new=True)

        assert isinstance(store, MongoDBVectorStore)
        assert store.provider_name == "mongodb"

        # Test store
        mock_col.find.return_value = []
        mem_id = store.store_memory("Test mongo memory", metadata={"user": "ahmed"})
        assert mem_id is not None
        mock_col.replace_one.assert_called_once()

        # Test search
        mock_col.find.return_value = [
            {"_id": "m1", "text": "Test mongo memory", "vector": [0.1, 0.2, 0.3], "metadata": {}}
        ]
        results = store.search_memories("Test query", k=1)
        assert len(results) == 1
        assert results[0].text == "Test mongo memory"
        assert results[0].score is not None
        print("MongoDB unit test passed!")


if __name__ == "__main__":
    test_qdrant_provider_unit()
    test_mongodb_provider_unit()
    print("All provider unit tests passed!")
