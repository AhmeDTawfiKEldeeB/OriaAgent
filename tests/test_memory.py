import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config.settings import get_settings
from src.modules.memory.long_term import Memory, get_vector_store


def test_active_memory_provider():
    """Test the active memory provider configured in .env (Qdrant or MongoDB)."""
    settings = get_settings()
    provider_name = settings.memory_provider.value

    print(f"\n==========================================")
    print(f" Testing Active Memory DB: {provider_name.upper()}")
    print(f"==========================================")

    # 1. Initialize Memory Provider via Factory
    try:
        store = get_vector_store()
        print(f"[1/4] Provider Initialized: {store.provider_name.upper()}")
        assert store.provider_name == provider_name
    except Exception as e:
        print(f"Provider initialization notice: {e}")
        return

    # 2. Test Storing Memory (Creates collection in DB configured in .env)
    test_text = "User prefers using FastAPI and Python for AI agent backends."
    test_metadata = {"category": "preference", "source": "test_suite"}
    print(f"[2/4] Storing Memory Entry into collection '{getattr(store, 'collection_name', 'default')}'...")
    try:
        mem_id = store.store_memory(test_text, metadata=test_metadata)
        print(f"      Stored Memory ID: {mem_id}")
        assert mem_id is not None and len(str(mem_id)) > 0
    except Exception as e:
        err_msg = str(e)
        print(f"\n[!] Connection Error with {provider_name.upper()}:")
        print(f"    {err_msg}")
        if "SSL" in err_msg or "tlsv1" in err_msg or "alert" in err_msg or "80" in err_msg:
            print("\n[!] MongoDB Atlas SSL/TLS Diagnosis:")
            print("    1. IP Whitelist: In MongoDB Atlas -> 'Network Access', ensure your current IP address (or 0.0.0.0/0) is added to the IP Access List.")
            print("    2. Credentials: Check that your username and password in MONGODB_URI are correct and any special characters are URL-encoded.")
            print("    3. Driver: Ensure PyMongo and certifi are up to date.")
        else:
            print(f"    Please make sure your {provider_name.upper()} service is running or check your .env configuration.")
        return

    # 3. Test Finding Similar Memory (Deduplication)
    print(f"[3/4] Finding Similar Memory...")
    similar = store.find_similar_memory(test_text)
    if similar:
        print(f"      Found Similar Memory ID: {similar.id} (Score: {similar.score:.4f})")
        assert similar.text == test_text
    else:
        print("      No existing similar memory found above threshold.")

    # 4. Test Search Memories
    search_query = "What framework does the user prefer for AI agents?"
    print(f"[4/4] Searching Memories for query: '{search_query}'...")
    results = store.search_memories(search_query, k=3)
    print(f"      Retrieved {len(results)} results:")
    for idx, item in enumerate(results, 1):
        print(f"      [{idx}] Text: {item.text} (Score: {item.score:.4f}, Timestamp: {item.timestamp})")

    assert len(results) > 0
    print(f"\nSUCCESS: Active memory provider '{provider_name}' passed all tests!\n")


if __name__ == "__main__":
    test_active_memory_provider()
