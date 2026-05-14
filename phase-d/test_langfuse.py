import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import build_pipeline, run_query
from config import LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL

def test_langfuse_integration():
    print("🚀 Starting Phase D: Langfuse Integration Test...")
    
    # Kiểm tra xem key đã được load chưa
    if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
        print("❌ Error: LANGFUSE keys not found in .env!")
        return

    print(f"  Base URL: {LANGFUSE_BASE_URL}")
    print(f"  Public Key: {LANGFUSE_PUBLIC_KEY[:10]}...")

    search, reranker = build_pipeline()
    
    test_queries = [
        "Phase A trong lab 24 yêu cầu làm gì?",
        "Langfuse giúp ích gì trong việc quan sát hệ thống RAG?",
    ]
    
    print("\n⏳ Sending queries to trace in Langfuse...")
    for q in test_queries:
        print(f"\nQuery: {q}")
        answer, _ = run_query(q, search, reranker)
        print(f"Answer: {answer}")
        # Chờ một chút để langfuse flush data (background thread)
        time.sleep(2)

    print("\n" + "=" * 60)
    print("PHASE D INTEGRATION TEST COMPLETE")
    print("=" * 60)
    print("Hãy kiểm tra Dashboard của bạn tại: https://cloud.langfuse.com")

if __name__ == "__main__":
    test_langfuse_integration()
