"""Production RAG Pipeline — Bài tập NHÓM: ghép M1+M2+M3+M4."""

import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60)

    # Step 1: Load & Chunk (M1)
    print("\n[1/3] Chunking documents...")
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for child in children:
            all_chunks.append({"text": child.text, "metadata": {**child.metadata, "parent_id": child.parent_id}})
    print(f"  {len(all_chunks)} chunks from {len(docs)} documents")

    # Step 2: Enrichment (M5)
    print("\n[2/4] Enriching chunks (M5)...")
    # Giới hạn 30 chunks đầu tiên để chạy demo cho nhanh
    limit = 30
    enriched = enrich_chunks(all_chunks[:limit], methods=["contextual"])
    
    if enriched:
        # Mix enriched chunks with the rest (or just use enriched for testing)
        enriched_data = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        remaining_data = all_chunks[limit:]
        all_chunks = enriched_data + remaining_data
        print(f"  Enriched {len(enriched)} chunks (First {limit} chunks)")
    else:
        print("  ⚠️  M5 not implemented — using raw chunks (fallback)")

    # Step 3: Index (M2)
    print("\n[3/4] Indexing (BM25 + Dense)...")
    search = HybridSearch()
    search.index(all_chunks)

    # Step 4: Reranker (M3)
    print("\n[4/4] Loading reranker...")
    reranker = CrossEncoderReranker()

    return search, reranker


from langfuse import observe

@observe()
def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker):
    """Run full RAG pipeline: Hybrid Search -> Rerank -> Generation -> Guardrail."""
    
    # 1. Search (Hybrid: BM25 + Dense)
    results = search.search(query)
    
    # 2. Reranking
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    contexts = [r.text for r in reranked] if reranked else [r.text for r in results[:3]]
    
    # 3. Generation (LLM)
    from openai import OpenAI
    from config import OPENAI_API_KEY
    from src.m6_guardrails import run_all_guardrails
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    context_str = "\n\n".join([f"--- Context {i+1} ---\n{c}" for i, c in enumerate(contexts)])
    
    try:
        # Trace bước LLM Generation
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """Bạn là một trợ lý ảo thông minh. 
                Hãy trả lời câu hỏi dựa TRÊN VÀ CHỈ DỰA TRÊN ngữ cảnh được cung cấp. 
                Nếu thông tin không có trong ngữ cảnh, hãy trả lời 'Tôi không tìm thấy thông tin này trong tài liệu.' 
                Trả lời bằng tiếng Việt, súc tích và chính xác."""},
                {"role": "user", "content": f"Ngữ cảnh:\n{context_str}\n\nCâu hỏi: {query}"},
            ],
            temperature=0.1,
        )
        answer = resp.choices[0].message.content.strip()
        
        # 4. Guardrail Verification (Phase B.3)
        guard_result = run_all_guardrails(query, answer, contexts)
        
        if not guard_result["passed"]:
            print(f"  ⚠️ Guardrail Rejected: {guard_result['details']['hallucination']['reason']}")
            answer = "Tôi xin lỗi, nhưng tôi không thể tìm thấy thông tin chính xác và tin cậy trong tài liệu để trả lời câu hỏi này."
            
    except Exception as e:
        print(f"Error in LLM Generation/Guardrail: {e}")
        answer = contexts[0] if contexts else "Không tìm thấy thông tin."
        
    return answer, contexts


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    print("\n[Eval] Running queries...")
    test_set = load_test_set()
    questions, answers, all_contexts, ground_truths = [], [], [], []

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...")

    print("\n[Eval] Running RAGAS...")
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
