"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2"):
        self.model_name = model_name
        self._ranker = None

    def _get_ranker(self):
        if self._ranker is None:
            from flashrank import Ranker
            # FlashRank sẽ tự động download model nhẹ về máy
            self._ranker = Ranker(model_name=self.model_name, cache_dir="flashrank_cache")
        return self._ranker

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents using FlashRank (Local)."""
        if not documents: return []
        
        ranker = self._get_ranker()
        
        # FlashRank yêu cầu format: {"id":..., "text":..., "metadata":...}
        passages = []
        for i, doc in enumerate(documents):
            passages.append({
                "id": i,
                "text": doc["text"],
                "metadata": doc.get("metadata", {})
            })
            
        from flashrank import RerankRequest
        rank_request = RerankRequest(query=query, passages=passages)
        results_raw = ranker.rerank(rank_request)
        
        # Chỉ lấy top_k
        results_raw = results_raw[:top_k]
        
        results = []
        for i, hit in enumerate(results_raw):
            results.append(RerankResult(
                text=hit["text"],
                original_score=0.0, # FlashRank không dùng original score trực tiếp
                rerank_score=float(hit["score"]),
                metadata=hit.get("metadata", {}),
                rank=i + 1
            ))
        return results


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        return []


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs."""
    import numpy as np
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        times.append((time.perf_counter() - start) * 1000)  # ms
    
    return {
        "avg_ms": float(np.mean(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times))
    }


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
