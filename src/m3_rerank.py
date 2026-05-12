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
    def __init__(self, model_name: str = "rerank-v3.5"):
        self.model_name = model_name
        self._client = None

    def _get_client(self):
        if self._client is None:
            import cohere
            from config import COHERE_API_KEY
            self._client = cohere.Client(COHERE_API_KEY)
        return self._client

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents using Cohere Rerank API."""
        if not documents: return []
        
        co = self._get_client()
        
        # Prepare passages for Cohere
        passages = [doc["text"] for doc in documents]
        
        response = co.rerank(
            model=self.model_name,
            query=query,
            documents=passages,
            top_n=top_k
        )
        
        results = []
        for i, hit in enumerate(response.results):
            # hit.index là vị trí ban đầu của document trong list passages
            original_doc = documents[hit.index]
            results.append(RerankResult(
                text=original_doc["text"],
                original_score=original_doc.get("score", 0.0),
                rerank_score=float(hit.relevance_score),
                metadata=original_doc.get("metadata", {}),
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
