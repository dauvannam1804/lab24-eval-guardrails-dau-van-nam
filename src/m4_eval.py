"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH, OPENAI_API_KEY


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from datasets import Dataset
    
    # Ragas yêu cầu dataset định dạng cụ thể
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }
    dataset = Dataset.from_dict(data)
    
    # Chạy đánh giá (mặc định dùng OpenAI qua biến môi trường OPENAI_API_KEY)
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        
        # Khởi tạo tường minh để tránh lỗi discovery của Ragas
        eval_llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
        eval_embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=eval_llm,
            embeddings=eval_embeddings
        )
    except Exception as e:
        print(f"Ragas evaluation failed with error: {e}")
        print("Attempting fallback with default discovery...")
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
        )
    
    df = result.to_pandas()
    
    per_question = []
    # Ghép kết quả bằng cách lặp qua danh sách gốc để đảm bảo không lỗi KeyError
    for i in range(len(questions)):
        try:
            per_question.append(EvalResult(
                question=questions[i],
                answer=answers[i],
                contexts=contexts[i],
                ground_truth=ground_truths[i],
                faithfulness=float(df.iloc[i].get("faithfulness", 0.0)),
                answer_relevancy=float(df.iloc[i].get("answer_relevancy", 0.0)),
                context_precision=float(df.iloc[i].get("context_precision", 0.0)),
                context_recall=float(df.iloc[i].get("context_recall", 0.0))
            ))
        except Exception as e:
            print(f"Warning: Could not extract scores for question {i}: {e}")

    # Tự tính toán điểm trung bình từ danh sách per_question để đảm bảo chính xác
    # (Do object EvaluationResult của Ragas đôi khi không cho truy cập key trực tiếp)
    n = len(per_question)
    avg_scores = {
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
        "context_precision": 0.0,
        "context_recall": 0.0
    }
    
    if n > 0:
        for m in avg_scores.keys():
            avg_scores[m] = sum(getattr(pq, m, 0.0) for pq in per_question) / n

    print(f"\nDEBUG: Calculated Averages: {avg_scores}")

    return {
        **avg_scores,
        "per_question": per_question
    }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    # 1. Tính điểm trung bình cho mỗi câu
    scored_results = []
    for res in eval_results:
        avg_score = (res.faithfulness + res.answer_relevancy + res.context_precision + res.context_recall) / 4
        scored_results.append((avg_score, res))
        
    # 2. Sort & Take bottom_n
    scored_results.sort(key=lambda x: x[0])
    worst = scored_results[:bottom_n]
    
    failures = []
    for avg, res in worst:
        # 3. Phân tích nguyên nhân dựa trên metric thấp nhất
        metrics = {
            "faithfulness": res.faithfulness,
            "answer_relevancy": res.answer_relevancy,
            "context_precision": res.context_precision,
            "context_recall": res.context_recall
        }
        worst_metric = min(metrics, key=metrics.get)
        
        diagnosis = "Unknown"
        fix = "Need more investigation"
        
        if worst_metric == "faithfulness" and metrics[worst_metric] < 0.8:
            diagnosis = "LLM hallucinating (Ảo giác)"
            fix = "Tighten prompt, lower temperature, or check if context is contradictory"
        elif worst_metric == "context_recall" and metrics[worst_metric] < 0.8:
            diagnosis = "Retrieval failure (Thiếu thông tin)"
            fix = "Improve chunking (M1) or use better Hybrid Search weights (M2)"
        elif worst_metric == "context_precision" and metrics[worst_metric] < 0.8:
            diagnosis = "Noise in context (Quá nhiều rác)"
            fix = "Add Reranking (M3) or improve Metadata Filtering"
        elif worst_metric == "answer_relevancy" and metrics[worst_metric] < 0.8:
            diagnosis = "Poor answer quality"
            fix = "Improve system prompt or use a stronger LLM (GPT-4o)"
            
        failures.append({
            "question": res.question,
            "worst_metric": worst_metric,
            "score": float(metrics[worst_metric]),
            "diagnosis": diagnosis,
            "suggested_fix": fix
        })
        
    return failures


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
