import os, sys, time
import pandas as pd
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import build_pipeline, run_query
from src.m6_guardrails import run_all_guardrails

def run_calibration():
    print("🚀 Starting Phase B.2: Guardrail Calibration...")
    
    # Load 10 câu hỏi mẫu từ test set
    df_test = pd.read_csv("phase-a/testset_v1.csv").head(10)
    questions = df_test["user_input"].tolist()
    
    search, reranker = build_pipeline()
    
    results = []
    
    print(f"\n⏳ Calibrating on {len(questions)} samples...")
    for i, q in enumerate(questions):
        print(f"  [{i+1}/{len(questions)}] Processing: {q[:50]}...")
        
        # 1. Lấy câu trả lời gốc (không qua lọc)
        # Để demo, ta sẽ gọi LLM và lấy kết quả trước khi Guardrail can thiệp
        # (Ở đây ta dùng lại logic run_query nhưng log lại quá trình guardrail)
        
        # Chạy query bình thường (đã có guardrail tích hợp ở src/pipeline.py)
        answer, contexts = run_query(q, search, reranker)
        
        # Chạy lại guardrail riêng để lấy chi tiết điểm số (Calibration)
        guard_result = run_all_guardrails(q, answer, contexts)
        
        results.append({
            "question": q,
            "answer": answer,
            "passed": guard_result["passed"],
            "hallucination_score": guard_result["details"]["hallucination"]["score"],
            "relevance_score": guard_result["details"]["relevance"]["score"],
            "reason": guard_result["details"]["hallucination"]["reason"]
        })
        
        # Respect Cohere rate limit
        time.sleep(12)

    # Save Calibration Results
    df_results = pd.DataFrame(results)
    output_path = "phase-b/calibration_results.csv"
    os.makedirs("phase-b", exist_ok=True)
    df_results.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    print("\n" + "=" * 60)
    print("PHASE B CALIBRATION SUMMARY")
    print("=" * 60)
    pass_rate = df_results["passed"].mean() * 100
    print(f"  Guardrail Pass Rate: {pass_rate:.1f}%")
    print(f"  Avg Hallucination Score: {df_results['hallucination_score'].mean():.4f}")
    print(f"  Avg Relevance Score: {df_results['relevance_score'].mean():.4f}")
    print("=" * 60)
    print(f"✨ Calibration report saved to {output_path}")

if __name__ == "__main__":
    run_calibration()
