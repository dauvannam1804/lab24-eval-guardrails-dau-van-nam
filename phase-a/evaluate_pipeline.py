import os
import pandas as pd
import sys
import time
from dotenv import load_dotenv

# Thêm root vào path để import được src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import build_pipeline, run_query
from src.m4_eval import evaluate_ragas, failure_analysis, save_report

# Load environment variables
load_dotenv()

def run_evaluation():
    print("🚀 Starting Phase A.2: Pipeline Evaluation...")
    
    testset_path = "phase-a/testset_v1.csv"
    if not os.path.exists(testset_path):
        print(f"❌ Error: {testset_path} not found. Run Task A.1 first.")
        return

    # 1. Load Test Set
    df_test = pd.read_csv(testset_path)
    
    # Rename columns to match the expected format in src/m4_eval.py
    df_test = df_test.rename(columns={
        "user_input": "question",
        "reference": "ground_truth",
        "reference_contexts": "contexts"
    })
    
    print(f"✅ Loaded {len(df_test)} test questions.")

    # 2. Build Day 18 Pipeline
    search, reranker = build_pipeline()

    # 3. Run queries to get answers and contexts
    questions = df_test["question"].tolist()
    ground_truths = df_test["ground_truth"].tolist()
    answers = []
    contexts = []

    print("\n⏳ Running queries through pipeline...")
    for i, q in enumerate(questions):
        start_time = time.time()
        answer, context = run_query(q, search, reranker)
        answers.append(answer)
        contexts.append(context)
        elapsed = time.time() - start_time
        print(f"  [{i+1}/{len(questions)}] Done ({elapsed:.1f}s)")

    # 4. Run Ragas Evaluation (M4)
    print("\n⏳ Calculating RAGAS Metrics (Task A.2)...")
    results = evaluate_ragas(questions, answers, contexts, ground_truths)

    # 5. Failure Analysis (Task A.3)
    print("\n⏳ Performing Failure Analysis (Task A.3)...")
    failures = failure_analysis(results.get("per_question", []))

    # 6. Save Results
    report_path = "phase-a/ragas_report.json"
    save_report(results, failures, path=report_path)
    
    # 7. Print Summary
    print("\n" + "=" * 60)
    print("PHASE A EVALUATION SUMMARY")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        score = results.get(m, 0)
        print(f"  {m.replace('_', ' ').title()}: {score:.4f}")
    print("=" * 60)
    print(f"✨ Detailed report saved to {report_path}")

if __name__ == "__main__":
    run_evaluation()
