# Phase A.3: Failure Cluster Analysis Report

## 1. Evaluation Summary (Task A.2)
- **Test Set Size:** 51 questions
- **Overall Scores:**
    - **Faithfulness:** 0.4250 (Low - Critical Issue)
    - **Answer Relevancy:** 0.4216 (Low)
    - **Context Recall:** 0.5294 (Average)
    - **Context Precision:** N/A (Data formatting issue)

## 2. Top Failure Clusters (Task A.3)

### Cluster 1: LLM Hallucination (Critical)
- **Symptoms:** High number of questions with `Faithfulness = 0.0`.
- **Diagnosis:** LLM attempts to answer even when the retrieved context is missing specific details.
- **Example Questions:**
    - *"Làm thế nào để sử dụng AI assistant trong quá trình tạo bộ câu hỏi tổng hợp..."*
    - *"Làm thế nào để thực hiện phân tích cụm thất bại..."*
- **Root Cause:** Retrieval failed to provide the exact technical steps from the documentation, but the prompt was too open, encouraging the LLM to provide general AI knowledge instead of grounding it in the provided documents.

### Cluster 2: Retrieval Insufficiency
- **Symptoms:** `Context Recall` is ~0.53.
- **Diagnosis:** The system only retrieves about half of the necessary information to form a complete answer.
- **Root Cause:**
    - Many chunks lack context (only the first 30 were enriched with M5).
    - Dense search might be overwhelmed by the number of similar-sounding technical terms in the documents.

## 3. Recommended Fixes (Acceptance Criteria)

| Priority | Area | Suggested Action |
| :--- | :--- | :--- |
| **P0** | **Prompting** | Update system prompt to be stricter: "Nếu không tìm thấy thông tin trong ngữ cảnh, hãy trả lời 'Tôi không biết'". |
| **P1** | **Enrichment** | Run M5 Enrichment for all 1643 chunks instead of just 30 to provide better retrieval context. |
| **P2** | **Retrieval** | Adjust Hybrid Search weights (alpha) to favor BM25 more for technical keyword matching. |

---
*Report generated automatically after Phase A.2 Evaluation.*
