# Lab 24: Full RAG Evaluation & Guardrail System 🚀

Dự án này triển khai một hệ thống RAG (Retrieval-Augmented Generation) hoàn chỉnh với đầy đủ các bước từ xử lý dữ liệu, đánh giá tự động, thiết lập hàng rào bảo vệ (Guardrails) và giám sát hệ thống (Observability).

## 🏗 Kiến trúc Pipeline (M1 - M6)

Hệ thống được xây dựng dựa trên 6 module cốt lõi:
- **M1 (Chunking):** Phân đoạn văn bản theo cấu trúc phân cấp (Hierarchical Chunking).
- **M2 (Search):** Tìm kiếm lai (Hybrid Search) kết hợp BM25 và Local Dense Vectors (Sentence-Transformers).
- **M3 (Rerank):** Sắp xếp lại kết quả bằng **FlashRank** (Local Cross-Encoder).
- **M4 (Eval):** Đánh giá tự động sử dụng khung **RAGAS** (Faithfulness, Relevancy, Precision, Recall).
- **M5 (Enrichment):** Làm giàu dữ liệu bằng phương pháp Contextual Enrichment.
- **M6 (Guardrails):** Lớp kiểm duyệt LLM-as-Judge để ngăn chặn ảo giác và nội dung không phù hợp.

---

## 📊 Kết quả thực hiện qua các Phase

### Phase A: Evaluation & Failure Analysis
- **Kết quả RAGAS:** Ghi nhận điểm số Faithfulness (~0.42) và Context Recall (~0.53).
- **Báo cáo lỗi:** Xác định 2 nhóm lỗi chính là **LLM Hallucination** và **Retrieval Insufficiency**.
- [Xem chi tiết Phase A](./phase-a/failure_analysis.md)

### Phase B: Guardrails & Calibration
- **Cơ chế:** Sử dụng `gpt-4o-mini` làm trọng tài để kiểm duyệt câu trả lời.
- **Hiệu chuẩn:** Thiết lập hệ thống cực kỳ nghiêm ngặt, chặn 100% các câu trả lời không chắc chắn hoặc thiếu căn cứ từ ngữ cảnh.
- [Xem báo cáo Calibration](./phase-b/calibration_report.md)

### Phase C: Final Integration & Stress Test
- **Local Deployment:** Toàn bộ phần Retrieval và Rerank chạy offline, không phụ thuộc API bên ngoài.
- **Stress Test:** Vượt qua 5/5 đòn tấn công giả lập (Jailbreak, Prompt Injection, Out-of-scope).
- [Xem báo cáo Stress Test](./phase-c/stress_test_report.md)

### Phase D: Observability with Langfuse
- **Tracing:** Tích hợp thành công Langfuse để theo dõi chi tiết từng bước chạy của Pipeline.
- **Monitoring:** Theo dõi Cost, Latency và Quality trực tiếp trên Dashboard.

---

## 🛠 Hướng dẫn cài đặt & Sử dụng

### 1. Cài đặt môi trường
Sử dụng `uv` để quản lý dependencies:
```bash
uv sync
```

### 2. Cấu hình biến môi trường
Tạo file `.env` từ `.env.example` và điền các API Key:
```bash
cp .env.example .env
```

### 3. Chạy Pipeline & Đánh giá
```bash
uv run python src/pipeline.py
```

### 4. Chạy Stress Test
```bash
uv run python phase-c/stress_test.py
```

## 📈 Dashboard Observability
Toàn bộ quá trình thực hiện được ghi lại tại: [Langfuse Cloud](https://cloud.langfuse.com)

---
**Dự án hoàn thành bởi: Antigravity AI Assistant 🤖**
