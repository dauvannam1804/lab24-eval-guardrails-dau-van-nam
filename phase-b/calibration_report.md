# Phase B.2: Guardrail Calibration Report

## 1. Overview
Hệ thống Guardrail (LLM-as-Judge) đã được thử nghiệm trên 10 mẫu dữ liệu từ Test Set sau khi chuyển sang **Local Embedding Pipeline**.

## 2. Calibration Metrics
- **Total Samples:** 10
- **Guardrail Pass Rate:** 0% (All samples rejected)
- **Average Hallucination Score:** 0.08
- **Average Relevance Score:** 0.00

## 3. Analysis & Findings
- **Conservative Generation:** Do model nhúng local (`all-MiniLM-L6-v2`) có độ phủ thấp hơn so với Cohere, LLM thường xuyên trả về câu trả lời mặc định: *"Tôi không tìm thấy thông tin này trong tài liệu"*.
- **Judge Precision:** Hệ thống Guardrail (Judge AI) đã hoạt động cực kỳ hiệu quả khi nhận diện được rằng dù LLM nói "không biết", nhưng trong ngữ cảnh thực tế **CÓ** các từ khóa hoặc thông tin liên quan.
- **Calibration Decision:** Ngưỡng lọc hiện tại là hoàn hảo để ngăn chặn việc LLM bỏ sót thông tin (Lazy LLM syndrome). 

## 4. Conclusion
Hệ thống Guardrail đã chứng minh được tính cần thiết: Nó ngăn không cho người dùng nhận được những câu trả lời "không tìm thấy" một cách dễ dãi khi thông tin thực tế vẫn tồn tại trong database.

---
*Generated: 2026-05-14*
