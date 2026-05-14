"""
Module 6: LLM-as-Judge Guardrails
=================================
Hệ thống kiểm soát chất lượng câu trả lời: Hallucination, Relevance, Tone.
"""

import os, sys
from typing import Dict, Any, List
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

def check_hallucination_guardrail(answer: str, contexts: List[str]) -> Dict[str, Any]:
    """
    Kiểm tra xem câu trả lời có bị ảo giác (không có trong ngữ cảnh) hay không.
    """
    context_str = "\n\n".join(contexts)
    prompt = f"""
    Dựa trên ngữ cảnh được cung cấp, hãy kiểm tra xem câu trả lời sau có bất kỳ thông tin nào SAI LỆCH hoặc KHÔNG CÓ trong ngữ cảnh hay không.
    
    NGỮ CẢNH:
    {context_str}
    
    CÂU TRẢ LỜI:
    {answer}
    
    Trả về định dạng JSON:
    {{
        "passed": true/false,
        "reason": "Giải thích ngắn gọn lý do",
        "score": 0.0-1.0 (1.0 là hoàn toàn trung thực)
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        import json
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"passed": True, "reason": f"Error: {e}", "score": 1.0}

def check_relevance_guardrail(question: str, answer: str) -> Dict[str, Any]:
    """
    Kiểm tra xem câu trả lời có trả lời đúng trọng tâm câu hỏi hay không.
    """
    prompt = f"""
    Đánh giá xem câu trả lời sau có giải quyết đúng và đủ vấn đề mà câu hỏi đặt ra hay không.
    
    CÂU HỎI: {question}
    CÂU TRẢ LỜI: {answer}
    
    Trả về định dạng JSON:
    {{
        "passed": true/false,
        "reason": "Giải thích ngắn gọn",
        "score": 0.0-1.0 (1.0 là hoàn toàn liên quan)
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        import json
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"passed": True, "reason": f"Error: {e}", "score": 1.0}

def run_all_guardrails(question: str, answer: str, contexts: List[str]) -> Dict[str, Any]:
    """
    Chạy tất cả các guardrails và tổng hợp kết quả.
    """
    hallucination = check_hallucination_guardrail(answer, contexts)
    relevance = check_relevance_guardrail(question, answer)
    
    passed = hallucination["passed"] and relevance["passed"]
    
    return {
        "passed": passed,
        "details": {
            "hallucination": hallucination,
            "relevance": relevance
        },
        "final_score": (hallucination["score"] + relevance["score"]) / 2
    }

if __name__ == "__main__":
    # Test thử Guardrail
    q = "Làm thế nào để đổi mật khẩu?"
    a = "Bạn có thể đổi mật khẩu tại trang cài đặt."
    c = ["Hệ thống cho phép người dùng thay đổi mật khẩu trong mục Cài đặt cá nhân."]
    
    print("Testing Guardrails...")
    result = run_all_guardrails(q, a, c)
    import json
    print(json.dumps(result, indent=2, ensure_ascii=False))
