"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Tóm tắt đoạn văn sau trong tối đa 2 câu ngắn gọn bằng tiếng Việt. Tập trung vào ý chính nhất."},
                {"role": "user", "content": text},
            ],
            max_tokens=150,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in summarize_chunk: {e}")
        return text[:100] + "..."


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    """
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi ngắn gọn mà đoạn văn này có thể trả lời. Trả về mỗi câu hỏi trên một dòng, không đánh số."},
                {"role": "user", "content": text},
            ],
            max_tokens=200,
        )
        questions = resp.choices[0].message.content.strip().split("\n")
        return [q.strip().lstrip("0123456789.-) ") for q in questions if q.strip()]
    except Exception as e:
        print(f"Error in generate_hypothesis_questions: {e}")
        return []


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    """
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Viết 1 câu cực ngắn (dưới 20 từ) mô tả đoạn văn này nói về chủ đề gì trong ngữ cảnh tài liệu. Chỉ trả về 1 câu duy nhất."},
                {"role": "user", "content": f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}"},
            ],
            max_tokens=80,
        )
        context = resp.choices[0].message.content.strip()
        return f"{context}\n\n{text}"
    except Exception as e:
        print(f"Error in contextual_prepend: {e}")
        return text


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    from openai import OpenAI
    import json
    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": 'Trích xuất metadata từ đoạn văn. Trả về định dạng JSON thuần túy: {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance|other", "language": "vi|en"}'},
                {"role": "user", "content": text},
            ],
            max_tokens=150,
            response_format={"type": "json_object"}
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"Error in extract_metadata: {e}")
        return {}


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks.
    """
    if methods is None:
        methods = ["contextual", "hyqa", "metadata"]

    enriched = []

    print(f"Enriching {len(chunks)} chunks using methods: {methods}...")
    for i, chunk in enumerate(chunks):
        if i % 5 == 0:
            print(f"  Processing chunk {i}/{len(chunks)}...")
            
        summary = ""
        if "summary" in methods or "full" in methods:
            summary = summarize_chunk(chunk["text"])
            
        questions = []
        if "hyqa" in methods or "full" in methods:
            questions = generate_hypothesis_questions(chunk["text"])
            
        enriched_text = chunk["text"]
        if "contextual" in methods or "full" in methods:
            enriched_text = contextual_prepend(chunk["text"], chunk["metadata"].get("source", ""))
            
        auto_meta = {}
        if "metadata" in methods or "full" in methods:
            auto_meta = extract_metadata(chunk["text"])
            
        enriched.append(EnrichedChunk(
            original_text=chunk["text"],
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk["metadata"], **auto_meta},
            method="+".join(methods),
        ))

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
