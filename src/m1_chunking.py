"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load all markdown/text files from data/. (Đã implement sẵn)"""
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})
    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề bằng Cohere API.
    """
    metadata = metadata or {}
    
    # 1. Split text into sentences:
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n\n', text) if s.strip()]
    if not sentences:
        return []

    # 2. Encode sentences using Cohere API:
    import cohere
    from config import COHERE_API_KEY, EMBEDDING_MODEL
    
    if not COHERE_API_KEY:
        raise ValueError("COHERE_API_KEY không được tìm thấy trong .env")
        
    co = cohere.Client(COHERE_API_KEY)
    
    # Gọi API để lấy embeddings cho tất cả các câu
    response = co.embed(
        texts=sentences,
        model=EMBEDDING_MODEL,
        input_type="search_document"
    )
    embeddings = response.embeddings

    # 3. Compare consecutive sentences:
    from numpy import dot
    from numpy.linalg import norm
    def cosine_sim(a, b): 
        if norm(a) == 0 or norm(b) == 0: return 0
        return dot(a, b) / (norm(a) * norm(b))

    # 4. Group sentences:
    chunks = []
    current_group = [sentences[0]]
    
    for i in range(1, len(sentences)):
        sim = cosine_sim(embeddings[i-1], embeddings[i])
        if sim < threshold:
            chunks.append(Chunk(
                text=" ".join(current_group), 
                metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"}
            ))
            current_group = []
        current_group.append(sentences[i])
    
    if current_group:
        chunks.append(Chunk(
            text=" ".join(current_group), 
            metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"}
        ))

    return chunks


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Args:
        text: Input text.
        parent_size: Chars per parent chunk.
        child_size: Chars per child chunk.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    # 1. Split text into parents:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    parents_list = []
    current_parent_text = ""
    
    for para in paragraphs:
        if len(current_parent_text) + len(para) > parent_size and current_parent_text:
            pid = f"{metadata.get('source', 'doc')}_p{len(parents_list)}"
            parents_list.append(Chunk(
                text=current_parent_text.strip(), 
                metadata={**metadata, "chunk_type": "parent", "chunk_id": pid, "parent_id": pid},
                parent_id=pid
            ))
            current_parent_text = ""
        current_parent_text += para + "\n\n"
    
    if current_parent_text:
        pid = f"{metadata.get('source', 'doc')}_p{len(parents_list)}"
        parents_list.append(Chunk(
            text=current_parent_text.strip(), 
            metadata={**metadata, "chunk_type": "parent", "chunk_id": pid, "parent_id": pid},
            parent_id=pid
        ))

    # 2. Split each parent into children:
    children_list = []
    for parent in parents_list:
        p_text = parent.text
        pid = parent.metadata["chunk_id"]
        
        # Slide window child_size trên parent text
        for i in range(0, len(p_text), child_size // 2): # Overlap 50%
            child_text = p_text[i : i + child_size].strip()
            if len(child_text) < 50: continue # Skip too small chunks
            
            children_list.append(Chunk(
                text=child_text, 
                metadata={**metadata, "chunk_type": "child", "parent_id": pid},
                parent_id=pid
            ))

    # 3. Return (parents_list, children_list)
    return parents_list, children_list


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.

    Args:
        text: Markdown text.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects, mỗi chunk = 1 section (header + content).
    """
    metadata = metadata or {}
    # 1. Split by markdown headers:
    sections = re.split(r'(^#{1,3}\s+.+$)', text, flags=re.MULTILINE)
    
    # 2. Pair headers with their content:
    chunks = []
    current_header = ""
    current_content = ""
    
    for part in sections:
        if re.match(r'^#{1,3}\s+', part):
            if current_content.strip():
                chunks.append(Chunk(
                    text=f"{current_header}\n{current_content}".strip(),
                    metadata={**metadata, "section": current_header, "strategy": "structure", "chunk_index": len(chunks)}
                ))
            current_header = part.strip()
            current_content = ""
        else:
            current_content += part
            
    # Don't forget last section
    if current_header or current_content.strip():
        chunks.append(Chunk(
            text=f"{current_header}\n{current_content}".strip(),
            metadata={**metadata, "section": current_header, "strategy": "structure", "chunk_index": len(chunks)}
        ))

    # 3. Return chunks — mỗi chunk = 1 section hoàn chỉnh
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.

    Returns:
        {"basic": {...}, "semantic": {...}, "hierarchical": {...}, "structure": {...}}
    """
    stats = {}
    
    all_strategies = {
        "basic": lambda text, meta: chunk_basic(text, metadata=meta),
        "semantic": lambda text, meta: chunk_semantic(text, metadata=meta),
        "hierarchical": lambda text, meta: chunk_hierarchical(text, metadata=meta)[1], # Take children
        "structure": lambda text, meta: chunk_structure_aware(text, metadata=meta)
    }

    print(f"\n{'Strategy':<15} | {'Chunks':<6} | {'Avg Len':<8} | {'Min':<5} | {'Max':<5}")
    print("-" * 50)

    for name, func in all_strategies.items():
        all_chunks = []
        for doc in documents:
            all_chunks.extend(func(doc["text"], doc["metadata"]))
        
        if not all_chunks:
            stats[name] = "Not implemented or no chunks"
            continue
            
        lengths = [len(c.text) for c in all_chunks]
        avg_len = sum(lengths) / len(lengths)
        
        stats[name] = {
            "num_chunks": len(all_chunks),
            "avg_len": int(avg_len),
            "min_len": min(lengths),
            "max_len": max(lengths)
        }
        
        print(f"{name:<15} | {len(all_chunks):<6} | {int(avg_len):<8} | {min(lengths):<5} | {max(lengths):<5}")

    return stats


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
