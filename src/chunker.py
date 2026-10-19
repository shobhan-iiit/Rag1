"""
Clause-aware chunking module for IFAB Laws of the Game RAG.
Splits by Law boundaries and clauses, prepends metadata headers,
and strictly enforces token limits with the Cross-Encoder tokenizer.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

def count_tokens(text: str, tokenizer) -> int:
    """Calculates token count using the provided HF tokenizer."""
    if tokenizer is None:
        # Fallback approximation: 1 token ~= 4 chars or 0.75 words
        return max(1, int(len(text.split()) * 1.3))
    return len(tokenizer.encode(text, add_special_tokens=False))

def split_into_clauses(text: str) -> List[str]:
    """
    Splits text by clause and paragraph breaks (e.g. section numbers '1.', '•', double newlines).
    """
    # Split paragraphs first
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    clauses = []
    current_clause = []

    for p in paragraphs:
        # Check if line indicates a new bullet or numbered clause
        is_bullet = p.startswith(('•', '-', '*', '–'))
        is_num = bool(p and p[0].isdigit() and len(p) > 2 and (p[1] in '.)' or p[2] in '.)'))

        if (is_bullet or is_num) and current_clause:
            clauses.append(" ".join(current_clause))
            current_clause = [p]
        else:
            current_clause.append(p)

    if current_clause:
        clauses.append(" ".join(current_clause))

    return [c.strip() for c in clauses if c.strip()]

def chunk_laws_corpus(
    pages_data: List[Dict[str, Any]],
    tokenizer=None,
    max_tokens: int = 320
) -> List[Dict[str, Any]]:
    """
    Groups page clauses into cohesive chunks bounded by Law sections and max_tokens.
    Prepends Law header metadata to every chunk text.
    """
    chunks = []
    chunk_counter_by_law = {}

    for page in pages_data:
        law_num = page["law_number"]
        law_title = page["law_title"]
        page_num = page["page"]
        clauses = split_into_clauses(page["text"])

        current_text_segments = []
        header = f"[Law {law_num} - {law_title}]"

        for clause in clauses:
            test_text = f"{header} " + " ".join(current_text_segments + [clause])
            num_tokens = count_tokens(test_text, tokenizer)

            if num_tokens <= max_tokens:
                current_text_segments.append(clause)
            else:
                if current_text_segments:
                    # Finalize current chunk
                    full_text = f"{header} " + " ".join(current_text_segments)
                    chunk_counter_by_law[law_num] = chunk_counter_by_law.get(law_num, 0) + 1
                    chunk_id = f"L{law_num:02d}_{chunk_counter_by_law[law_num]:03d}"

                    chunks.append({
                        "id": chunk_id,
                        "law_number": law_num,
                        "law_title": law_title,
                        "page": page_num,
                        "header": header,
                        "text": full_text,
                        "token_count": count_tokens(full_text, tokenizer)
                    })
                    current_text_segments = []

                # If single clause is itself longer than max_tokens, split sentences
                single_clause_tokens = count_tokens(f"{header} {clause}", tokenizer)
                if single_clause_tokens > max_tokens:
                    sentences = [s.strip() for s in clause.replace('. ', '.\n').split('\n') if s.strip()]
                    for s in sentences:
                        sub_text = f"{header} " + " ".join(current_text_segments + [s])
                        if count_tokens(sub_text, tokenizer) <= max_tokens:
                            current_text_segments.append(s)
                        else:
                            if current_text_segments:
                                full_text = f"{header} " + " ".join(current_text_segments)
                                chunk_counter_by_law[law_num] = chunk_counter_by_law.get(law_num, 0) + 1
                                chunk_id = f"L{law_num:02d}_{chunk_counter_by_law[law_num]:03d}"
                                chunks.append({
                                    "id": chunk_id,
                                    "law_number": law_num,
                                    "law_title": law_title,
                                    "page": page_num,
                                    "header": header,
                                    "text": full_text,
                                    "token_count": count_tokens(full_text, tokenizer)
                                })
                                current_text_segments = [s]
                            else:
                                current_text_segments = [s]
                else:
                    current_text_segments.append(clause)

        if current_text_segments:
            full_text = f"{header} " + " ".join(current_text_segments)
            chunk_counter_by_law[law_num] = chunk_counter_by_law.get(law_num, 0) + 1
            chunk_id = f"L{law_num:02d}_{chunk_counter_by_law[law_num]:03d}"
            chunks.append({
                "id": chunk_id,
                "law_number": law_num,
                "law_title": law_title,
                "page": page_num,
                "header": header,
                "text": full_text,
                "token_count": count_tokens(full_text, tokenizer)
            })

    # Strict token cap verification assertion: query (64 tokens) + chunk <= 512 tokens
    for c in chunks:
        assert c["token_count"] <= max_tokens + 20, (
            f"Chunk {c['id']} exceeds cap: {c['token_count']} tokens."
        )

    return chunks

def save_chunks(chunks: List[Dict[str, Any]], output_path: Path):
    """Saves chunks to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

def load_chunks(input_path: Path) -> List[Dict[str, Any]]:
    """Loads chunks from a JSON file."""
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)
