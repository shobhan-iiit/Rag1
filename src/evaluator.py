"""
Evaluation metrics calculation module:
- Recall@K, Pool Recall@10
- Mean Reciprocal Rank (MRR)
- nDCG@10
- Category-level breakdown and rank shift tracking
"""

import numpy as np
import re
from typing import List, Dict, Any, Set

def normalize_text(text: str) -> str:
    """Normalizes whitespace and quotes for robust span matching."""
    text = ''.join(ch for ch in text if ord(ch) >= 32 or ch in '\n\t')
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('’', "'").replace('‘', "'").replace('“', '"').replace('”', '"')
    return text.strip().lower()

def map_gold_spans_to_chunks(chunks: List[Dict[str, Any]], queries: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
    """
    Identifies which chunk IDs contain the verbatim gold answer spans for each query.
    Returns mapping: query_id -> set of matching chunk_ids.
    """
    normalized_chunks = [(c["id"], normalize_text(c["text"])) for c in chunks]
    query_to_gold_chunks = {}

    for q in queries:
        qid = q["id"]
        gold_spans = [normalize_text(s) for s in q.get("gold_spans", [])]
        matched_chunk_ids = set()

        for chunk_id, chunk_text in normalized_chunks:
            # Match if any gold span is found inside the chunk text
            for span in gold_spans:
                if len(span) > 10 and span in chunk_text:
                    matched_chunk_ids.add(chunk_id)
                    break

        query_to_gold_chunks[qid] = matched_chunk_ids

    return query_to_gold_chunks

def compute_reciprocal_rank(retrieved_ids: List[str], gold_ids: Set[str]) -> float:
    """Computes Reciprocal Rank (1/rank) for the first relevant item."""
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in gold_ids:
            return 1.0 / rank
    return 0.0

def compute_recall_at_k(retrieved_ids: List[str], gold_ids: Set[str], k: int) -> float:
    """Returns 1.0 if any gold chunk is present in the top-k retrieved list, else 0.0."""
    top_k_ids = set(retrieved_ids[:k])
    return 1.0 if bool(top_k_ids.intersection(gold_ids)) else 0.0

def compute_ndcg_at_k(retrieved_ids: List[str], gold_ids: Set[str], k: int = 10) -> float:
    """Computes Normalized Discounted Cumulative Gain at K (binary relevance)."""
    dcg = 0.0
    for rank, cid in enumerate(retrieved_ids[:k], start=1):
        if cid in gold_ids:
            dcg += 1.0 / np.log2(rank + 1)

    # Ideal DCG (best possible ordering with min(len(gold_ids), k) relevant chunks)
    ideal_hits = min(len(gold_ids), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / np.log2(r + 1) for r in range(1, ideal_hits + 1))
    return float(dcg / idcg) if idcg > 0 else 0.0

def evaluate_retrieval(
    queries: List[Dict[str, Any]],
    baseline_results: Dict[str, List[Dict[str, Any]]],
    reranked_results: Dict[str, List[Dict[str, Any]]],
    query_to_gold_chunks: Dict[str, Set[str]]
) -> Dict[str, Any]:
    """
    Computes comparative metrics across all queries and categorized subsets.
    """
    records = []
    categories = sorted(list(set(q["category"] for q in queries)))

    for q in queries:
        qid = q["id"]
        category = q["category"]
        gold_ids = query_to_gold_chunks.get(qid, set())

        base_candidates = baseline_results.get(qid, [])
        rerank_candidates = reranked_results.get(qid, [])

        base_ids = [c["id"] for c in base_candidates]
        rerank_ids = [c["id"] for c in rerank_candidates]

        # Find best rank in baseline & reranked
        base_rank = next((r for r, cid in enumerate(base_ids, 1) if cid in gold_ids), None)
        rerank_rank = next((r for r, cid in enumerate(rerank_ids, 1) if cid in gold_ids), None)

        records.append({
            "id": qid,
            "category": category,
            "query": q["query"],
            "gold_chunk_count": len(gold_ids),
            "gold_chunk_ids": list(gold_ids),
            "pool_recall_10": compute_recall_at_k(base_ids, gold_ids, 10),
            "baseline_rank": base_rank,
            "reranked_rank": rerank_rank,
            "rank_shift": (base_rank - rerank_rank) if (base_rank and rerank_rank) else None,
            "baseline_r3": compute_recall_at_k(base_ids, gold_ids, 3),
            "reranked_r3": compute_recall_at_k(rerank_ids, gold_ids, 3),
            "baseline_mrr": compute_reciprocal_rank(base_ids, gold_ids),
            "reranked_mrr": compute_reciprocal_rank(rerank_ids, gold_ids),
            "baseline_ndcg": compute_ndcg_at_k(base_ids, gold_ids, 10),
            "reranked_ndcg": compute_ndcg_at_k(rerank_ids, gold_ids, 10)
        })

    def aggregate(subset):
        if not subset:
            return {}
        n = len(subset)
        return {
            "count": n,
            "pool_recall_10": sum(r["pool_recall_10"] for r in subset) / n,
            "baseline_r3": sum(r["baseline_r3"] for r in subset) / n,
            "reranked_r3": sum(r["reranked_r3"] for r in subset) / n,
            "delta_r3": (sum(r["reranked_r3"] for r in subset) - sum(r["baseline_r3"] for r in subset)) / n,
            "baseline_mrr": sum(r["baseline_mrr"] for r in subset) / n,
            "reranked_mrr": sum(r["reranked_mrr"] for r in subset) / n,
            "delta_mrr": (sum(r["reranked_mrr"] for r in subset) - sum(r["baseline_mrr"] for r in subset)) / n,
            "baseline_ndcg": sum(r["baseline_ndcg"] for r in subset) / n,
            "reranked_ndcg": sum(r["reranked_ndcg"] for r in subset) / n
        }

    overall = aggregate(records)
    by_category = {cat: aggregate([r for r in records if r["category"] == cat]) for cat in categories}

    return {
        "records": records,
        "overall": overall,
        "by_category": by_category
    }
