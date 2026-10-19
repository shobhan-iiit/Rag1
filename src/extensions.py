"""
Extensions module:
- Core 1: Candidate Pool Size Ablation (K = 5, 10, 20, 50)
- Core 2: Hybrid Retrieval (BM25 + Dense via Reciprocal Rank Fusion) + Re-Ranking
"""

import time
import re
from typing import List, Dict, Any, Set
from collections import defaultdict

def run_candidate_pool_ablation(
    rag_pipeline,
    queries: List[Dict[str, Any]],
    query_to_gold_chunks: Dict[str, Set[str]],
    pool_sizes: List[int] = [5, 10, 20, 50]
) -> List[Dict[str, Any]]:
    """
    Evaluates effect of candidate pool size K on Recall@3 and re-ranking latency.
    """
    results = []

    for k in pool_sizes:
        recalls_at_3 = []
        latencies = []

        for q in queries:
            qid = q["id"]
            gold_ids = query_to_gold_chunks.get(qid, set())
            query_str = q["query"]

            t0 = time.perf_counter()
            # Retrieve K candidates
            candidates = rag_pipeline.indexer.search(query_str, top_k=k, prefix=rag_pipeline.bge_prefix)
            # Re-rank Top K down to Top 3
            reranked_top3 = rag_pipeline.reranker.rerank(query_str, candidates, top_k=3)
            t1 = time.perf_counter()

            latencies.append((t1 - t0) * 1000.0)
            hit = 1.0 if bool(set(c["id"] for c in reranked_top3).intersection(gold_ids)) else 0.0
            recalls_at_3.append(hit)

        results.append({
            "pool_size_k": k,
            "mean_recall_at_3": float(sum(recalls_at_3) / len(recalls_at_3)),
            "mean_latency_ms": float(sum(latencies) / len(latencies))
        })

    return results

class BM25Retriever:
    """Sparse lexical retriever using BM25Okapi."""
    def __init__(self, chunks: List[Dict[str, Any]]):
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self.tokenized_corpus = [self._tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _tokenize(self, text: str) -> List[str]:
        # Simple word tokenization and lowercasing
        return re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            c = dict(self.chunks[idx])
            c["bm25_score"] = float(scores[idx])
            c["bm25_rank"] = rank
            results.append(c)
        return results

def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    rrf_k: int = 60,
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Merges dense and sparse candidates using standard Reciprocal Rank Fusion (RRF).
    Score(d) = sum(1 / (k + rank_i))
    """
    scores = defaultdict(float)
    chunk_map = {}

    for rank, c in enumerate(dense_results, start=1):
        cid = c["id"]
        scores[cid] += 1.0 / (rrf_k + rank)
        chunk_map[cid] = c

    for rank, c in enumerate(sparse_results, start=1):
        cid = c["id"]
        scores[cid] += 1.0 / (rrf_k + rank)
        if cid not in chunk_map:
            chunk_map[cid] = c

    sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)[:top_k]
    fused_results = []
    for rank, cid in enumerate(sorted_ids, start=1):
        item = dict(chunk_map[cid])
        item["rrf_score"] = scores[cid]
        item["initial_rank"] = rank
        fused_results.append(item)

    return fused_results
