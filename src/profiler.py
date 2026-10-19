"""
Latency Profiling and Scaling Analysis Module.
- Stage-by-stage runtime profiling (Mean, P50, P95).
- Scaling extrapolation comparing All-Cross-Encoder vs Retrieve-then-Re-Rank.
"""

import time
import numpy as np
from typing import List, Dict, Any, Tuple

def profile_retrieval_latencies(
    rag_pipeline,
    queries: List[str],
    num_warmup: int = 3,
    num_runs: int = 5
) -> Dict[str, Dict[str, float]]:
    """
    Profiles execution times of:
    1. Query embedding (Bi-encoder)
    2. FAISS vector search
    3. Cross-encoder scoring (10 pairs)
    4. Pipeline 1 Total vs Pipeline 2 Total
    """
    import faiss

    # Warm-up runs
    for q in queries[:num_warmup]:
        rag_pipeline.run_pipeline_2_rerank(q, top_k_retrieve=10, top_k_prompt=3)

    t_embed = []
    t_faiss = []
    t_ce = []
    t_p1 = []
    t_p2 = []

    prefix = rag_pipeline.bge_prefix
    indexer = rag_pipeline.indexer
    reranker = rag_pipeline.reranker

    for q in queries:
        for _ in range(num_runs):
            # 1. Query Embedding
            t0 = time.perf_counter()
            formatted_query = f"{prefix}{q}"
            q_vec = indexer.model.encode([formatted_query], normalize_embeddings=True, device=indexer.device)
            q_vec = np.array(q_vec, dtype=np.float32)
            t1 = time.perf_counter()
            t_embed.append((t1 - t0) * 1000.0)

            # 2. FAISS Vector Search
            t2 = time.perf_counter()
            scores, indices = indexer.index.search(q_vec, 10)
            t3 = time.perf_counter()
            t_faiss.append((t3 - t2) * 1000.0)

            # Baseline total (retrieval only)
            p1_time = (t1 - t0 + t3 - t2) * 1000.0
            t_p1.append(p1_time)

            # Candidate preparation
            candidates = [indexer.chunks[idx] for idx in indices[0]]
            pairs = [[q, c["text"]] for c in candidates]

            # 3. Cross-Encoder Scoring (10 pairs)
            t4 = time.perf_counter()
            _ = reranker.model.predict(pairs, show_progress_bar=False)
            t5 = time.perf_counter()
            ce_time = (t5 - t4) * 1000.0
            t_ce.append(ce_time)

            # Re-ranking total (retrieval + rerank)
            t_p2.append(p1_time + ce_time)

    def stats(arr):
        return {
            "mean_ms": float(np.mean(arr)),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "std_ms": float(np.std(arr))
        }

    return {
        "query_embedding": stats(t_embed),
        "faiss_search": stats(t_faiss),
        "cross_encoder_10_pairs": stats(t_ce),
        "pipeline_1_baseline_total": stats(t_p1),
        "pipeline_2_rerank_total": stats(t_p2),
        "reranking_overhead": {
            "mean_ms": float(np.mean(t_ce)),
            "relative_slowdown": float(np.mean(t_p2) / max(1e-4, np.mean(t_p1)))
        }
    }

def run_scaling_extrapolation(
    reranker,
    sample_text: str,
    query: str = "Is a player in an offside position penalised?"
) -> List[Dict[str, Any]]:
    """
    Measures runtime per (query, chunk) pair and extrapolates Cross-Encoder execution
    time over full corpora of size N = [current, 1,000, 10,000, 100,000, 1,000,000].
    Contrasts against Two-Stage Retrieve-then-Re-Rank.
    """
    # Profile single pair evaluation over multiple trials
    pairs = [[query, sample_text]] * 20
    # Warmup
    _ = reranker.model.predict(pairs[:5], show_progress_bar=False)

    t0 = time.perf_counter()
    _ = reranker.model.predict(pairs, show_progress_bar=False)
    t1 = time.perf_counter()

    per_pair_sec = (t1 - t0) / len(pairs)
    per_pair_ms = per_pair_sec * 1000.0

    scales = [
        {"name": "Current IFAB Corpus (~500 chunks)", "corpus_size": 500},
        {"name": "Medium Documentation (10,000 chunks)", "corpus_size": 10000},
        {"name": "Large Enterprise Corpus (100,000 chunks)", "corpus_size": 100000},
        {"name": "Web/Legal Library (1,000,000 chunks)", "corpus_size": 1000000}
    ]

    results = []
    for s in scales:
        n = s["corpus_size"]
        exhaustive_ce_sec = n * per_pair_sec
        # Two-stage: bi-encoder vector search ~ constant/O(log N) + 10 CE pairs
        two_stage_sec = 0.005 + (10 * per_pair_sec)

        results.append({
            "corpus_label": s["name"],
            "chunk_count": n,
            "exhaustive_cross_encoder_sec": exhaustive_ce_sec,
            "exhaustive_cross_encoder_formatted": f"{exhaustive_ce_sec:.2f} s" if exhaustive_ce_sec < 60 else f"{exhaustive_ce_sec/60:.1f} min",
            "two_stage_rag_sec": two_stage_sec,
            "two_stage_rag_formatted": f"{two_stage_sec*1000:.1f} ms",
            "speedup_factor": f"{int(exhaustive_ce_sec / two_stage_sec):,}x"
        })

    return results
