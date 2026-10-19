"""
Two-stage retrieval implementation:
- Pipeline 1: Baseline Bi-Encoder retrieval (Top 10 -> Top 3).
- Pipeline 2: Cross-Encoder re-ranking over Top 10 candidates (re-scored -> Top 3).
"""

from typing import List, Dict, Any, Tuple
import copy

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = None):
        from sentence_transformers import CrossEncoder
        import torch

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_name = model_name
        self.model = CrossEncoder(model_name, max_length=512, device=self.device)

    def rerank(self, raw_query: str, candidates: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Takes raw query (NO BGE prefix) and candidate chunks.
        Scores all pairs with CrossEncoder, re-sorts descending, and assigns reranked ranks.
        """
        if not candidates:
            return []

        # Construct pairs: (raw query, candidate chunk text)
        pairs = [[raw_query, c["text"]] for c in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Clone candidates and assign cross-encoder logits
        reranked_candidates = []
        for i, c in enumerate(candidates):
            item = copy.deepcopy(c)
            item["ce_score"] = float(scores[i])
            reranked_candidates.append(item)

        # Sort descending by cross-encoder score
        reranked_candidates.sort(key=lambda x: x["ce_score"], reverse=True)

        # Assign reranked rank
        for new_rank, item in enumerate(reranked_candidates, start=1):
            item["reranked_rank"] = new_rank
            item["rank_shift"] = item["initial_rank"] - new_rank  # Positive means promoted!

        return reranked_candidates[:top_k]

class RAGPipeline:
    def __init__(self, indexer, reranker, bge_prefix: str = "Represent this sentence for searching relevant passages: "):
        self.indexer = indexer
        self.reranker = reranker
        self.bge_prefix = bge_prefix

    def run_pipeline_1_baseline(self, query: str, top_k_retrieve: int = 10, top_k_prompt: int = 3) -> Dict[str, Any]:
        """
        Pipeline 1:
        1. Retrieve Top-K candidates using Bi-Encoder (with query prefix).
        2. Slice Top-3 directly as context.
        """
        candidates = self.indexer.search(query, top_k=top_k_retrieve, prefix=self.bge_prefix)
        top_context = candidates[:top_k_prompt]
        return {
            "query": query,
            "pipeline": "Baseline (Bi-Encoder)",
            "all_candidates": candidates,
            "top_context": top_context
        }

    def run_pipeline_2_rerank(self, query: str, top_k_retrieve: int = 10, top_k_prompt: int = 3) -> Dict[str, Any]:
        """
        Pipeline 2:
        1. Retrieve Top-K candidates using Bi-Encoder (with query prefix).
        2. Re-rank Top-K using Cross-Encoder with raw query (no prefix).
        3. Slice Top-3 re-ranked chunks as context.
        """
        candidates = self.indexer.search(query, top_k=top_k_retrieve, prefix=self.bge_prefix)
        reranked_top = self.reranker.rerank(query, candidates, top_k=top_k_prompt)
        # Also return all candidates with their reranked scores
        all_reranked = self.reranker.rerank(query, candidates, top_k=len(candidates))

        return {
            "query": query,
            "pipeline": "Re-Ranking (Cross-Encoder)",
            "all_candidates": all_reranked,
            "top_context": reranked_top
        }
