"""
Dense embedding and FAISS vector index builder for IFAB Laws of the Game.
Uses BAAI/bge-small-en-v1.5 with L2 normalization and IndexFlatIP (cosine similarity).
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple

class BiEncoderIndexer:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str = None):
        from sentence_transformers import SentenceTransformer
        import torch

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.chunks = []

    def build_index(self, chunks: List[Dict[str, Any]], batch_size: int = 32) -> Any:
        """
        Embeds chunk texts and builds a FAISS IndexFlatIP index.
        Note: Chunk texts are embedded directly WITHOUT query instruction prefix.
        """
        import faiss

        self.chunks = chunks
        texts = [c["text"] for c in chunks]

        # Generate dense embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            device=self.device
        )
        embeddings = np.array(embeddings, dtype=np.float32)

        # Build exact Cosine Similarity index (IndexFlatIP with normalized vectors)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)
        return self.index

    def save(self, index_path: Path, chunks_path: Path):
        """Persists the FAISS index and chunk metadata."""
        import faiss
        import json

        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2, ensure_ascii=False)

    def load(self, index_path: Path, chunks_path: Path):
        """Loads index and metadata from disk."""
        import faiss
        import json

        self.index = faiss.read_index(str(index_path))
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

    def search(self, query: str, top_k: int = 10, prefix: str = "") -> List[Dict[str, Any]]:
        """
        Encodes query with BGE prefix, normalizes, and performs cosine search.
        Returns top-K retrieved candidate dictionaries.
        """
        import faiss

        formatted_query = f"{prefix}{query}" if prefix else query
        query_vec = self.model.encode(
            [formatted_query],
            normalize_embeddings=True,
            device=self.device
        )
        query_vec = np.array(query_vec, dtype=np.float32)

        scores, indices = self.index.search(query_vec, top_k)
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)
            chunk["initial_rank"] = rank
            results.append(chunk)

        return results
