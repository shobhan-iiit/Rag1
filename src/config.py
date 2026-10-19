"""
Configuration and parameter management for IFAB Laws of the Game RAG.
Supports local execution, Google Colab, and Kaggle environments automatically.
"""

import os
from pathlib import Path

# Base Paths
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = WORKSPACE_DIR / "artifacts"
DATA_DIR = WORKSPACE_DIR / "data"

# Auto-detect PDF across common paths (Kaggle dataset, local dir, data dir)
PDF_CANDIDATE_PATHS = [
    Path("/kaggle/input/laws-of-the-game-2026-27/Laws of the Game 2026_27_single pages.pdf"),
    Path("/kaggle/input/ifab-laws-of-the-game-2026-27/Laws of the Game 2026_27_single pages.pdf"),
    WORKSPACE_DIR / "Laws of the Game 2026_27_single pages.pdf",
    DATA_DIR / "Laws of the Game 2026_27_single pages.pdf",
]

OFFICIAL_PDF_URL = "https://downloads.theifab.com/assets/download/laws-of-the-game-2026-27-single-pages.pdf"

# Model Specifications
BI_ENCODER_MODEL = "BAAI/bge-small-en-v1.5"
# Required by assignment brief
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Secondary contingency bi-encoder if needed
CONTINGENCY_BI_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"

# Retrieval & Chunking Parameters
CHUNK_TOKEN_CAP = 320      # Stricter cap evaluated via Cross-Encoder tokenizer
CHUNK_TOKEN_OVERLAP = 0    # Clause-aware, non-overlapping
TOP_K_RETRIEVE = 10        # Candidate pool size
TOP_K_RERANK = 3           # Final context passages for LLM

# Document Scope (Fixed)
START_PAGE = 45            # Law 1 start
END_PAGE = 150             # Law 17 end

# BGE Instruction Prefix (Applied to Queries ONLY, never chunks)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Execution Flags
REBUILD_INDEX = False      # If True, rebuilds FAISS from PDF even if artifacts exist
RUN_LLM = True             # If True, invokes LLM; if False or key missing, uses cached outputs
SEED = 42

def resolve_pdf_path() -> Path:
    """Finds existing PDF path from candidate locations."""
    for path in PDF_CANDIDATE_PATHS:
        if path.exists():
            return path
    # Default to workspace root
    return WORKSPACE_DIR / "Laws of the Game 2026_27_single pages.pdf"
