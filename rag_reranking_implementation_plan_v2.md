# Final Implementation Plan (v2): IFAB Laws of the Game RAG

**Assignment:** RAG Baseline vs. Cross-Encoder Re-Ranking
**Deadline:** 19 October 2026 (plan targets submission by midday for buffer)
**Deliverables:** GitHub link (single, self-contained notebook) + concise report in own words
**Document:** IFAB Laws of the Game, current edition (record the edition in the README)

---

## 0. Assumptions (stakeholder questions resolved)

These were the open questions from the review. The plan assumes the following answers. If any is wrong, the affected section is noted.

| # | Assumption | Affects |
|---|---|---|
| A1 | Embedding model and LLM are free to choose (local or API) | S3, S5, Models |
| A2 | Indexing a **subset** of the PDF (Laws 1-17) is acceptable if documented | S2 |
| A3 | No fixed query count beyond "at least two failure queries"; a larger annotated set is welcome | S8 |
| A4 | Extensions are welcome but must not bloat the report | Section 5 |
| A5 | The report has a length limit, so only the four required sections are written | Section 6 |
| A6 | Individual work; AI may help with code and planning, **not** report text | Section 6 |
| A7 | An LLM may be used as a secondary cross-check, never as primary ground truth | S9 |
| A8 | The PDF should not be committed; a download cell is used instead | Section 6 |
| A9 | Graders may run the notebook on Colab (free tier, T4 or CPU) | S1, Section 6 |
| A10 | Deadline is end of day 19 Oct, but I aim to submit earlier | Timeline |

---

## 1. Scope in one line

One self-contained notebook that builds a single chunked corpus from the IFAB Laws of the Game, runs a baseline dense retriever and a retrieve-then-rerank pipeline on it, stress-tests both on an annotated query set, profiles latency, and runs two extensions.

---

## 2. Architecture

```
IFAB PDF (downloaded by notebook)
   |  PyMuPDF extraction + cleaning
   v
Clean text + metadata (Law no., Law title, page)
   |  Clause-aware chunking, <=320 tokens (cross-encoder tokenizer), no overlap
   v
Chunk corpus (single source of truth for both pipelines)
   |  BGE-small embeddings (query prefix on queries only) -> L2 normalize
   v
FAISS IndexFlatIP (cosine)
   |
   |-- Pipeline 1: Top-10 -> Top-3 ----------------> Prompt -> LLM -> Answer 1
   |
   '-- Pipeline 2: same Top-10 -> Cross-Encoder (raw query, no prefix)
                   score 10 pairs -> sort -> Top-3 -> Prompt -> LLM -> Answer 2
                   (identical prompt function)

Controls: oracle-context answer (gold chunk -> LLM) for showcase queries
```

Shared between pipelines: corpus, index, chunk text (including header), prompt function, LLM settings. The only variable is the re-ranking step.

---

## 3. Models Used

| Role | Model | Notes |
|---|---|---|
| Bi-encoder | `BAAI/bge-small-en-v1.5` | 384-dim, local. Apply the BGE query instruction prefix to **queries only**. |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Required by the brief. Raw logits, 512-token limit on query + chunk combined. The brief spells it `msmarco-...`; verify the exact Hugging Face ID when loading. |
| LLM | Gemini free tier **or** Ollama (Qwen2.5-7B-Instruct / Llama-3.1-8B) | Temperature 0, fixed prompt. Hosted APIs are not fully deterministic, so outputs are cached and committed. |
| Contingency bi-encoder | `sentence-transformers/all-MiniLM-L6-v2` | Used only if fewer than 5 genuine rank promotions are found (see S9). |
| Extension: sparse | BM25 via `rank_bm25` | Hybrid retrieval. |
| Stretch: alt rerankers | `ms-marco-MiniLM-L-12-v2`, `BAAI/bge-reranker-base` | Quality vs latency table. |
| Optional cross-check | Same LLM as grader | Secondary only; manual labels are ground truth. |

---

## 4. Functional Implementation (by notebook section)

### S1. Config, environment, self-containment
- One config cell: model names, token cap, K values (10 and 3), seed, paths, flags.
- Flags: `RUN_LLM` (use live API/Ollama or load cached answers), `REBUILD_INDEX` (rebuild or load artifacts).
- API keys read from environment variables or Colab secrets, never hardcoded.
- First cells install pinned dependencies (Colab-compatible). `requirements.txt` also provided.
- **Self-contained rule:** if artifacts are missing, the notebook rebuilds chunks and index from the PDF. If no API key is present, it loads the committed cached answers so every output cell still renders.

### S2. Ingestion and cleaning
- Download cell fetches the PDF from the official IFAB URL (set in config); the PDF is not committed.
- **Extraction check first (15 min):** confirm text is selectable, inspect columns, sidebars, and tables on sample pages.
- Extract with PyMuPDF page by page. Remove running headers and footers, page numbers, and hyphenation breaks.
- Detect Law headings (regex such as "Law 11 - Offside") and attach Law number, Law title, and page as metadata.
- **Scope (fixed, documented):** Laws 1-17 only. Front matter, glossary, and protocol sections are excluded. The decision is stated in the notebook and report and never changed after annotation begins.
- **Tables and diagrams:** extract table text if it is clean; otherwise skip. **No gold query may depend on content that was not extracted.**

### S3. Chunking
- Split within Law boundaries, then on clause/heading, paragraph, and sentence boundaries.
- **Hard cap: about 320 tokens**, measured with the **cross-encoder's tokenizer** (the stricter constraint, since the query and chunk must both fit in 512 tokens). Assert in code that no chunk plus a 64-token query exceeds 512.
- **No token overlap** (clause-aware boundaries instead). This avoids near-duplicate candidates crowding the top 10 and keeps gold labels simple. Overlap is only revisited in the optional chunking ablation.
- **Header decision (fixed):** a short header `Law N - Title` is stored as a metadata field and **prepended to the text for both embedding and cross-encoder input in both pipelines**. It is documented as a design choice. Optional: one with/without comparison on the showcase set.
- Stable chunk IDs (e.g. `L12_007`). Save `chunks.json` with ID, Law, page, text, token count.
- Log token-length distribution and eyeball 10 random chunks.

### S4. Indexing
- Embed chunks in batches, L2-normalize, build `IndexFlatIP`. Persist embeddings and index.
- Exact flat search is used; the report notes that ANN would be needed at much larger scale.

### S5. Pipeline 1 (baseline)
- `retrieve(query, k=10)` returns chunk IDs, scores, and text. The BGE prefix is applied here to the query only.
- Take the top 3, call `build_prompt(query, chunks)`, generate **Answer 1**.

### S6. Pipeline 2 (re-ranking)
- `rerank(query, candidates)` receives the **raw query (no prefix)** and the same 10 candidates, builds 10 (query, chunk) pairs, and calls `CrossEncoder.predict`.
- Sort descending, keep the top 3, call the same `build_prompt`, generate **Answer 2**.
- Log original rank and new rank for every candidate.
- Scores are raw logits and are compared only within a query.

### S7. Shared prompt
- One function used by both pipelines and the oracle control. Rule: "Answer only from the provided context. If the answer is not in the context, say so." Chunk IDs are included so the model can cite them.

### S8. Query set and gold labels
- **Adversarial set:** 25-30 queries tagged by type, at least 4-5 each:
  - **Keyword trap:** high lexical overlap with an irrelevant passage; true answer uses different terminology.
  - **Exceptions:** general rule chunk retrieved, exception elsewhere.
  - **Negation and constraints:** "not allowed", "cannot", "must not", role-specific limits.
  - **Buried chunk:** gold chunk expected at ranks 4-10.
- **Control set:** 8-10 natural, plain queries (not engineered to fail), so results are not purely adversarial.
- **Span-anchored gold labels:** each query stores Law, page, and a **verbatim answer span** from the document (plus optional alternate acceptable spans). A script maps each span to the chunk(s) that contain it (whitespace-normalized match). A chunk is relevant iff it contains a gold span. Labels therefore survive any re-chunking.
- Multiple acceptable gold chunks are allowed where rules recur across Laws (e.g. restart phrasing).
- Annotation is done **before** running either pipeline, to avoid biasing labels toward retriever output.
- Gold-span coverage check: assert every span is found in exactly the expected chunk(s); fix queries that fail.

### S9. Evaluation and failure mining
- Per query, log gold rank under baseline and reranker.
- **Metric conventions:** if no gold chunk is in the top 10, rank is undefined, with MRR = 0 and nDCG@10 = 0. Also report **Recall@10 (pool recall)** as the ceiling for any reranker.
- Report Recall@3, MRR, nDCG@10 for both pipelines, **split by query type**, and state clearly that the adversarial set is a stress test, not a measure of general retriever quality. Show the control set separately.
- **Showcase selection rule:** at least one promotion case **per failure category** (keyword trap, exception, negation, buried chunk), chosen by largest rank improvement within each category, plus **at least one honest case where re-ranking does not help or hurts**, with a diagnosis (e.g. 512-token truncation, MS MARCO domain shift, negation missed by both models, gold not in the pool).
- **No-failures contingency:** if fewer than 5 genuine promotions exist, (a) write harder queries, (b) report the finding honestly, and (c) run the contingency bi-encoder `all-MiniLM-L6-v2` as a documented, clearly labeled secondary comparison.
- **Answer analysis:** for each showcase query, print Answer 1 and Answer 2 side by side and label each manually as correct, partial (omits an essential clause), or hallucinated/misstated.
- **Oracle control:** for showcase queries, also generate an answer from the gold chunk(s) directly. If the LLM still errs, the failure is attributed to generation, not retrieval, and is reported as such. This keeps the Answer 1 vs Answer 2 claims honest.

### S10. Latency profiling
- 3-5 warm-up runs, then 20+ timed runs per query using `time.perf_counter()`.
- **Retrieval-side stages** (query embedding, FAISS search, cross-encoder on 10 pairs) are measured over all queries, with the LLM excluded.
- **LLM generation latency** is measured separately on a small **uncached** sample (e.g. 5 queries x 3 runs), because cached calls would show near-zero time. Note that API latency depends on network and rate limits.
- Table: mean, p50, p95 per stage for Pipeline 1 vs Pipeline 2, plus the re-ranking delta.
- Record the context: hardware, CPU or GPU, batch size, corpus size, model names.

### S11. Scaling experiment
- Measure time per (query, chunk) pair, then extrapolate to the corpus size, 10k chunks, and 1M chunks, and compare against FAISS search time. This provides numeric backing for the report's scaling answer.

### S12. Extensions (see Section 5)

---

## 5. Extensions

| Priority | Extension | Purpose |
|---|---|---|
| **Core 1** | Candidate-pool ablation (re-rank top 5 / 10 / 20 / 50) | Recall@3 vs latency; shows the reranker ceiling when the gold chunk is outside the pool |
| **Core 2** | Hybrid retrieval (BM25 + dense via Reciprocal Rank Fusion), then rerank | Tests whether keyword-trap failures stem from the embedding model or the retrieval strategy |
| Stretch | Reranker comparison (L-6 vs L-12 vs bge-reranker-base) | Quality vs latency tradeoff |
| Stretch | Chunking ablation at **128 / 256 / 384** tokens (and overlap on/off) | Effect on both pipelines; gold labels survive because they are span-anchored |

Only the two core extensions are committed to. Stretch items run only if the core work and report outline are done.

---

## 6. Deliverables and Repo Layout

```
repo/
|-- rag_reranking.ipynb      # single self-contained notebook, run top to bottom
|-- artifacts/               # chunks.json, embeddings, cached LLM outputs (committed)
|-- queries.json             # queries, type tags, gold spans (not chunk IDs)
|-- requirements.txt
'-- README.md                # run order, IFAB edition, scope decisions, hardware, flags
```

Notes:
- The PDF is **downloaded by the notebook**, not committed (licensing).
- A grader with no API key can still run the notebook end to end using cached outputs.

### Report (four required sections only)
1. **Architecture comparison:** Bi-Encoder (independent embeddings, cosine) vs Cross-Encoder (joint self-attention over query and chunk).
2. **Rank shift and answer analysis:** rank-change table for failure queries (including at least one per category and one honest failure), plus Answer 1 vs Answer 2, with the oracle control where relevant.
3. **Latency table:** stage breakdown with hardware context.
4. **Scaling:** why a Cross-Encoder is not run over the whole collection, supported by your measured per-pair time.

**Write the report yourself, in your own words, from your own results.** Plan, code, and debugging help is fine; the report text must be yours.

---

## 7. Out of Scope

- Fine-tuning the embedding model or cross-encoder
- Multi-document or multi-PDF corpora
- Indexing the front matter, glossary, and protocol sections of the IFAB PDF
- Image, diagram, or table understanding (OCR and vision models)
- Agentic or multi-hop retrieval, query decomposition, HyDE
- Production serving, APIs, or web apps (Gradio demo not planned)
- Approximate nearest-neighbor indexes (HNSW, IVF), beyond a note in the report
- Large-scale benchmarking beyond the self-annotated set
- Multilingual support (English edition only)
- LLM-as-judge as the primary metric (manual labels are ground truth)

---

## 8. Timeline

| Day | Work | Exit criterion |
|---|---|---|
| **Fri 16 Oct** | Extraction check, S1-S4, S5 (Pipeline 1), begin writing queries and spans | Baseline retrieves and answers end to end; chunk cap assertion passes |
| **Sat 17 Oct** | Finish annotation (S8), S6 (Pipeline 2), run all queries, S9 mining, oracle control | Failure candidates found per category, or contingency triggered |
| **Sun 18 Oct** | S10 latency, S11 scaling, Core extensions 1 and 2, notebook cleanup, push repo, test a clean-kernel run | Notebook runs top to bottom; repo pushed |
| **Mon 19 Oct** | Write the report from your results, final checks, submit by midday | Submitted with buffer |

Scope-cut order if time runs short: stretch extensions first, then Core 2, then Core 1. Never cut the failure analysis, latency table, or scaling section.

---

## 9. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Baseline rarely fails on a small single-document corpus | Harder queries; honest reporting; contingency weaker bi-encoder; oracle and pool-recall metrics |
| Cross-references mean some answers need more than one chunk | Allow multiple gold chunks; report these as limits of top-3 reranking |
| Repeated phrasing across Laws makes gold ambiguous | Span-anchored multi-gold labels |
| MS MARCO domain shift hurts on rulebook text | Keep and diagnose at least one reranker failure |
| LLM prior knowledge of football leaks into answers | "Context only" prompt rule; check for answers not supported by context; oracle control |
| PDF extraction noise or missing table content | Extraction check; inspect 10 random chunks; no gold queries on unextracted content |
| API rate limits or non-determinism | Cache outputs, commit them, `RUN_LLM` flag |
| Cross-encoder scores misinterpreted as probabilities | Compare only within a query; document that they are logits |
| Time pressure | Fixed scope cut order; report gets a full day |

---

## 10. Requirements Traceability

| Assignment requirement | Where satisfied |
|---|---|
| Single complex PDF | S2 (IFAB Laws 1-17) |
| Bi-encoder, vector store, top 10 by cosine | S3-S5 |
| Top 3 baseline chunks to LLM, Answer 1 | S5 |
| Cross-encoder re-ranks same top 10, top 3, Answer 2 | S6 |
| Identical prompt structure | S7 |
| At least 2 failure queries with restored relevance | S8-S9 (one per category) |
| Keyword trap, exception, negation, buried chunk demos | S8-S9 |
| Single well-commented `.ipynb` on GitHub | Section 6, S1 self-containment |
| Report 1: Bi-Encoder vs Cross-Encoder | Report section 1 |
| Report 2: rank shifts and Answer 1 vs 2 | S9, report section 2 |
| Report 3: latency table | S10, report section 3 |
| Report 4: scaling | S11, report section 4 |
| Report in own words | Section 6 (written by you) |

---

## 11. Change Log vs v1

| Review finding | Change in v2 |
|---|---|
| Chunk size exceeded model limits; 1024 ablation meaningless | Cap about 320 tokens by cross-encoder tokenizer with assertion; ablation 128/256/384 |
| Gold labels tied to chunk IDs; overlap duplicates | Span-anchored gold labels mapped to chunks; no overlap |
| Stress-test premise not guaranteed | No-failures contingency; contingency bi-encoder; pool-recall ceiling |
| Header prepending confound | Explicit fixed decision, applied to both pipelines |
| Query prefix ambiguity | Prefix on bi-encoder query only; cross-encoder gets raw query |
| Latency vs caching conflict | LLM latency measured on uncached sample; retrieval stages over all queries |
| Selection and metric bias | One showcase per category plus one honest failure; metrics split by type; control set |
| Notebook not self-contained | Rebuild-or-load logic, env-based keys, cached outputs, download cell |
| Scope half-decided | Laws 1-17 fixed; unextractable content excluded from gold queries |
| Over-strong hallucination attribution | Oracle-context control |
| Tight timeline and extension creep | Two core extensions; stretch items optional; report given a full day with explicit cut order |
