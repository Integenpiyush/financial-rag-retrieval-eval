# Financial Document Intelligence & RAG System

A retrieval evaluation system over real SEC 10-K filings (Apple, Microsoft,
Amazon — fiscal year 2024), comparing four retrieval strategies against a
hand-labeled ground-truth question set.

**Every number in this README was actually computed by running the code in
this repo** — not estimated or invented. Run `python3 src/evaluate_retrieval.py`
yourself to reproduce them.

## Why this project (and not another PDF chatbot)

The interesting engineering problem in RAG isn't "call an LLM with some
context" — it's proving which retrieval strategy actually finds the right
information, and understanding *why*. This project is built around a
retrieval evaluation harness first; the LLM answer-generation layer is
secondary (see `src/generate_answer.py`).

## Dataset

Real SEC 10-K filings, fetched directly from EDGAR:
- **Apple Inc.**, FY2024 (fiscal year ended Sept 28, 2024)
- **Microsoft Corp.**, FY2024 (fiscal year ended June 30, 2024)
- **Amazon.com, Inc.**, FY2024 (fiscal year ended Dec 31, 2024)

Sections used: Item 1 (Business), Item 1A (Risk Factors), Item 2 (Properties),
Item 3 (Legal Proceedings), Item 7 (MD&A). Chunked into 35 passages
(~120 words each, section-aware, 1-sentence overlap — see
`src/chunk_documents.py`).

## Evaluation set

22 hand-written questions with labeled ground-truth chunk IDs and gold
answers, covering financial figures (net sales, tax rate, headcount),
qualitative facts (segment names, acquisition dates), and legal/regulatory
facts (DOJ lawsuit, DMA investigation) — see `eval/eval_questions.jsonl`.

**Honesty note:** 22 questions over 3 companies is a small evaluation set,
appropriate for demonstrating methodology, not a claim of a comprehensive
financial-QA benchmark. Say this proactively in interviews — it preempts the
obvious follow-up question and shows you understand evaluation validity.

## Results (real, reproducible — run the code yourself)

| Method | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| TF-IDF (baseline) | 0.636 | 0.909 | 0.909 | 0.742 |
| BM25 | **0.682** | 0.909 | **0.955** | **0.784** |
| LSA (dense proxy)* | 0.591 | 0.818 | 0.909 | 0.712 |
| Hybrid (BM25 + LSA, RRF) | 0.682 | 0.864 | 0.909 | 0.782 |

\* LSA (TF-IDF + Truncated SVD) is a dense-retrieval **proxy** used because
this sandbox couldn't download a sentence-transformer model. See "Upgrade
path" below to swap in real embeddings — the retriever interface is
identical, so nothing else needs to change.

### The honest, non-obvious finding

**BM25 alone outperformed the hybrid (BM25 + LSA) approach** on this corpus
— hybrid retrieval did NOT win here. This is a real, defensible result, not
a mistake: this corpus is small (35 chunks) and financial questions are
heavily numeric/keyword-driven ("What was net sales in fiscal 2024" needs
the exact term "net sales" and "2024" matched, not paraphrase-level semantic
similarity). LSA's topic-level signal added noise rather than value here,
so blending it into BM25 via reciprocal rank fusion pulled a few correct
top-1 hits down in rank. **This is a better interview story than "hybrid
always wins"** — it shows you evaluated rather than assumed, and understand
*when* dense/hybrid retrieval helps (paraphrase-heavy queries, large diverse
corpora) versus when strong sparse retrieval alone is sufficient (small,
keyword-heavy, numeric-fact corpora like this one).

### Failure analysis

BM25 missed 1 of 22 questions at k=5: *"What was Apple's total net sales in
fiscal year 2024?"* — the correct chunk states the figure inside a
per-segment breakdown table ("Total net sales: $391,035") rather than as a
standalone sentence, and competing chunks from Microsoft/Amazon MD&A sections
matched more of the generic words ("total", "net", "sales") in the query.
This is a real, generalizable limitation of keyword retrieval: it struggles
when the exact answer is embedded in a table-like block competing against
superficially similar text from *other* documents in the corpus.

## Upgrade path (Tier 2 — run locally or on Colab with full internet)

This sandbox's network is restricted (no Hugging Face model downloads), so
this project ships with a self-contained "Tier 1" evaluation. To get to a
resume-ready "Tier 2" version:

1. `pip install sentence-transformers`
2. In `src/retrievers.py`, replace `LsaRetriever` with a real embedding
   model (`all-mpnet-base-v2` or similar) — the `retrieve(query, k)`
   interface is identical, so `evaluate_retrieval.py` needs zero changes.
3. Add a cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) as
   a final stage after retrieval.
4. Re-run `python3 src/evaluate_retrieval.py` and update the numbers above
   with your real results.
5. For LLM answer generation with citations, run `src/generate_answer.py`
   with your own `ANTHROPIC_API_KEY`.

## Repo structure

```
rag_project/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/           # real 10-K text (Apple, Microsoft, Amazon FY2024)
│   └── chunks/        # chunked passages (chunks.jsonl)
├── eval/
│   └── eval_questions.jsonl   # 22 hand-labeled ground-truth questions
├── src/
│   ├── chunk_documents.py     # section-aware chunking
│   ├── retrievers.py          # TF-IDF, BM25, LSA-proxy, Hybrid (RRF)
│   ├── evaluate_retrieval.py  # Recall@K, MRR evaluation harness
│   └── generate_answer.py     # LLM answer generation (needs your API key)
└── reports/
    └── results/
        └── retrieval_eval.json   # full per-question results
```

## How to reproduce every number in this README

```bash
pip install -r requirements.txt
python3 src/chunk_documents.py
python3 src/evaluate_retrieval.py
```
