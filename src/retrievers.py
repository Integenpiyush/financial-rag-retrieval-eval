"""
Retrieval methods for the Financial Document RAG project.

Implements three retrievers that can all run fully offline (no model downloads
needed, important for reproducibility and for environments without internet
access to a model hub):

1. TfidfRetriever   - classic sparse lexical retrieval (baseline)
2. Bm25Retriever    - BM25 sparse retrieval (stronger baseline, accounts for
                      term saturation and document length normalization)
3. LsaRetriever     - a "dense" retrieval proxy: TF-IDF -> Truncated SVD (this
                      is Latent Semantic Analysis). This captures some semantic/
                      topic-level similarity beyond exact keyword match, playing
                      the same conceptual role as a transformer embedding model,
                      without requiring a model download.

UPGRADE PATH (do this once you have unrestricted internet, e.g. on your own
machine or Colab): replace LsaRetriever with a true sentence-transformer
model, e.g.:

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-mpnet-base-v2')
    doc_embeddings = model.encode(chunk_texts)
    query_embedding = model.encode([query])
    # then cosine similarity same as below

The interface (retrieve(query, k)) is identical, so evaluate_retrieval.py
does not need to change - only which retriever class gets instantiated.
"""
import json
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import re


def load_chunks(path):
    chunks = []
    with open(path) as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def simple_tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class TfidfRetriever:
    name = "TF-IDF"

    def __init__(self, chunks):
        self.chunks = chunks
        self.texts = [c["text"] for c in chunks]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_matrix = self.vectorizer.fit_transform(self.texts)

    def retrieve(self, query, k=5):
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.doc_matrix)[0]
        top_idx = np.argsort(-sims)[:k]
        return [(self.chunks[i]["chunk_id"], float(sims[i])) for i in top_idx]


class Bm25Retriever:
    name = "BM25"

    def __init__(self, chunks):
        self.chunks = chunks
        self.tokenized_texts = [simple_tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_texts)

    def retrieve(self, query, k=5):
        tokenized_query = simple_tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_idx = np.argsort(-scores)[:k]
        return [(self.chunks[i]["chunk_id"], float(scores[i])) for i in top_idx]


class LsaRetriever:
    """Dense-retrieval PROXY using Latent Semantic Analysis (TF-IDF + SVD).
    Stand-in for a sentence-transformer model when model downloads are
    unavailable. See module docstring for the real upgrade path."""
    name = "LSA (dense proxy)"

    def __init__(self, chunks, n_components=20):
        self.chunks = chunks
        self.texts = [c["text"] for c in chunks]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = self.vectorizer.fit_transform(self.texts)
        # n_components capped by corpus size for small demo corpora
        n_components = min(n_components, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1] - 1)
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.doc_embeddings = self.svd.fit_transform(tfidf_matrix)

    def retrieve(self, query, k=5):
        q_tfidf = self.vectorizer.transform([query])
        q_embedding = self.svd.transform(q_tfidf)
        sims = cosine_similarity(q_embedding, self.doc_embeddings)[0]
        top_idx = np.argsort(-sims)[:k]
        return [(self.chunks[i]["chunk_id"], float(sims[i])) for i in top_idx]


class HybridRetriever:
    """Combines BM25 and LSA scores via reciprocal rank fusion (RRF).
    RRF is chosen over raw score averaging because BM25 and cosine-similarity
    scores live on different scales; RRF only needs each method's *rank*
    ordering, which is scale-free and much more robust to combine."""
    name = "Hybrid (BM25 + LSA, RRF)"

    def __init__(self, chunks, rrf_k=60):
        self.chunks = chunks
        self.bm25 = Bm25Retriever(chunks)
        self.lsa = LsaRetriever(chunks)
        self.rrf_k = rrf_k

    def retrieve(self, query, k=5):
        n = len(self.chunks)
        bm25_results = self.bm25.retrieve(query, k=n)
        lsa_results = self.lsa.retrieve(query, k=n)

        rrf_scores = {}
        for rank, (chunk_id, _) in enumerate(bm25_results):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (self.rrf_k + rank + 1)
        for rank, (chunk_id, _) in enumerate(lsa_results):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (self.rrf_k + rank + 1)

        ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])[:k]
        return ranked
