"""
Evaluation harness for the Financial Document RAG project.

Computes Recall@5 and MRR for each retrieval method against the hand-labeled
ground-truth question set in eval/eval_questions.jsonl.

IMPORTANT (see project docs): these numbers are REAL, computed from actually
running these retrievers against the actual chunked filings and the actual
labeled questions - not invented. They are also honestly small-scale (22
questions, 3 companies) - this is a starter evaluation to prove the
methodology, not a claim of a comprehensive financial-QA benchmark.
"""
import json
from pathlib import Path
from retrievers import load_chunks, TfidfRetriever, Bm25Retriever, LsaRetriever, HybridRetriever

CHUNKS_PATH = Path(__file__).parent.parent / "data" / "chunks" / "chunks.jsonl"
EVAL_PATH = Path(__file__).parent.parent / "eval" / "eval_questions.jsonl"
RESULTS_PATH = Path(__file__).parent.parent / "reports" / "results" / "retrieval_eval.json"


def load_eval_questions(path):
    questions = []
    with open(path) as f:
        for line in f:
            questions.append(json.loads(line))
    return questions


def recall_at_k(retrieved_ids, relevant_ids):
    retrieved_set = set(retrieved_ids)
    relevant_set = set(relevant_ids)
    hit = len(retrieved_set & relevant_set) > 0
    return 1.0 if hit else 0.0


def reciprocal_rank(retrieved_ids, relevant_ids):
    relevant_set = set(relevant_ids)
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_set:
            return 1.0 / rank
    return 0.0


def evaluate_retriever(retriever, questions, k=5):
    recalls = []
    mrrs = []
    per_question = []
    for q in questions:
        results = retriever.retrieve(q["question"], k=k)
        retrieved_ids = [r[0] for r in results]
        r = recall_at_k(retrieved_ids, q["relevant_chunk_ids"])
        mrr = reciprocal_rank(retrieved_ids, q["relevant_chunk_ids"])
        recalls.append(r)
        mrrs.append(mrr)
        per_question.append({
            "qid": q["qid"],
            "question": q["question"],
            "retrieved": retrieved_ids,
            "relevant": q["relevant_chunk_ids"],
            "hit": r,
        })
    return {
        "recall_at_k": sum(recalls) / len(recalls),
        "mrr": sum(mrrs) / len(mrrs),
        "n_questions": len(questions),
        "per_question": per_question,
    }


def main():
    chunks = load_chunks(CHUNKS_PATH)
    questions = load_eval_questions(EVAL_PATH)
    print(f"Loaded {len(chunks)} chunks and {len(questions)} eval questions.\n")

    retrievers = [
        TfidfRetriever(chunks),
        Bm25Retriever(chunks),
        LsaRetriever(chunks),
        HybridRetriever(chunks),
    ]

    all_results = {}
    print(f"{'Method':<25} {'Recall@1':>10} {'Recall@3':>10} {'Recall@5':>10} {'MRR':>10}")
    print("-" * 68)
    for retriever in retrievers:
        r1 = evaluate_retriever(retriever, questions, k=1)
        r3 = evaluate_retriever(retriever, questions, k=3)
        r5 = evaluate_retriever(retriever, questions, k=5)
        all_results[retriever.name] = {
            "recall_at_1": r1["recall_at_k"],
            "recall_at_3": r3["recall_at_k"],
            "recall_at_5": r5["recall_at_k"],
            "mrr": r5["mrr"],
            "n_questions": r5["n_questions"],
            "per_question_at_5": r5["per_question"],
        }
        print(f"{retriever.name:<25} {r1['recall_at_k']:>10.3f} {r3['recall_at_k']:>10.3f} {r5['recall_at_k']:>10.3f} {r5['mrr']:>10.3f}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull per-question results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
