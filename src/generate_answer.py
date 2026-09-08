"""
Answer generation with citation, using retrieved chunks as context.

SETUP REQUIRED: this needs your own Anthropic API key (the sandbox this
project was scaffolded in doesn't have one available for scripts). To run:

    pip install anthropic
    export ANTHROPIC_API_KEY=your_key_here
    python3 generate_answer.py

This is deliberately kept simple: retrieve top-k chunks with your best
retriever (BM25 won the evaluation in this project - see reports/results/),
then ask the LLM to answer ONLY from those chunks and say "not found in the
provided context" if the answer isn't there. This groundedness constraint is
the single most important line in the prompt - without it you will get
confident-sounding hallucinations that pull in the model's general financial
knowledge instead of your actual retrieved documents, defeating the point of
building a RAG system at all.
"""
import json
import os
import sys
from pathlib import Path
from retrievers import load_chunks, Bm25Retriever

CHUNKS_PATH = Path(__file__).parent.parent / "data" / "chunks" / "chunks.jsonl"

SYSTEM_PROMPT = """You are a financial document QA assistant. You will be given
retrieved passages from SEC 10-K filings and a question. Answer using ONLY the
information in the provided passages. If the passages don't contain the answer,
say "Not found in the provided context." Always cite which chunk_id(s) support
your answer. Do not use outside financial knowledge."""


def build_context(chunks_with_scores, all_chunks_by_id):
    context_blocks = []
    for chunk_id, score in chunks_with_scores:
        chunk = all_chunks_by_id[chunk_id]
        context_blocks.append(f"[{chunk_id}] ({chunk['company']}, {chunk['section']})\n{chunk['text']}")
    return "\n\n".join(context_blocks)


def answer_question(question, retriever, all_chunks_by_id, k=5):
    try:
        import anthropic
    except ImportError:
        print("Run: pip install anthropic")
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Set ANTHROPIC_API_KEY environment variable first.")
        sys.exit(1)

    results = retriever.retrieve(question, k=k)
    context = build_context(results, all_chunks_by_id)

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Retrieved passages:\n\n{context}\n\nQuestion: {question}"
        }]
    )
    return response.content[0].text, [r[0] for r in results]


if __name__ == "__main__":
    chunks = load_chunks(CHUNKS_PATH)
    all_chunks_by_id = {c["chunk_id"]: c for c in chunks}
    retriever = Bm25Retriever(chunks)  # BM25 won the retrieval eval - see reports/

    question = sys.argv[1] if len(sys.argv) > 1 else "What was Apple's total net sales in fiscal 2024?"
    answer, retrieved = answer_question(question, retriever, all_chunks_by_id)
    print(f"Question: {question}")
    print(f"Retrieved: {retrieved}")
    print(f"Answer: {answer}")
