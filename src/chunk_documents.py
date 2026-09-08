"""
Chunking script for the Financial Document RAG project.

Splits each raw 10-K text file into overlapping paragraph-based chunks and
writes them to data/chunks/chunks.jsonl with metadata (company, chunk_id, source).

Design choice: paragraph-based chunking with a target size of ~120 words and
a 1-sentence overlap between consecutive chunks, rather than naive fixed
character-count splitting. This keeps each chunk semantically coherent
(doesn't cut a sentence in half) which matters a lot for retrieval quality
on financial text where a single sentence often carries the key number.
"""
import json
import os
import re
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
OUT_PATH = Path(__file__).parent.parent / "data" / "chunks" / "chunks.jsonl"

TARGET_WORDS = 120
OVERLAP_SENTENCES = 1


def split_sentences(text):
    # Simple sentence splitter tuned for financial text (handles "U.S." "$10.9" etc.
    # reasonably well by requiring a capital letter or newline after the period).
    text = text.replace("\n", " ")
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(text, target_words=TARGET_WORDS, overlap=OVERLAP_SENTENCES):
    sentences = split_sentences(text)
    chunks = []
    current = []
    current_word_count = 0

    for sent in sentences:
        word_count = len(sent.split())
        if current_word_count + word_count > target_words and current:
            chunks.append(" ".join(current))
            # keep last `overlap` sentences for continuity
            current = current[-overlap:] if overlap > 0 else []
            current_word_count = sum(len(s.split()) for s in current)
        current.append(sent)
        current_word_count += word_count

    if current:
        chunks.append(" ".join(current))

    return chunks


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_chunks = []

    for filepath in sorted(RAW_DIR.glob("*.txt")):
        company = filepath.stem.split("_")[0]  # e.g. AAPL, MSFT, AMZN
        text = filepath.read_text()

        # Split by section headers (lines in ALL CAPS starting with ITEM) so we
        # don't accidentally merge unrelated sections (e.g. Risk Factors bleeding
        # into MD&A) into one chunk.
        sections = re.split(r'\n(?=ITEM \d)', text)

        chunk_idx = 0
        for section in sections:
            section = section.strip()
            if not section:
                continue
            # crude section title = first line
            section_title = section.split("\n")[0][:80]
            section_body = "\n".join(section.split("\n")[1:]) if "\n" in section else section

            for chunk in chunk_text(section_body):
                if len(chunk.split()) < 15:
                    continue  # skip tiny fragments
                all_chunks.append({
                    "chunk_id": f"{company}_{chunk_idx}",
                    "company": company,
                    "section": section_title,
                    "source_file": filepath.name,
                    "text": chunk,
                })
                chunk_idx += 1

    with open(OUT_PATH, "w") as f:
        for c in all_chunks:
            f.write(json.dumps(c) + "\n")

    print(f"Wrote {len(all_chunks)} chunks to {OUT_PATH}")
    by_company = {}
    for c in all_chunks:
        by_company[c["company"]] = by_company.get(c["company"], 0) + 1
    for company, count in by_company.items():
        print(f"  {company}: {count} chunks")


if __name__ == "__main__":
    main()
