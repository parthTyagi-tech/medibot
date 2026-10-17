import os
import sys
import json
import time
import argparse
from typing import List
from dotenv import load_dotenv

load_dotenv()

from research.src.helper import (
    load_pdf_files,
    filter_to_minimal_docs,
    text_split,
    download_embeddings
)
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medical-chatbot-e5")
EMBED_MODEL = os.getenv("EMBED_MODEL", "multilingual-e5-large")
EMBED_TOKEN_CAP = int(os.getenv("EMBED_TOKEN_CAP", "4500000"))
CHECKPOINT_PATH = os.path.join("data", "indexing_checkpoint.json")


def is_non_clinical_page(doc: Document) -> bool:
    """Detects index, bibliography, and organization directory pages."""
    content = doc.page_content.strip()
    if not content:
        return True
    first_lines = "\n".join(content.splitlines()[:5]).upper()
    non_clinical_headers = [
        "INDEX", "GENERAL INDEX", "ORGANIZATIONS", "DIRECTORY OF ORGANIZATIONS",
        "GLOSSARY", "BIBLIOGRAPHY", "GENERAL BIBLIOGRAPHY", "PHOTO CREDITS",
        "CONTRIBUTORS", "ADVISORY BOARD", "INTRODUCTION"
    ]
    return any(h in first_lines for h in non_clinical_headers)


def filter_clinical_pages(docs: List[Document], skip_backmatter: bool = True) -> List[Document]:
    """Filters out non-clinical backmatter/index pages while retaining clinical entries."""
    if not skip_backmatter:
        return docs
    filtered = []
    skipped_count = 0
    for doc in docs:
        if is_non_clinical_page(doc):
            skipped_count += 1
            continue
        filtered.append(doc)
    print(f"[Corpus] Retained {len(filtered)} clinical pages (skipped {skipped_count} index/glossary pages).")
    return filtered


def load_checkpoint() -> set:
    """Loads set of chunk indices already successfully embedded and upserted."""
    if os.path.exists(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return set(data.get("completed_indices", []))
        except Exception as e:
            print(f"[Checkpoint] Warning: could not read {CHECKPOINT_PATH}: {e}")
    return set()


def save_checkpoint(completed: set):
    """Saves completed chunk indices to checkpoint file."""
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    temp_path = CHECKPOINT_PATH + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump({"completed_indices": sorted(list(completed)), "count": len(completed)}, f)
    os.replace(temp_path, CHECKPOINT_PATH)


def run_dry_run_and_extrapolate(
    pc: Pinecone,
    chunks: List[Document],
    model: str,
    token_cap: int,
    sample_size: int = 50
) -> float:
    """
    Amendment 1: Runs a dry run on ~50 chunks, reads token usage from Pinecone,
    extrapolates to full corpus, and asserts against EMBED_TOKEN_CAP.
    """
    actual_sample_size = min(sample_size, len(chunks))
    sample_chunks = chunks[:actual_sample_size]
    sample_texts = [c.page_content for c in sample_chunks]

    print(f"\n{'='*60}")
    print(f"RUNNING DRY-RUN TOKEN PROBE ON {actual_sample_size} CHUNKS...")
    print(f"{'='*60}")

    res = pc.inference.embed(
        model=model,
        inputs=sample_texts,
        parameters={"input_type": "passage", "truncate": "END"}
    )

    total_sample_tokens = 0
    if hasattr(res, "usage") and isinstance(res.usage, dict):
        total_sample_tokens = res.usage.get("total_tokens", 0)
    elif hasattr(res, "usage"):
        total_sample_tokens = getattr(res.usage, "total_tokens", 0)

    avg_tokens_per_chunk = total_sample_tokens / actual_sample_size if actual_sample_size > 0 else 0
    extrapolated_tokens = int(avg_tokens_per_chunk * len(chunks))

    print(f"Sample size:              {actual_sample_size} chunks")
    print(f"Sample tokens used:       {total_sample_tokens:,} tokens")
    print(f"Average tokens/chunk:     {avg_tokens_per_chunk:.1f} tokens")
    print(f"Total corpus chunks:      {len(chunks):,}")
    print(f"Extrapolated token total: {extrapolated_tokens:,} tokens")
    print(f"Hard Token Cap:           {token_cap:,} tokens")
    print(f"Cap Headroom:             {(token_cap - extrapolated_tokens):,} tokens")
    print(f"{'='*60}\n")

    if extrapolated_tokens > token_cap:
        raise ValueError(
            f"Extrapolated token total ({extrapolated_tokens:,}) exceeds EMBED_TOKEN_CAP ({token_cap:,}). "
            "Please remove overlap or skip additional non-clinical pages."
        )

    return avg_tokens_per_chunk


def main():
    parser = argparse.ArgumentParser(description="Pinecone Hosted Inference Indexing Pipeline")
    parser.add_argument("--dry-run-only", action="store_true", help="Run token extrapolation probe and exit")
    parser.add_argument("--skip-backmatter", action="store_true", default=True, help="Skip index/glossary/bibliography pages")
    parser.add_argument("--no-overlap", action="store_true", help="Remove chunk overlap (chunk_overlap=0)")
    parser.add_argument("--max-chunks", type=int, default=0, help="Max chunks to index (0 for all)")
    args = parser.parse_args()

    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is missing from environment variables.")

    print(f"[Init] Initializing Pinecone client...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    embedding = download_embeddings()
    dim = getattr(embedding, "dimension", 1024)
    print(f"[Init] Probed model '{EMBED_MODEL}' dimension: {dim}D")

    # 1. Ensure Index Exists
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"[Index] Creating new index '{PINECONE_INDEX_NAME}' ({dim}D, cosine, AWS us-east-1)...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            print("[Index] Waiting for index creation to complete...")
            time.sleep(2)
        print(f"[Index] Index '{PINECONE_INDEX_NAME}' created successfully.")
    else:
        print(f"[Index] Using existing index '{PINECONE_INDEX_NAME}'.")

    desc = pc.describe_index(PINECONE_INDEX_NAME)
    index = pc.Index(PINECONE_INDEX_NAME, host=desc.host)

    # 2. Load and Filter Documents
    print(f"[Corpus] Loading PDF files from 'data'...")
    raw_docs = load_pdf_files("data")
    print(f"[Corpus] Loaded {len(raw_docs)} raw pages.")

    clinical_docs = filter_clinical_pages(raw_docs, skip_backmatter=args.skip_backmatter)
    minimal_docs = filter_to_minimal_docs(clinical_docs)

    overlap = 0 if args.no_overlap else 50
    print(f"[Chunking] Splitting text: chunk_size=1100, chunk_overlap={overlap}...")
    chunks = text_split(minimal_docs, chunk_size=1100, chunk_overlap=overlap)
    print(f"[Chunking] Generated {len(chunks)} text chunks.")

    if args.max_chunks > 0:
        chunks = chunks[:args.max_chunks]
        print(f"[Chunking] Limited to first {len(chunks)} chunks by --max-chunks.")

    # 3. Dry-Run Token Usage Extrapolation (Amendment 1)
    run_dry_run_and_extrapolate(
        pc=pc,
        chunks=chunks,
        model=EMBED_MODEL,
        token_cap=EMBED_TOKEN_CAP,
        sample_size=50
    )

    if args.dry_run_only:
        print("[DryRun] --dry-run-only specified. Exiting cleanly before full indexing.")
        return

    # 4. Resume from Checkpoint
    completed_indices = load_checkpoint()
    print(f"[Checkpoint] Found {len(completed_indices)} already indexed chunks in checkpoint.")

    remaining_indices = [i for i in range(len(chunks)) if i not in completed_indices]
    print(f"[Indexing] {len(remaining_indices)} chunks remaining to embed and upsert.")

    if not remaining_indices:
        print("[Indexing] All chunks are already indexed! Done.")
        return

    # 5. Paced Indexing Loop (Amendment 5)
    # Batch size <= 96 for inference; sleep ~10s per batch to respect TPM limits.
    # Batch size <= 50 for upserting.
    EMBED_BATCH_SIZE = 96
    UPSERT_BATCH_SIZE = 50

    for batch_start in range(0, len(remaining_indices), EMBED_BATCH_SIZE):
        batch_idx_chunk = remaining_indices[batch_start:batch_start + EMBED_BATCH_SIZE]
        texts_to_embed = [chunks[i].page_content for i in batch_idx_chunk]

        # Retry with exponential backoff on 429
        backoff = 10
        vectors_generated = None
        for attempt in range(5):
            try:
                res = pc.inference.embed(
                    model=EMBED_MODEL,
                    inputs=texts_to_embed,
                    parameters={"input_type": "passage", "truncate": "END"}
                )
                if hasattr(res, "data"):
                    vectors_generated = [item.values for item in res.data]
                break
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "rate limit" in err_str:
                    print(f"[RateLimit] 429 encountered. Backing off for {backoff}s...")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 60)
                else:
                    raise

        if not vectors_generated or len(vectors_generated) != len(texts_to_embed):
            raise RuntimeError(f"Failed to generate embeddings for batch starting at {batch_start}")

        # Prepare records with source and page metadata (Amendment 8)
        records = []
        for i, (chunk_idx, vec) in enumerate(zip(batch_idx_chunk, vectors_generated)):
            chunk = chunks[chunk_idx]
            records.append({
                "id": f"chunk-{chunk_idx}",
                "values": vec,
                "metadata": {
                    "source": chunk.metadata.get("source", "The Gale Encyclopedia of Medicine"),
                    "page": chunk.metadata.get("page", 0),
                    "text": chunk.page_content
                }
            })

        # Upsert in batches <= 50 (Amendment 5) with retry on transient network glitch
        for u in range(0, len(records), UPSERT_BATCH_SIZE):
            upsert_subbatch = records[u:u + UPSERT_BATCH_SIZE]
            upsert_backoff = 5
            for upsert_attempt in range(5):
                try:
                    index.upsert(vectors=upsert_subbatch)
                    break
                except Exception as ue:
                    if upsert_attempt < 4:
                        print(f"[Upsert] Notice: transient error ({ue}). Retrying in {upsert_backoff}s...")
                        time.sleep(upsert_backoff)
                        upsert_backoff = min(upsert_backoff * 2, 30)
                    else:
                        raise

        # Update checkpoint
        for chunk_idx in batch_idx_chunk:
            completed_indices.add(chunk_idx)
        save_checkpoint(completed_indices)

        progress = len(completed_indices) / len(chunks) * 100
        print(f"[Progress] Indexed {len(completed_indices)}/{len(chunks)} chunks ({progress:.1f}%). Pacing 10s...")
        time.sleep(10)  # TPM rate pacing

    print(f"\n[Complete] All {len(chunks)} chunks successfully indexed in '{PINECONE_INDEX_NAME}'.")


if __name__ == "__main__":
    main()