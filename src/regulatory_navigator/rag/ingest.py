from __future__ import annotations

import logging
from pathlib import Path

import chromadb

from .chunking import Chunk, chunk_pdf
from .embeddings import embed_texts

logger = logging.getLogger(__name__)

# Paths

RAW_DIR          = Path("data/raw")
CHROMA_DIR       = Path("data/vectorstore")
COLLECTION       = "regulatory_docs"
EMBED_BATCH_SIZE = 64


# Vector store helpers

def get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def delete_doc(collection: chromadb.Collection, doc_name: str) -> None:
    result = collection.get(where={"doc_name": doc_name}, include=[])
    if result["ids"]:
        collection.delete(ids=result["ids"])
        logger.debug("Deleted %d existing vectors for '%s'", len(result["ids"]), doc_name)


def upsert_chunks(collection: chromadb.Collection, chunks: list[Chunk]) -> None:
    for batch_start in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch      = chunks[batch_start : batch_start + EMBED_BATCH_SIZE]
        texts      = [c.text for c in batch]
        embeddings = embed_texts(texts)

        collection.upsert(
            ids=[f"{c.doc_name}__{c.chunk_index}" for c in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {
                    "doc_name":      c.doc_name,
                    "document_type": c.document_type,
                    "article":       c.article or "",
                    "page_start":    c.page_start,
                    "page_end":      c.page_end,
                    "chunk_index":   c.chunk_index,
                    "token_count":   c.token_count,
                }
                for c in batch
            ],
        )
        logger.debug("Upserted batch %d-%d", batch_start, batch_start + len(batch))


# Per-file ingestion

def ingest_pdf(pdf_path: Path, collection: chromadb.Collection) -> None:
    logger.info("  ingest  %s", pdf_path.name)
    chunks = chunk_pdf(pdf_path)
    logger.info("    chunked -> %d chunks", len(chunks))
    delete_doc(collection, pdf_path.stem.lower())
    upsert_chunks(collection, chunks)
    logger.info("    stored")


# Main entry point

def ingest_all(raw_dir: Path = RAW_DIR) -> None:
    """Ingest all PDFs in raw_dir into ChromaDB."""
    collection = get_collection()

    pdfs = sorted(raw_dir.glob("*.pdf"))
    if not pdfs:
        logger.warning("No PDFs found in %s", raw_dir)
        return

    logger.info("Found %d PDF(s) in %s", len(pdfs), raw_dir)
    for p in pdfs:
        ingest_pdf(p, collection)
    logger.info("Done — %d file(s) ingested.", len(pdfs))


# CLI

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(description="Ingest regulatory PDFs into ChromaDB.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args()

    ingest_all(raw_dir=args.raw_dir)
