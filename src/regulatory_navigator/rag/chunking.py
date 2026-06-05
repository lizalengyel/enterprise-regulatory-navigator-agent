from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # pymupdf
from transformers import AutoTokenizer

# Configuration

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
CHUNK_SIZE = 450        # tokens — bge-base max is 512 incl. [CLS]/[SEP]
CHUNK_OVERLAP = 64      # tokens

# Map PDF stem → canonical regulatory document type
_DOC_TYPE_MAP: dict[str, str] = {
    "gdpr":    "GDPR",
    "dora":    "DORA",
    "nis2":    "NIS2",
    "ai_act":  "AI Act"
}

# Legal unit heading patterns
_ARTICLE_RE = re.compile(
    r"^(Article\s+\d+[\w]*"
    r"|Recital\s+\d+"
    r"|Chapter\s+[IVXLCDM\d]+"
    r"|Section\s+\d+"
    r"|Annex\s+[IVXLCDM\d]+)",
    re.IGNORECASE | re.MULTILINE,
)


# Data model

@dataclass
class Chunk:
    text: str
    doc_name: str           # e.g. "gdpr"
    document_type: str      # e.g. "GDPR", "DORA", "NIS2", "AI Act"
    page_start: int         # 0-indexed
    page_end: int
    article: str | None     # e.g. "Article 17"
    chunk_index: int        # position within the document
    token_count: int
    metadata: dict = field(default_factory=dict)


# Tokenizer  (lazy-loaded — downloading happens once, then cached by HF)

_tokenizer: AutoTokenizer | None = None


def get_tokenizer() -> AutoTokenizer:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    return _tokenizer


def count_tokens(text: str) -> int:
    return len(get_tokenizer().encode(text, add_special_tokens=False))


def token_split(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping token windows using the BGE tokenizer."""
    tok = get_tokenizer()
    token_ids = tok.encode(text, add_special_tokens=False)

    if len(token_ids) <= size:
        return [text]

    chunks: list[str] = []
    step = size - overlap
    for start in range(0, len(token_ids), step):
        window = token_ids[start : start + size]
        chunks.append(tok.decode(window, skip_special_tokens=True))
        if start + size >= len(token_ids):
            break
    return chunks


# PDF extraction

def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    doc = fitz.open(pdf_path)
    return [(i, page.get_text("text")) for i, page in enumerate(doc)]


def full_text_with_page_map(pages: list[tuple[int, str]]) -> tuple[str, list[tuple[int, int]]]:
    parts: list[str] = []
    page_map: list[tuple[int, int]] = []
    cursor = 0
    for _page_num, text in pages:
        start = cursor
        parts.append(text)
        cursor += len(text)
        page_map.append((start, cursor))
    return "".join(parts), page_map


def char_to_page(char_idx: int, page_map: list[tuple[int, int]]) -> int:
    for page_num, (start, end) in enumerate(page_map):
        if start <= char_idx < end:
            return page_num
    return len(page_map) - 1


# Structure-aware splitting

def split_on_legal_boundaries(text: str) -> list[tuple[str | None, str]]:
    matches = list(_ARTICLE_RE.finditer(text))
    if not matches:
        return [(None, text)]

    segments: list[tuple[str | None, str]] = []
    if matches[0].start() > 0:
        segments.append((None, text[: matches[0].start()]))

    for i, match in enumerate(matches):
        heading = match.group(0).strip()
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        segments.append((heading, text[body_start:body_end]))

    return segments


# Main entry points

def chunk_pdf(pdf_path: Path, doc_name: str | None = None) -> list[Chunk]:
    name = doc_name or pdf_path.stem.lower()
    document_type = _DOC_TYPE_MAP.get(name, name.upper())

    pages = extract_pages(pdf_path)
    full_text, page_map = full_text_with_page_map(pages)
    segments = split_on_legal_boundaries(full_text)

    chunks: list[Chunk] = []
    chunk_index = 0
    cursor = 0

    for heading, body in segments:
        body_stripped = body.strip()
        if not body_stripped:
            cursor += len(body)
            continue

        segment_text = f"{heading}\n{body_stripped}" if heading else body_stripped
        sub_chunks = token_split(segment_text)

        for sub_text in sub_chunks:
            page_start = char_to_page(cursor, page_map)
            page_end = char_to_page(cursor + len(body), page_map)

            chunks.append(Chunk(
                text=sub_text,
                doc_name=name,
                document_type=document_type,
                page_start=page_start,
                page_end=page_end,
                article=heading,
                chunk_index=chunk_index,
                token_count=count_tokens(sub_text),
            ))
            chunk_index += 1

        cursor += len(body)

    return chunks


def chunk_directory(raw_dir: Path) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for pdf_path in sorted(raw_dir.glob("*.pdf")):
        all_chunks.extend(chunk_pdf(pdf_path))
    return all_chunks