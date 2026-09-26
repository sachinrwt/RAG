"""Recursive character splitter that prefers paragraph/sentence boundaries."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# Prefer larger structural breaks first, then finer ones.
_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks, preferring natural boundaries.

    Falls back to hard character cuts only when a piece is still too large.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be strictly positive")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = _clean_text(text)
    if not text:
        return []

    return _split_recursive(text, chunk_size, overlap, _SEPARATORS)


def _split_recursive(
    text: str,
    chunk_size: int,
    overlap: int,
    separators: List[str],
) -> List[str]:
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    separator = separators[0]
    next_separators = separators[1:]

    if separator == "":
        return _hard_split(text, chunk_size, overlap)

    parts = text.split(separator) if separator else list(text)
    chunks: List[str] = []
    current = ""

    for i, part in enumerate(parts):
        piece = part if i == len(parts) - 1 else part + separator
        candidate = current + piece

        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current.strip():
            chunks.append(current.strip())

        if len(piece) > chunk_size:
            chunks.extend(_split_recursive(piece, chunk_size, overlap, next_separators))
            current = ""
        else:
            # Keep overlap from the end of the previous chunk when possible.
            if chunks and overlap > 0:
                overlap_text = chunks[-1][-overlap:]
                current = overlap_text + piece
                if len(current) > chunk_size:
                    current = piece
            else:
                current = piece

    if current.strip():
        chunks.append(current.strip())

    return _merge_small_tail(chunks, chunk_size)


def _hard_split(text: str, chunk_size: int, overlap: int) -> List[str]:
    step = chunk_size - overlap
    return [text[i : i + chunk_size] for i in range(0, len(text), step)]


def _merge_small_tail(chunks: List[str], chunk_size: int) -> List[str]:
    if len(chunks) < 2:
        return chunks
    if len(chunks[-1]) >= chunk_size // 3:
        return chunks
    merged = chunks[-2] + " " + chunks[-1]
    if len(merged) <= chunk_size + chunk_size // 4:
        return chunks[:-2] + [merged.strip()]
    return chunks


def chunk_document(
    text: str,
    source: str,
    chunk_size: int = 1000,
    overlap: int = 200,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Split a document and attach metadata to each chunk."""
    text_chunks = split_text(text, chunk_size, overlap)
    base_meta = dict(extra_metadata or {})

    docs: List[Dict[str, Any]] = []
    for i, chunk in enumerate(text_chunks):
        meta = {
            "source": source,
            "chunk_id": i,
            **base_meta,
        }
        docs.append({"text": chunk, "metadata": meta})
    return docs


def chunk_pages(
    pages: List[Dict[str, Any]],
    source: str,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> List[Dict[str, Any]]:
    """
    Chunk page-aware document content.

    Each page dict should have 'page' (1-based) and 'text'.
    """
    docs: List[Dict[str, Any]] = []
    chunk_id = 0

    for page in pages:
        page_num = page.get("page")
        page_text = page.get("text", "")
        for piece in split_text(page_text, chunk_size, overlap):
            docs.append(
                {
                    "text": piece,
                    "metadata": {
                        "source": source,
                        "chunk_id": chunk_id,
                        "page": page_num,
                    },
                }
            )
            chunk_id += 1

    return docs
