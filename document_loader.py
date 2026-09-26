"""Load PDF / TXT / DOCX files into plain text or page-aware structures."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Union

from docx import Document
from pypdf import PdfReader


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Fix common PDF extraction artifacts: hyphenated line breaks
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_txt(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return _normalize_whitespace(f.read())


def load_pdf(filepath: str) -> str:
    pages = load_pdf_pages(filepath)
    return "\n\n".join(p["text"] for p in pages if p["text"])


def load_pdf_pages(filepath: str) -> List[Dict[str, Any]]:
    reader = PdfReader(filepath)
    pages: List[Dict[str, Any]] = []
    for i, page in enumerate(reader.pages):
        extracted = page.extract_text() or ""
        cleaned = _normalize_whitespace(extracted)
        if cleaned:
            pages.append({"page": i + 1, "text": cleaned})
    return pages


def load_docx(filepath: str) -> str:
    doc = Document(filepath)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return _normalize_whitespace("\n\n".join(paragraphs))


def load_document(filepath: str) -> str:
    """Return full document text for the given file path."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".txt":
        return load_txt(filepath)
    if ext == ".pdf":
        return load_pdf(filepath)
    if ext == ".docx":
        return load_docx(filepath)
    raise ValueError(f"Unsupported file extension: {ext}")


def load_document_structured(filepath: str) -> Union[str, List[Dict[str, Any]]]:
    """
    Load a document preserving page structure for PDFs.

    Returns a list of {page, text} for PDFs, or a plain string otherwise.
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        return load_pdf_pages(filepath)
    return load_document(filepath)
