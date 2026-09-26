"""RAG pipeline: ingest → embed → retrieve → generate with citations."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from sentence_transformers import SentenceTransformer

from config import Config, load_config
from document_loader import load_document_structured
from text_splitter import chunk_document, chunk_pages
from vector_store import SimpleVectorStore


class RAGPipeline:
    def __init__(self, config: Optional[Config] = None, api_key: Optional[str] = None):
        if config is None:
            if api_key:
                os.environ.setdefault("GEMINI_API_KEY", api_key)
            config = load_config()

        self.config = config
        self.client = genai.Client(api_key=config.gemini_api_key)
        self.vector_store = SimpleVectorStore()
        self._file_hashes: Dict[str, str] = {}

        print(
            f"Loading embedding model '{config.embedding_model}' "
            "(downloads once on first run)..."
        )
        self.embedding_model = SentenceTransformer(config.embedding_model)
        self.generation_model = config.generation_model

    def ingest_document(self, filepath: str, force: bool = False) -> int:
        """Load, chunk, embed, and store a document. Returns number of chunks added."""
        abs_path = os.path.abspath(filepath)
        file_hash = self._hash_file(abs_path)

        if not force and self._file_hashes.get(abs_path) == file_hash:
            print(f"Skipping unchanged file: {filepath}")
            return 0

        print(f"Loading {filepath}...")
        content = load_document_structured(filepath)

        print(f"Chunking {filepath}...")
        if isinstance(content, list):
            chunks = chunk_pages(
                content,
                source=abs_path,
                chunk_size=self.config.chunk_size,
                overlap=self.config.chunk_overlap,
            )
        else:
            chunks = chunk_document(
                content,
                source=abs_path,
                chunk_size=self.config.chunk_size,
                overlap=self.config.chunk_overlap,
            )

        if not chunks:
            print(f"No text extracted from {filepath}.")
            return 0

        # Replace previous chunks for this source so re-ingest stays clean.
        removed = self.vector_store.remove_by_source(abs_path)
        if removed:
            print(f"Replaced {removed} existing chunks for this file.")

        print(f"Embedding {len(chunks)} chunks...")
        embeddings = self._get_embeddings([c["text"] for c in chunks])
        self.vector_store.add_chunks(chunks, embeddings)
        self._file_hashes[abs_path] = file_hash
        print(f"Ingested {len(chunks)} chunks from {os.path.basename(filepath)}.\n")
        return len(chunks)

    def _hash_file(self, filepath: str) -> str:
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                h.update(block)
        return h.hexdigest()

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=len(texts) > 20,
            batch_size=32,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        top_k = top_k if top_k is not None else self.config.top_k
        min_score = min_score if min_score is not None else self.config.min_similarity
        query_embedding = self._get_embeddings([query])[0]
        return self.vector_store.search(query_embedding, top_k=top_k, min_score=min_score)

    def generate_answer(
        self,
        query: str,
        top_k: Optional[int] = None,
        show_context: bool = False,
    ) -> str:
        results = self.retrieve(query, top_k=top_k)

        if not results:
            if len(self.vector_store) == 0:
                return "I don't have any documents ingested to answer this question."
            return (
                "I found documents, but nothing similar enough to your question. "
                "Try rephrasing, or lower MIN_SIMILARITY."
            )

        context_parts = []
        citations = []
        for i, (chunk, score) in enumerate(results, start=1):
            meta = chunk["metadata"]
            source = os.path.basename(meta["source"])
            chunk_id = meta["chunk_id"]
            page = meta.get("page")
            location = f"page {page}" if page is not None else f"chunk {chunk_id}"

            context_parts.append(
                f"[Citation {i}] Source: {source} ({location})\n{chunk['text']}"
            )
            citations.append(
                f"[{i}] {source} ({location}) — similarity {score:.3f}"
            )

        context_str = "\n\n".join(context_parts)
        prompt = f"""You are a careful research assistant. Answer the question using ONLY the context below.
Rules:
- Base every claim on the context. If the context is insufficient, say you cannot answer from the documents.
- Cite sources inline like [1], [2] when you use them.
- Prefer concise, factual answers. Do not invent details.

Context:
{context_str}

Question: {query}

Answer:"""

        response = self.client.models.generate_content(
            model=self.generation_model,
            contents=prompt,
        )
        answer = (response.text or "").strip()

        sections = [answer, "", "Sources:", *citations]
        if show_context:
            sections.extend(["", "Retrieved context:", context_str])
        return "\n".join(sections)

    def save_index(self, path: Optional[str] = None) -> None:
        path = path or self.config.index_path
        self.vector_store.save(path)
        hash_path = path + ".hashes.json"
        with open(hash_path, "w", encoding="utf-8") as f:
            json.dump(self._file_hashes, f)
        print(f"Index saved to {path} ({len(self.vector_store)} chunks).")

    def load_index(self, path: Optional[str] = None) -> bool:
        path = path or self.config.index_path
        ok = self.vector_store.load(path)
        if not ok:
            return False

        hash_path = path + ".hashes.json"
        if os.path.exists(hash_path):
            with open(hash_path, "r", encoding="utf-8") as f:
                self._file_hashes = json.load(f)
        print(f"Loaded index with {len(self.vector_store)} chunks from {path}.")
        return True
