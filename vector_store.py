"""In-memory vector store with numpy matrices and optional disk persistence."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class SimpleVectorStore:
    def __init__(self):
        self._embeddings: Optional[np.ndarray] = None  # shape (n, dim)
        self.metadata: List[Dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self.metadata)

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must be equal.")
        if not chunks:
            return

        new_matrix = np.asarray(embeddings, dtype=np.float32)
        if self._embeddings is None:
            self._embeddings = new_matrix
        else:
            self._embeddings = np.vstack([self._embeddings, new_matrix])
        self.metadata.extend(chunks)

    def remove_by_source(self, source: str) -> int:
        """Remove all chunks whose metadata.source matches. Returns count removed."""
        if not self.metadata:
            return 0

        keep_idx = [
            i
            for i, chunk in enumerate(self.metadata)
            if chunk.get("metadata", {}).get("source") != source
        ]
        removed = len(self.metadata) - len(keep_idx)
        if removed == 0:
            return 0

        if not keep_idx:
            self._embeddings = None
            self.metadata = []
            return removed

        self._embeddings = self._embeddings[keep_idx]
        self.metadata = [self.metadata[i] for i in keep_idx]
        return removed

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 3,
        min_score: float = 0.0,
    ) -> List[Tuple[Dict[str, Any], float]]:
        if self._embeddings is None or len(self.metadata) == 0:
            return []

        query_vec = np.asarray(query_embedding, dtype=np.float32)
        store_vecs = self._embeddings

        # Cosine similarity via normalized vectors
        query_norm = np.linalg.norm(query_vec) + 1e-10
        store_norms = np.linalg.norm(store_vecs, axis=1) + 1e-10
        similarities = (store_vecs @ query_vec) / (store_norms * query_norm)

        k = min(top_k, len(similarities))
        # argpartition is faster than full sort for large n
        candidate_idx = np.argpartition(similarities, -k)[-k:]
        ranked = candidate_idx[np.argsort(similarities[candidate_idx])[::-1]]

        results: List[Tuple[Dict[str, Any], float]] = []
        for idx in ranked:
            score = float(similarities[idx])
            if score < min_score:
                continue
            results.append((self.metadata[int(idx)], score))
        return results

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        meta_path = path + ".meta.json"
        if self._embeddings is None:
            np.savez_compressed(path, embeddings=np.empty((0, 0), dtype=np.float32))
        else:
            np.savez_compressed(path, embeddings=self._embeddings)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False)

    def load(self, path: str) -> bool:
        meta_path = path + ".meta.json"
        if not os.path.exists(path) or not os.path.exists(meta_path):
            return False

        data = np.load(path)
        embeddings = data["embeddings"]
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        if embeddings.size == 0 or not metadata:
            self._embeddings = None
            self.metadata = []
            return True

        self._embeddings = embeddings.astype(np.float32)
        self.metadata = metadata
        return True

    def clear(self) -> None:
        self._embeddings = None
        self.metadata = []
