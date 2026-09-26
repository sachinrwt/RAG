"""Central configuration loaded from environment variables."""

import os
from dataclasses import dataclass


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None else default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value is not None else default


@dataclass(frozen=True)
class Config:
    gemini_api_key: str
    generation_model: str = "gemini-3.6-flash"
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 4
    min_similarity: float = 0.25
    docs_dir: str = "docs"
    index_path: str = "data/index.npz"


def load_config() -> Config:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. Create a .env file with "
            "GEMINI_API_KEY=your_key or set the environment variable."
        )

    return Config(
        gemini_api_key=api_key,
        generation_model=os.environ.get("GEMINI_MODEL", "gemini-3.6-flash"),
        embedding_model=os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        chunk_size=_env_int("CHUNK_SIZE", 1000),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 200),
        top_k=_env_int("TOP_K", 4),
        min_similarity=_env_float("MIN_SIMILARITY", 0.25),
        docs_dir=os.environ.get("DOCS_DIR", "docs"),
        index_path=os.environ.get("INDEX_PATH", "data/index.npz"),
    )
