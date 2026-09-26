"""Interactive CLI for the from-scratch RAG pipeline."""

from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import replace

from dotenv import load_dotenv

from config import load_config
from rag_pipeline import RAGPipeline


SUPPORTED = ("*.txt", "*.pdf", "*.docx")


def find_documents(docs_dir: str) -> list[str]:
    files: list[str] = []
    for pattern in SUPPORTED:
        files.extend(glob.glob(os.path.join(docs_dir, pattern)))
    return sorted(files)


def ensure_docs_dir(docs_dir: str) -> None:
    if os.path.exists(docs_dir):
        return
    os.makedirs(docs_dir)
    sample = os.path.join(docs_dir, "sample.txt")
    with open(sample, "w", encoding="utf-8") as f:
        f.write(
            "Gemini is a family of multimodal large language models developed by "
            "Google DeepMind, serving as the successor to LaMDA and PaLM 2. "
            "The Gemini family includes Gemini Ultra, Gemini Pro, and Gemini Nano."
        )
    print(f"Created '{docs_dir}/' with sample.txt. Add your own PDF/TXT/DOCX files there.")


def ingest_all(pipeline: RAGPipeline, docs_dir: str, force: bool = False) -> int:
    files = find_documents(docs_dir)
    if not files:
        print(f"No documents found in {docs_dir}/.")
        return 0

    total = 0
    for path in files:
        try:
            total += pipeline.ingest_document(path, force=force)
        except Exception as exc:
            print(f"Failed to ingest {path}: {exc}")
    return total


def chat_loop(pipeline: RAGPipeline, show_context: bool = False) -> None:
    print("\n=== Chat ===")
    print("Commands: exit | quit | rebuild | sources | help")
    while True:
        try:
            query = input("\nAsk a question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not query:
            continue
        lower = query.lower()
        if lower in {"exit", "quit"}:
            break
        if lower == "help":
            print("Ask anything about your documents. Type 'rebuild' to re-ingest.")
            continue
        if lower == "sources":
            sources = {
                os.path.basename(c["metadata"]["source"])
                for c in pipeline.vector_store.metadata
            }
            print("Indexed sources:", ", ".join(sorted(sources)) or "(none)")
            continue
        if lower == "rebuild":
            pipeline.vector_store.clear()
            pipeline._file_hashes.clear()
            ingest_all(pipeline, pipeline.config.docs_dir, force=True)
            pipeline.save_index()
            continue

        print("\nThinking...\n")
        try:
            answer = pipeline.generate_answer(query, show_context=show_context)
            print(answer)
        except Exception as exc:
            print(f"Error: {exc}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RAG From Scratch — ask questions grounded in your documents."
    )
    parser.add_argument("--docs", default=None, help="Documents folder (default: docs/)")
    parser.add_argument("--rebuild", action="store_true", help="Force re-embed all docs")
    parser.add_argument("--no-cache", action="store_true", help="Ignore saved index")
    parser.add_argument("--query", "-q", default=None, help="One-shot question (no chat loop)")
    parser.add_argument("--top-k", type=int, default=None, help="Number of chunks to retrieve")
    parser.add_argument("--show-context", action="store_true", help="Print retrieved chunks")
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)

    try:
        config = load_config()
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    overrides = {}
    if args.docs:
        overrides["docs_dir"] = args.docs
    if args.top_k:
        overrides["top_k"] = args.top_k
    if overrides:
        config = replace(config, **overrides)

    print("=== RAG From Scratch ===")
    pipeline = RAGPipeline(config=config)
    ensure_docs_dir(config.docs_dir)

    loaded = False
    if not args.rebuild and not args.no_cache:
        loaded = pipeline.load_index()

    if args.rebuild or not loaded or len(pipeline.vector_store) == 0:
        print("Ingesting documents...")
        added = ingest_all(pipeline, config.docs_dir, force=args.rebuild)
        if added == 0 and len(pipeline.vector_store) == 0:
            print("Nothing to index. Add files and try again.")
            return 1
        pipeline.save_index()
    else:
        # Pick up any new/changed files without full rebuild
        print("Checking for new or changed documents...")
        ingest_all(pipeline, config.docs_dir, force=False)
        pipeline.save_index()

    if args.query:
        print("\nThinking...\n")
        print(pipeline.generate_answer(args.query, show_context=args.show_context))
        return 0

    chat_loop(pipeline, show_context=args.show_context)
    return 0


if __name__ == "__main__":
    sys.exit(main())
