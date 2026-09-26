# RAG From Scratch

Ask questions about your own PDF, TXT, and DOCX files. Documents are chunked, embedded locally with MiniLM, retrieved by cosine similarity, and answered by Gemini with citations.

```
docs/  →  load  →  chunk  →  embed (local)  →  vector store
                                                      ↓
query  →  retrieve top-k  →  Gemini (grounded prompt + citations)
```

## Setup

```powershell
cd RAG
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file:

```
GEMINI_API_KEY=your_key_here
```

Optional settings:

| Variable | Default | Meaning |
|---|---|---|
| `GEMINI_MODEL` | `gemini-3.6-flash` | Generation model |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local sentence-transformers model |
| `CHUNK_SIZE` | `1000` | Max characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K` | `4` | Chunks retrieved per question |
| `MIN_SIMILARITY` | `0.25` | Drop weak matches below this score |
| `DOCS_DIR` | `docs` | Folder to ingest |
| `INDEX_PATH` | `data/index.npz` | Saved embedding index |

## Run

Put files in `docs/`, then:

```powershell
python main.py
```

Useful flags:

```powershell
python main.py --rebuild              # re-embed everything
python main.py -q "Who was Einstein?" # one-shot question
python main.py --show-context         # print retrieved chunks
python main.py --docs my_folder --top-k 6
```

In the chat loop: `sources`, `rebuild`, `exit`.

## What improved

- **Smarter chunking** — prefers paragraphs/sentences over hard character cuts
- **PDF page citations** — answers can reference page numbers
- **Persistent index** — embeddings saved under `data/` so restarts skip re-embedding unchanged files
- **Similarity threshold** — low-relevance noise is filtered before generation
- **Cleaner re-ingest** — same file replaces old chunks; content hashes skip unchanged files
- **Better grounded prompt** — inline citation style + refusal when context is weak
- **CLI** — one-shot queries, rebuild, show-context

## Project layout

| File | Role |
|---|---|
| `main.py` | CLI entrypoint |
| `rag_pipeline.py` | Ingest / retrieve / generate |
| `document_loader.py` | PDF / TXT / DOCX extraction |
| `text_splitter.py` | Recursive text chunking |
| `vector_store.py` | Cosine search + disk save/load |
| `config.py` | Env-based settings |
