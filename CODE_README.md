# Code Overview: ANLP HW2 RAG System

This repository implements a **Retrieval-Augmented Generation (RAG)** system for Pittsburgh/CMU question-answering, as required by CMU Advanced NLP Assignment 2. It provides end-to-end document chunking, dense/sparse/hybrid retrieval, and extractive or generative readers.

---

## Directory Structure

```
anlp-spring2026-hw2/
├── run_rag.py              # Main entry: run RAG pipeline from CLI
├── run_all_configs.sh      # Batch experiments (baseline + ablations)
├── test_run_rag.py         # Tests for run_rag.py (simple/extractive, output format)
├── test_queries.json       # Small query set for tests
├── leaderboard_queries.json # Leaderboard query set (if provided)
├── requirements.txt        # Python dependencies
│
├── src/
│   ├── rag_pipeline.py     # End-to-end RAG: chunk → retrieve → read → answer
│   ├── data/
│   │   ├── chunking.py     # FixedSizeChunker, SentenceAwareChunker, ParagraphChunker
│   │   └── loaders.py      # load_document (.txt/.html/.pdf), load_leaderboard_queries
│   ├── retrieval/
│   │   ├── embedder.py     # Sentence-transformers embedding
│   │   ├── dense_retriever.py   # FAISS dense retrieval
│   │   ├── sparse_retriever.py  # BM25 sparse retrieval
│   │   └── hybrid_retriever.py  # RRF fusion of dense + sparse
│   ├── generation/
│   │   └── reader.py      # ExtractiveReader, GenerativeReader, SimpleReader
│   └── evaluation/
│       └── metrics.py     # F1, EM, ROUGE-L (for local eval)
│
├── scripts/
│   └── build_knowledge.py  # Build knowledge from raw HTML/PDF → data/knowledge/
│
├── data/
│   ├── raw/               # Raw sources (HTML, PDF, etc.) — optional
│   └── knowledge/         # Processed docs (.txt, .html, .pdf) used by RAG
│
└── system_outputs/        # Default output directory for predictions
```

---

## Setup

### 1. Install dependencies

From the project root:

```bash
pip install -r requirements.txt
```

For **GPU** (faster embedding, FAISS, and readers on Colab or a machine with CUDA):

```bash
pip install faiss-gpu   # optional; use faiss-cpu otherwise
```

### 2. Prepare the knowledge base

- **Option A:** Place processed documents (`.txt`, `.html`, `.pdf`) directly under `data/knowledge/` (or in subdirectories). The pipeline will discover them recursively.
- **Option B:** Put raw HTML/PDF in `data/raw/` and run:

  ```bash
  python scripts/build_knowledge.py
  ```

  This produces (or updates) `data/knowledge/` with cleaned text.

---

## Running the RAG Pipeline

### Main entry: `run_rag.py`

All runs are done from the **project root**:

```bash
python run_rag.py --queries <queries.json> --output <output.json> [options]
```

**Frequently used options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--knowledge` | `data/knowledge` | Path to knowledge directory or file |
| `--queries` | `leaderboard_queries.json` | Path to query JSON file |
| `--output` | `system_outputs/system_output_1.json` | Output JSON path |
| `--retrieval` | `hybrid` | `dense` \| `sparse` \| `hybrid` |
| `--reader` | `generative` | `extractive` \| `generative` \| `simple` |
| `--chunker` | `fixed` | `fixed` \| `sentence` \| `paragraph` |
| `--top-k` | `5` | Number of retrieved chunks per query |
| `--andrew-id` | `Porygon` | Andrew ID for leaderboard submissions |
| `--format` | `leaderboard` | `leaderboard` (with andrewid) \| `test` |
| `--use-gpu` | off | Use GPU for embedder, FAISS, and readers |
| `--closed-book` | off | No retrieval; reader gets empty context |
| `--embedding-model` | `sentence-transformers/all-MiniLM-L6-v2` | Dense embedding model |
| `--extractive-model` | `deepset/roberta-base-squad2` | Extractive QA model |
| `--generative-model` | `mistralai/Mistral-7B-Instruct-v0.2` | Generative LLM |

**Examples:**

```bash
# Default: hybrid retrieval + generative reader
python run_rag.py --queries leaderboard_queries.json --output system_outputs/out.json

# Leaderboard submission with your Andrew ID
python run_rag.py --andrew-id YOUR_ANDREW_ID --queries leaderboard_queries.json \
  --output system_outputs/leaderboard_submit.json --format leaderboard

# GPU, BGE embedding, generative reader
python run_rag.py --use-gpu --retrieval hybrid --reader generative \
  --embedding-model BAAI/bge-base-en-v1.5 --output system_outputs/bge_hybrid.json

# Dense-only + extractive reader
python run_rag.py --retrieval dense --reader extractive --output system_outputs/dense_extractive.json

# Sentence chunking
python run_rag.py --chunker sentence --output system_outputs/sentence_chunk.json

# Closed-book (no retrieval)
python run_rag.py --reader generative --closed-book --output system_outputs/closedbook.json
```

---

## Batch Experiments: `run_all_configs.sh`

The script runs a fixed set of configurations (baseline + ablations) with GPU and writes results under `system_outputs/`.

From project root:

```bash
bash run_all_configs.sh
```

Configurations included:

- **Baseline:** MiniLM + hybrid + generative (Mistral)
- **1:** Extractive reader (MiniLM + hybrid)
- **2:** BGE embedding + hybrid + generative
- **3:** MiniLM + hybrid + Qwen2 generative
- **4:** Dense-only and sparse-only (generative), in addition to hybrid baseline
- **5:** Closed-book extractive and closed-book generative
- **6:** Sentence chunking (MiniLM + hybrid + generative)
- **7:** Paragraph chunking (MiniLM + hybrid + generative)

---

## Testing

`test_run_rag.py` runs the pipeline with small configs and checks output format:

```bash
python test_run_rag.py
```

It uses `test_queries.json` and writes to `system_outputs/test_output.json`. Tests include:

- `--reader simple` (fast, no model download)
- Leaderboard format (presence of `andrewid`, question IDs as keys, string answers)
- `--format test` (no `andrewid`)
- Optional extractive reader run (slower; may be skipped on timeout)

---

## Query and Output Formats

**Query file** (e.g. `leaderboard_queries.json`, `test_queries.json`):

```json
[
  {"question": "When was Carnegie Mellon University founded?", "id": "1"},
  {"question": "What is the origin of Pittsburgh's name?", "id": "2"}
]
```

**Leaderboard output** (`--format leaderboard`):

```json
{
  "andrewid": "YOUR_ANDREW_ID",
  "1": "Answer 1",
  "2": "Answer 2"
}
```

**Test output** (`--format test`):

```json
{
  "1": "Answer 1",
  "2": "Answer 2; Answer 3"
}
```

---

## Implementation Summary

- **Chunking:** `FixedSizeChunker` (configurable size/overlap), `SentenceAwareChunker`, `ParagraphChunker` — select via `--chunker fixed|sentence|paragraph`.
- **Dense retrieval:** sentence-transformers + FAISS (CPU or GPU with `--use-gpu`).
- **Sparse retrieval:** BM25 (`rank-bm25`).
- **Hybrid:** Reciprocal Rank Fusion (RRF) of dense and sparse rankings.
- **Readers:** Extractive (RoBERTa-SQuAD), generative (Mistral/Qwen2, optional 4-bit with GPU), and a simple placeholder reader for quick tests.

For full assignment details, data policy, and report requirements, see the main **README.md**.
