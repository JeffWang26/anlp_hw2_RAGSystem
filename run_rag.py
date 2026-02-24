#!/usr/bin/env python3
"""
Main entry point for RAG pipeline.
Usage:
  python run_rag.py --queries leaderboard_queries.json --output system_outputs/output.json
  python run_rag.py --knowledge data/knowledge/ --queries leaderboard_queries.json
"""

import argparse
from pathlib import Path

# Ensure project root is on path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.rag_pipeline import RAGPipeline
from src.data.chunking import FixedSizeChunker, SentenceAwareChunker, ParagraphChunker


def main():
    parser = argparse.ArgumentParser(description="Run RAG pipeline for Pittsburgh/CMU QA")
    parser.add_argument(
        "--knowledge",
        type=str,
        default="data/knowledge",
        help="Path to knowledge resource (directory or file).",
    )
    parser.add_argument(
        "--queries",
        type=str,
        default="leaderboard_queries.json",
        help="Path to queries JSON file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="system_outputs/system_output_1.json",
        help="Output path for predictions.",
    )
    parser.add_argument(
        "--retrieval",
        choices=["dense", "sparse", "hybrid"],
        default="hybrid",
        help="Retrieval mode.",
    )
    parser.add_argument(
        "--reader",
        choices=["extractive", "generative", "simple"],
        default="generative",
        help="Document reader type.",
    )
    parser.add_argument(
        "--chunker",
        choices=["fixed", "sentence", "paragraph"],
        default="fixed",
        help="Chunking strategy.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of documents to retrieve per query.",
    )
    parser.add_argument(
        "--andrew-id",
        type=str,
        default="Porygon",
        help="Andrew ID for submission.",
    )
    parser.add_argument(
        "--format",
        choices=["leaderboard", "test"],
        default="leaderboard",
        help="Output format: leaderboard (with andrewid) or test.",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU for embedding, FAISS, and reader (install faiss-gpu, bitsandbytes for full support).",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model for dense retrieval (e.g. BAAI/bge-base-en-v1.5).",
    )
    parser.add_argument(
        "--extractive-model",
        type=str,
        default="deepset/roberta-base-squad2",
        help="Model for extractive reader (used when --reader extractive).",
    )
    parser.add_argument(
        "--generative-model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Model for generative reader (e.g. Qwen/Qwen2-7B-Instruct).",
    )
    parser.add_argument(
        "--closed-book",
        action="store_true",
        help="Run reader without retrieval (empty context) for RAG vs closed-book comparison.",
    )
    args = parser.parse_args()

    # Chunker
    if args.chunker == "fixed":
        chunker = FixedSizeChunker(chunk_size=512, overlap=50)
    elif args.chunker == "sentence":
        chunker = SentenceAwareChunker(chunk_size=512)
    else:
        chunker = ParagraphChunker(max_chunk_size=512)

    # Knowledge paths
    knowledge_path = Path(args.knowledge)
    if knowledge_path.is_file():
        knowledge_paths = [str(knowledge_path)]
    elif knowledge_path.is_dir():
        knowledge_paths = [str(args.knowledge)]
    else:
        # Fallback: use a sample doc if no knowledge exists yet
        sample = Path("data/knowledge/sample.txt")
        if sample.exists():
            knowledge_paths = [str(sample)]
        else:
            print("Warning: No knowledge resource found. Create data/knowledge/ with .txt/.html/.pdf files.")
            # Create minimal sample for testing
            sample.parent.mkdir(parents=True, exist_ok=True)
            sample.write_text(
                "Carnegie Mellon University was founded in 1900. "
                "It was established by Andrew Carnegie. Pittsburgh is named after William Pitt."
            )
            knowledge_paths = [str(sample)]

    pipeline = RAGPipeline(
        knowledge_paths=knowledge_paths,
        chunker=chunker,
        retrieval_mode=args.retrieval,
        reader_type=args.reader,
        top_k=args.top_k,
        use_gpu=args.use_gpu,
        closed_book=args.closed_book,
        embedding_model=args.embedding_model,
        extractive_model=args.extractive_model,
        generative_model=args.generative_model,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    results = pipeline.run_on_queries(
        args.queries,
        args.output,
        format=args.format,
        andrew_id=args.andrew_id,
    )
    print(f"Saved {len([k for k in results if k != 'andrewid'])} predictions to {args.output}")


if __name__ == "__main__":
    main()
