"""
RAG Pipeline: retrieval + generation.
Orchestrates chunking -> retrieval -> document reader -> answer.
"""

from pathlib import Path
from typing import List, Optional, Tuple

from .data.chunking import BaseChunker, FixedSizeChunker
from .data.loaders import load_document, load_leaderboard_queries
from .generation.reader import ExtractiveReader, SimpleReader, _get_qa_device
from .retrieval.dense_retriever import DenseRetriever
from .retrieval.sparse_retriever import SparseRetriever
from .retrieval.hybrid_retriever import HybridRetriever


class RAGPipeline:
    """
    End-to-end RAG pipeline.
    """

    def __init__(
        self,
        knowledge_paths: List[str],
        chunker: Optional[BaseChunker] = None,
        retrieval_mode: str = "hybrid",  # "dense", "sparse", "hybrid"
        reader_type: str = "generative",  # "extractive", "generative", "simple"
        top_k: int = 5,
        index_dir: Optional[str] = None,
        use_gpu: bool = False,
        closed_book: bool = False,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        extractive_model: str = "deepset/roberta-base-squad2",
        generative_model: str = "mistralai/Mistral-7B-Instruct-v0.2",
    ):
        self.chunker = chunker or FixedSizeChunker(chunk_size=512, overlap=50)
        self.retrieval_mode = retrieval_mode
        self.top_k = top_k
        self.closed_book = closed_book
        self.index_dir = Path(index_dir) if index_dir else None

        # Load and chunk documents (skip for closed-book)
        self.documents: List[str] = []
        if not closed_book:
            for path in knowledge_paths:
                p = Path(path)
                if p.is_file():
                    text = load_document(str(p))
                    self.documents.extend(self.chunker.chunk(text))
                elif p.is_dir():
                    for f in p.rglob("*"):
                        if f.suffix.lower() in (".txt", ".html", ".htm", ".pdf"):
                            try:
                                text = load_document(str(f))
                                self.documents.extend(self.chunker.chunk(text))
                            except Exception as e:
                                print(f"Skip {f}: {e}")

        # Build retrievers (skip for closed-book)
        if not closed_book:
            self.dense = DenseRetriever(use_gpu=use_gpu, model_name=embedding_model)
            self.sparse = SparseRetriever()
            self.dense.build_index(self.documents, use_flat=len(self.documents) < 10000)
            self.sparse.build_index(self.documents)
            if retrieval_mode == "hybrid":
                self.retriever = HybridRetriever(self.dense, self.sparse, fusion="rrf")
            elif retrieval_mode == "dense":
                self.retriever = self.dense
            else:
                self.retriever = self.sparse
        else:
            self.retriever = None
            self.dense = None
            self.sparse = None

        # Reader
        if reader_type == "extractive":
            self.reader = ExtractiveReader(model_name=extractive_model, device=_get_qa_device(use_gpu))
        elif reader_type == "simple":
            self.reader = SimpleReader()
        else:
            from .generation.reader import GenerativeReader
            self.reader = GenerativeReader(model_name=generative_model, use_4bit=use_gpu)

    def retrieve(self, query: str) -> List[Tuple[str, float]]:
        if self.retriever is None:
            return []
        return self.retriever.retrieve(query, k=self.top_k)

    def answer(self, question: str) -> str:
        if self.closed_book:
            return self.reader.answer(question, "")
        results = self.retrieve(question)
        # Separate chunks clearly so the reader treats them as distinct passages (reduces stitching)
        context = "\n\n---\n\n".join(doc for doc, _ in results)
        return self.reader.answer(question, context)

    def run_on_queries(
        self,
        queries_path: str,
        output_path: str,
        format: str = "leaderboard",  # "leaderboard" or "test"
        andrew_id: str = "YOUR_ANDREW_ID",
    ) -> dict:
        """
        Run pipeline on queries file and save results.
        format: "leaderboard" includes andrewid; "test" is id -> answer only.
        """
        queries = load_leaderboard_queries(queries_path)
        results = {}
        if format == "leaderboard":
            results["andrewid"] = andrew_id
        for question, qid in queries:
            ans = self.answer(question)
            results[str(qid)] = ans
        import json
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        return results
