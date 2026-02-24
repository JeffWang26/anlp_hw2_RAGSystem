"""
Sparse retrieval using BM25.
Uses rank_bm25 (or bm25s) as specified in README.
"""

import re
from typing import List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi


def simple_tokenizer(text: str) -> List[str]:
    """Simple whitespace tokenizer; can be replaced with NLTK for better tokenization."""
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return text.split()


class SparseRetriever:
    """
    Sparse retriever using BM25.
    """

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer or simple_tokenizer
        self.bm25 = None
        self.documents: List[str] = []
        self.doc_ids: List[str] = []

    def build_index(
        self,
        documents: List[str],
        doc_ids: Optional[List[str]] = None,
    ) -> None:
        """Build BM25 index from documents."""
        self.documents = documents
        self.doc_ids = doc_ids or [str(i) for i in range(len(documents))]
        tokenized = [self.tokenizer(d) for d in documents]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(
        self,
        query: str,
        k: int = 5,
        return_scores: bool = True,
    ) -> List[Tuple[str, float]]:
        """
        Retrieve top-k documents for a query.
        Returns list of (doc_text, score) sorted by BM25 score.
        """
        if self.bm25 is None:
            raise RuntimeError("Call build_index() first")
        q_tokens = self.tokenizer(query)
        scores = self.bm25.get_scores(q_tokens)
        top_indices = np.argsort(scores)[::-1][:k]
        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            doc_text = self.documents[idx]
            score = float(scores[idx]) if return_scores else 0.0
            results.append((doc_text, score))
        return results
