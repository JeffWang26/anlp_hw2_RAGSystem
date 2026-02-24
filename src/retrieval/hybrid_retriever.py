"""
Hybrid retrieval: combines dense and sparse retrieval.
Implements fusion strategies: Reciprocal Rank Fusion (RRF), weighted score fusion.
"""

from collections import defaultdict
from typing import List, Optional, Tuple

from .dense_retriever import DenseRetriever
from .sparse_retriever import SparseRetriever


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[str, float]]],
    k: int = 60,
) -> List[Tuple[str, float]]:
    """
    Reciprocal Rank Fusion (RRF): score(d) = sum 1/(k + rank(d)).
    Lower rank (higher position) gives higher score.
    ranked_lists: list of retrieval result lists, each as [(doc, score), ...] sorted by relevance.
    """
    doc_scores = defaultdict(float)
    for rl in ranked_lists:
        for rank, (doc, _) in enumerate(rl, start=1):
            doc_scores[doc] += 1.0 / (k + rank)
    sorted_docs = sorted(doc_scores.items(), key=lambda x: -x[1])
    return sorted_docs


def min_max_normalize(scores: List[float]) -> List[float]:
    """Min-max normalization to [0, 1]."""
    if not scores:
        return scores
    mn, mx = min(scores), max(scores)
    if mx == mn:
        return [1.0] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]


def weighted_fusion(
    dense_results: List[Tuple[str, float]],
    sparse_results: List[Tuple[str, float]],
    alpha: float = 0.5,
) -> List[Tuple[str, float]]:
    """
    Weighted score fusion after min-max normalization.
    alpha: weight for dense (1-alpha for sparse).
    """
    dense_scores = {doc: s for doc, s in dense_results}
    sparse_scores = {doc: s for doc, s in sparse_results}
    all_docs = set(dense_scores.keys()) | set(sparse_scores.keys())
    d_vals = [dense_scores.get(d, 0) for d in all_docs]
    s_vals = [sparse_scores.get(d, 0) for d in all_docs]
    d_norm = min_max_normalize(d_vals)
    s_norm = min_max_normalize(s_vals)
    doc_to_idx = {d: i for i, d in enumerate(all_docs)}
    combined = []
    for doc in all_docs:
        i = doc_to_idx[doc]
        score = alpha * d_norm[i] + (1 - alpha) * s_norm[i]
        combined.append((doc, score))
    combined.sort(key=lambda x: -x[1])
    return combined


class HybridRetriever:
    """
    Hybrid retriever combining dense and sparse retrieval.
    Supports RRF and weighted score fusion.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        fusion: str = "rrf",  # "rrf" or "weighted"
        rrf_k: int = 60,
        alpha: float = 0.5,
    ):
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.fusion = fusion
        self.rrf_k = rrf_k
        self.alpha = alpha

    def retrieve(
        self,
        query: str,
        k: int = 5,
        dense_k: Optional[int] = None,
        sparse_k: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """
        Retrieve using hybrid strategy.
        Fetches more from each retriever (dense_k, sparse_k) then fuses to top k.
        """
        dk = dense_k or k * 2
        sk = sparse_k or k * 2
        dense_results = self.dense.retrieve(query, k=dk, return_scores=True)
        sparse_results = self.sparse.retrieve(query, k=sk, return_scores=True)
        if self.fusion == "rrf":
            fused = reciprocal_rank_fusion(
                [dense_results, sparse_results],
                k=self.rrf_k,
            )
        else:
            fused = weighted_fusion(dense_results, sparse_results, alpha=self.alpha)
        return fused[:k]
