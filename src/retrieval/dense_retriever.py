"""
Dense retrieval using FAISS for vector similarity search.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np

from .embedder import Embedder, _get_device


class DenseRetriever:
    """
    Dense retriever: embed documents, build FAISS index, retrieve by similarity.
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_gpu: bool = False,
        device: Optional[str] = None,
    ):
        if device is None:
            device = _get_device(use_gpu)
        self.embedder = embedder or Embedder(model_name=model_name, device=device)
        self.use_gpu = use_gpu
        self.index = None
        self.documents: List[str] = []
        self.doc_ids: List[str] = []  # Optional: document source IDs

    def build_index(
        self,
        documents: List[str],
        doc_ids: Optional[List[str]] = None,
        nlist: int = 100,
        use_flat: bool = False,
    ) -> None:
        """
        Embed documents and build FAISS index.
        Args:
            documents: List of document/chunk texts.
            doc_ids: Optional IDs for each document.
            nlist: Number of clusters for IVF (ignored if use_flat=True).
            use_flat: If True, use exact search (IndexFlatIP); else IVF for large corpora.
        """
        self.documents = documents
        self.doc_ids = doc_ids or [str(i) for i in range(len(documents))]
        embeddings = self.embedder.encode(documents, show_progress=True)
        embeddings = np.array(embeddings, dtype=np.float32)
        # Normalize for cosine similarity (inner product = cosine when normalized)
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        if use_flat or len(documents) < 1000:
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(embeddings)
        else:
            nlist = min(nlist, len(documents) // 10)
            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.train(embeddings)
            self.index.add(embeddings)
        if self.use_gpu and hasattr(faiss, "StandardGpuResources"):
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

    def retrieve(
        self,
        query: str,
        k: int = 5,
        return_scores: bool = True,
    ) -> List[Tuple[str, float]]:
        """
        Retrieve top-k documents for a query.
        Returns list of (doc_text, score) sorted by relevance.
        """
        if self.index is None:
            raise RuntimeError("Call build_index() first")
        q_embed = self.embedder.encode(query)
        q_embed = np.array(q_embed, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(q_embed)
        scores, indices = self.index.search(q_embed, min(k, len(self.documents)))
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0:
                continue
            doc_text = self.documents[idx]
            score = float(scores[0][i]) if return_scores else 0.0
            results.append((doc_text, score))
        return results

    def save_index(self, path: str) -> None:
        """Save FAISS index and document list."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        import json
        with open(path / "documents.json", "w", encoding="utf-8") as f:
            json.dump({"docs": self.documents, "ids": self.doc_ids}, f, ensure_ascii=False)

    def load_index(self, path: str) -> None:
        """Load FAISS index and document list."""
        path = Path(path)
        self.index = faiss.read_index(str(path / "index.faiss"))
        import json
        with open(path / "documents.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        self.documents = data["docs"]
        self.doc_ids = data.get("ids", [str(i) for i in range(len(self.documents))])
