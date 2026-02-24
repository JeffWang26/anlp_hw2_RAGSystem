"""
Document and query embedder using sentence-transformers.
"""

from typing import List, Union

from sentence_transformers import SentenceTransformer


def _get_device(use_gpu: bool) -> str:
    """Return 'cuda' if GPU requested and available, else 'cpu'."""
    if not use_gpu:
        return "cpu"
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class Embedder:
    """
    Encode documents and queries into dense vectors using sentence-transformers.
    Recommended: all-MiniLM-L6-v2 for efficiency; see MTEB leaderboard for alternatives.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> "numpy.ndarray":
        """
        Encode text(s) into embeddings.
        Returns numpy array of shape (n_texts, embedding_dim).
        """
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
