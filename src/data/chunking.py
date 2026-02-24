"""
Document chunking strategies for RAG retrieval.
Implements multiple approaches: fixed-size with overlap, sentence-aware, paragraph-based.
"""

import re
from abc import ABC, abstractmethod
from typing import List, Optional


class BaseChunker(ABC):
    """Base class for document chunkers."""

    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        """Split document into chunks. Returns list of text chunks."""
        pass


class FixedSizeChunker(BaseChunker):
    """
    Fixed-size chunking with optional overlap.
    Splits text by character/token count with configurable overlap.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 50,
        tokenizer=None,
        use_tokens: bool = False,
    ):
        """
        Args:
            chunk_size: Maximum size of each chunk (chars or tokens).
            overlap: Number of chars/tokens to overlap between chunks.
            tokenizer: Callable for tokenization if use_tokens=True (e.g., str.split).
            use_tokens: If True, use token count; else use character count.
        """
        self.chunk_size = chunk_size
        self.overlap = min(overlap, chunk_size - 1)
        self.tokenizer = tokenizer or (lambda s: s.split())
        self.use_tokens = use_tokens

    def chunk(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        text = text.strip()
        if self.use_tokens:
            return self._chunk_by_tokens(text)
        return self._chunk_by_chars(text)

    def _chunk_by_chars(self, text: str) -> List[str]:
        chunks = []
        start = 0
        step = self.chunk_size - self.overlap
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk.strip())
            start += step
        return chunks

    def _chunk_by_tokens(self, text: str) -> List[str]:
        tokens = self.tokenizer(text)
        chunks = []
        start = 0
        step = self.chunk_size - self.overlap
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = " ".join(chunk_tokens) if isinstance(tokens[0], str) else str(chunk_tokens)
            if chunk_text.strip():
                chunks.append(chunk_text.strip())
            start += step
        return chunks


class SentenceAwareChunker(BaseChunker):
    """
    Sentence-aware chunking: respects sentence boundaries.
    Splits on sentence boundaries while keeping chunks under chunk_size.
    """

    def __init__(self, chunk_size: int = 512, overlap_sentences: int = 1):
        self.chunk_size = chunk_size
        self.overlap_sentences = overlap_sentences

    def _split_sentences(self, text: str) -> List[str]:
        # Simple sentence splitting (can be replaced with NLTK/spaCy for better accuracy)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        sentences = self._split_sentences(text.strip())
        chunks = []
        current = []
        current_len = 0
        for sent in sentences:
            sent_len = len(sent) + 1  # +1 for space
            if current_len + sent_len > self.chunk_size and current:
                chunks.append(" ".join(current))
                # Overlap: keep last overlap_sentences
                overlap = current[-self.overlap_sentences:] if self.overlap_sentences else []
                current = overlap
                current_len = sum(len(s) + 1 for s in current)
            current.append(sent)
            current_len += sent_len
        if current:
            chunks.append(" ".join(current))
        return chunks


class ParagraphChunker(BaseChunker):
    """
    Paragraph-based chunking: splits on paragraph boundaries.
    Merges short paragraphs and splits long ones if needed.
    """

    def __init__(self, max_chunk_size: int = 512, min_chunk_size: int = 50):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def _split_paragraphs(self, text: str) -> List[str]:
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]

    def chunk(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        paragraphs = self._split_paragraphs(text.strip())
        chunks = []
        current = []
        current_len = 0
        for para in paragraphs:
            para_len = len(para) + 2  # newlines
            if para_len > self.max_chunk_size:
                # Split long paragraph with fixed-size
                if current:
                    chunks.append("\n\n".join(current))
                    current = []
                    current_len = 0
                sub_chunks = FixedSizeChunker(
                    chunk_size=self.max_chunk_size,
                    overlap=0,
                ).chunk(para)
                chunks.extend(sub_chunks)
            elif current_len + para_len > self.max_chunk_size and current:
                chunks.append("\n\n".join(current))
                current = [para]
                current_len = para_len
            else:
                current.append(para)
                current_len += para_len
        if current:
            chunk_text = "\n\n".join(current)
            if len(chunk_text) >= self.min_chunk_size or not chunks:
                chunks.append(chunk_text)
        return chunks
