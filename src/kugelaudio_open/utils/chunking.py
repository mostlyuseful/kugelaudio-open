"""Shared text chunking utilities for long-form KugelAudio generation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Literal

BoundaryType = Literal["sentence", "clause", "hard_wrap"]

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[,;:])\s+|\s+[—–-]\s+")


@dataclass(frozen=True)
class ChunkPlan:
    """Chunk plan for a piece of text.

    Attributes:
        chunks: Final text chunks in generation order.
        boundary_types: Split strategy used to create each chunk. The first chunk is
            labeled with the strategy that primarily determined its size.
    """

    chunks: List[str]
    boundary_types: List[BoundaryType]


def split_text_into_chunks(text: str, max_words_per_chunk: int) -> ChunkPlan:
    """Split text into sentence/clause-aware chunks.

    Args:
        text: Input text to split.
        max_words_per_chunk: Maximum number of words per output chunk. Must be > 0.

    Returns:
        ChunkPlan containing final chunks and their boundary types.

    Raises:
        ValueError: If text is empty/whitespace or max_words_per_chunk <= 0.
    """
    normalized = text.strip()
    if not normalized:
        raise ValueError("Text input is required")
    if max_words_per_chunk <= 0:
        raise ValueError("max_words_per_chunk must be greater than 0")

    sentence_candidates = _split_sentences(normalized)
    chunks: List[str] = []
    boundary_types: List[BoundaryType] = []
    current_parts: List[str] = []
    current_count = 0
    current_boundary: BoundaryType = "sentence"

    for sentence in sentence_candidates:
        pieces, piece_boundary = _split_oversized_sentence(sentence, max_words_per_chunk)
        for piece in pieces:
            piece_word_count = _word_count(piece)
            if current_parts and current_count + piece_word_count > max_words_per_chunk:
                chunks.append(" ".join(current_parts).strip())
                boundary_types.append(current_boundary)
                current_parts = [piece]
                current_count = piece_word_count
                current_boundary = piece_boundary
            else:
                current_parts.append(piece)
                current_count += piece_word_count
                current_boundary = _merge_boundary_type(current_boundary, piece_boundary)

    if current_parts:
        chunks.append(" ".join(current_parts).strip())
        boundary_types.append(current_boundary)

    return ChunkPlan(chunks=chunks, boundary_types=boundary_types)


def _split_sentences(text: str) -> List[str]:
    sentences = [segment.strip() for segment in _SENTENCE_SPLIT_RE.split(text) if segment.strip()]
    return sentences or [text.strip()]


def _split_oversized_sentence(sentence: str, max_words: int) -> tuple[List[str], BoundaryType]:
    if _word_count(sentence) <= max_words:
        return [sentence], "sentence"

    clause_parts = [part.strip() for part in _CLAUSE_SPLIT_RE.split(sentence) if part.strip()]
    if len(clause_parts) > 1:
        chunks: List[str] = []
        current_parts: List[str] = []
        current_count = 0
        used_hard_wrap = False
        for clause in clause_parts:
            clause_words = _word_count(clause)
            if clause_words > max_words:
                if current_parts:
                    chunks.append(" ".join(current_parts).strip())
                    current_parts = []
                    current_count = 0
                wrapped = _hard_wrap_words(clause, max_words)
                chunks.extend(wrapped)
                used_hard_wrap = True
                continue

            if current_parts and current_count + clause_words > max_words:
                chunks.append(" ".join(current_parts).strip())
                current_parts = [clause]
                current_count = clause_words
            else:
                current_parts.append(clause)
                current_count += clause_words

        if current_parts:
            chunks.append(" ".join(current_parts).strip())

        return chunks, "hard_wrap" if used_hard_wrap else "clause"

    return _hard_wrap_words(sentence, max_words), "hard_wrap"


def _hard_wrap_words(text: str, max_words: int) -> List[str]:
    words = text.split()
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def _word_count(text: str) -> int:
    return len(text.split())


def _merge_boundary_type(current: BoundaryType, incoming: BoundaryType) -> BoundaryType:
    priority = {"sentence": 0, "clause": 1, "hard_wrap": 2}
    return incoming if priority[incoming] > priority[current] else current
