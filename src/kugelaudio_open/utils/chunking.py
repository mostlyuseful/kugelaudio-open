"""Shared text chunking utilities for long-form KugelAudio generation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Literal, Optional

BoundaryType = Literal["sentence", "clause", "hard_wrap"]
ChunkingStrategyName = Literal["heuristic", "syntax-aware"]

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[,;:])\s+|\s+[—–-]\s+")
_SYNTAX_AWARE_PHRASE_SPLIT_RE = re.compile(
    r"(?<=[,;:])\s+|\s+[—–-]\s+|\s+(?=(?:and|but|or|because|which|that|while|although|however|then)\b)",
    re.IGNORECASE,
)
_AVAILABLE_CHUNKING_STRATEGIES: tuple[ChunkingStrategyName, ...] = ("heuristic", "syntax-aware")


@dataclass(frozen=True)
class ChunkPlan:
    """Chunk plan for a piece of text.

    Attributes:
        chunks: Final text chunks in generation order, including any optional
            overlap prefix.
        boundary_types: Split strategy used to create each chunk. The first chunk is
            labeled with the strategy that primarily determined its size.
        overlap_prefixes: Context carried over from the previous chunk. Empty for the
            first chunk and when overlap is disabled or unavailable.
        new_texts: The non-overlap portion newly introduced by each chunk.
    """

    chunks: List[str]
    boundary_types: List[BoundaryType]
    overlap_prefixes: List[str]
    new_texts: List[str]


def get_available_chunking_strategies() -> List[str]:
    """Return user-selectable chunking strategies."""
    return list(_AVAILABLE_CHUNKING_STRATEGIES)


def split_text_into_chunks(
    text: str,
    max_words_per_chunk: int,
    overlap_sentences: int = 0,
    chunking_strategy: str = "heuristic",
) -> ChunkPlan:
    """Split text into sentence/clause-aware chunks.

    Args:
        text: Input text to split.
        max_words_per_chunk: Maximum number of words per output chunk. Must be > 0.
        overlap_sentences: Number of completed trailing sentences from the previous
            chunk to prepend as overlap context. Must be >= 0.
        chunking_strategy: Chunk planning strategy. Supported values are
            ``heuristic`` and ``syntax-aware``. The syntax-aware strategy uses an
            optional English-oriented `pysbd` sentence segmenter when installed and
            otherwise falls back to the heuristic path.

    Returns:
        ChunkPlan containing final chunks and their boundary types.

    Raises:
        ValueError: If text is empty/whitespace, max_words_per_chunk <= 0,
            overlap_sentences < 0, or chunking_strategy is unsupported.
    """
    normalized = text.strip()
    if not normalized:
        raise ValueError("Text input is required")
    if max_words_per_chunk <= 0:
        raise ValueError("max_words_per_chunk must be greater than 0")
    if overlap_sentences < 0:
        raise ValueError("overlap_sentences must be greater than or equal to 0")

    strategy = _normalize_chunking_strategy(chunking_strategy)
    sentence_candidates = _split_sentences(normalized, strategy)
    chunks: List[str] = []
    boundary_types: List[BoundaryType] = []
    current_parts: List[str] = []
    current_count = 0
    current_boundary: BoundaryType = "sentence"

    for sentence in sentence_candidates:
        pieces, piece_boundary = _split_oversized_sentence(sentence, max_words_per_chunk, strategy)
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

    overlap_prefixes = [""]
    new_texts = [chunks[0]] if chunks else []
    overlapped_chunks = [chunks[0]] if chunks else []

    for idx in range(1, len(chunks)):
        overlap_prefix = _extract_overlap_prefix(chunks[idx - 1], overlap_sentences)
        overlap_prefixes.append(overlap_prefix)
        new_texts.append(chunks[idx])
        overlapped_chunks.append(
            f"{overlap_prefix} {chunks[idx]}".strip() if overlap_prefix else chunks[idx]
        )

    return ChunkPlan(
        chunks=overlapped_chunks,
        boundary_types=boundary_types,
        overlap_prefixes=overlap_prefixes,
        new_texts=new_texts,
    )


def _normalize_chunking_strategy(chunking_strategy: str) -> ChunkingStrategyName:
    normalized = chunking_strategy.strip().lower().replace("_", "-")
    if normalized not in _AVAILABLE_CHUNKING_STRATEGIES:
        raise ValueError(
            "chunking_strategy must be one of: " + ", ".join(_AVAILABLE_CHUNKING_STRATEGIES)
        )
    return normalized  # type: ignore[return-value]


def _split_sentences(text: str, strategy: ChunkingStrategyName) -> List[str]:
    if strategy == "syntax-aware":
        syntax_sentences = _split_sentences_syntax_aware(text)
        if syntax_sentences:
            return syntax_sentences
    return _split_sentences_heuristic(text)


def _split_sentences_heuristic(text: str) -> List[str]:
    sentences = [segment.strip() for segment in _SENTENCE_SPLIT_RE.split(text) if segment.strip()]
    return sentences or [text.strip()]


def _split_sentences_syntax_aware(text: str) -> Optional[List[str]]:
    segmenter = _load_syntax_aware_segmenter()
    if segmenter is None:
        return None

    try:
        sentences = [segment.strip() for segment in segmenter.segment(text) if segment.strip()]
    except Exception:
        return None
    return sentences or None


@lru_cache(maxsize=1)
def _load_syntax_aware_segmenter():
    """Load the optional syntax-aware sentence segmenter.

    This backend currently uses `pysbd` with English segmentation rules.
    When the dependency is unavailable, callers should fall back to the
    default heuristic chunker.
    """
    try:
        import pysbd
    except ImportError:
        return None
    return pysbd.Segmenter(language="en", clean=False)


def _split_oversized_sentence(
    sentence: str,
    max_words: int,
    strategy: ChunkingStrategyName,
) -> tuple[List[str], BoundaryType]:
    if _word_count(sentence) <= max_words:
        return [sentence], "sentence"

    if strategy == "syntax-aware":
        syntax_aware = _split_oversized_sentence_syntax_aware(sentence, max_words)
        if syntax_aware is not None:
            return syntax_aware

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


def _split_oversized_sentence_syntax_aware(
    sentence: str,
    max_words: int,
) -> Optional[tuple[List[str], BoundaryType]]:
    if _load_syntax_aware_segmenter() is None:
        return None

    phrase_parts = [
        part.strip() for part in _SYNTAX_AWARE_PHRASE_SPLIT_RE.split(sentence) if part.strip()
    ]
    if len(phrase_parts) <= 1:
        return None

    chunks: List[str] = []
    current_parts: List[str] = []
    current_count = 0
    used_hard_wrap = False

    for phrase in phrase_parts:
        phrase_words = _word_count(phrase)
        if phrase_words > max_words:
            if current_parts:
                chunks.append(" ".join(current_parts).strip())
                current_parts = []
                current_count = 0
            chunks.extend(_hard_wrap_words(phrase, max_words))
            used_hard_wrap = True
            continue

        if current_parts and current_count + phrase_words > max_words:
            chunks.append(" ".join(current_parts).strip())
            current_parts = [phrase]
            current_count = phrase_words
        else:
            current_parts.append(phrase)
            current_count += phrase_words

    if current_parts:
        chunks.append(" ".join(current_parts).strip())

    return chunks, "hard_wrap" if used_hard_wrap else "clause"


def _hard_wrap_words(text: str, max_words: int) -> List[str]:
    words = text.split()
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def _word_count(text: str) -> int:
    return len(text.split())


def _merge_boundary_type(current: BoundaryType, incoming: BoundaryType) -> BoundaryType:
    priority = {"sentence": 0, "clause": 1, "hard_wrap": 2}
    return incoming if priority[incoming] > priority[current] else current


def _extract_overlap_prefix(text: str, overlap_sentences: int) -> str:
    if overlap_sentences <= 0:
        return ""

    completed_sentences = [
        sentence for sentence in _split_sentences_heuristic(text) if sentence.endswith((".", "!", "?"))
    ]
    if not completed_sentences:
        return ""

    return " ".join(completed_sentences[-overlap_sentences:])
