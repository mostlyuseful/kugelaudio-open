"""Shared audio stitching utilities for long-form KugelAudio generation."""

from __future__ import annotations

import re
from typing import Iterable, List, Literal, Optional

import torch

PauseMode = Literal["none", "punctuation", "speaker-aware"]

_PUNCTUATION_PAUSE_MS = {
    ",": 120,
    ";": 180,
    ":": 180,
    ".": 260,
    "!": 260,
    "?": 260,
}
_SPEAKER_PAUSE_MS = 360
_SPEAKER_RE = re.compile(r"^\s*speaker\s+\d+\s*:", re.IGNORECASE)


def stitch_audio_chunks(
    audio_chunks: List[torch.Tensor],
    chunk_texts: Optional[Iterable[str]] = None,
    pause_mode: PauseMode = "punctuation",
    crossfade_ms: int = 30,
    sample_rate: int = 24000,
) -> torch.Tensor:
    """Stitch chunk waveforms into one output waveform.

    Args:
        audio_chunks: Generated waveforms in order.
        chunk_texts: Source texts corresponding to each chunk.
        pause_mode: Pause insertion strategy.
        crossfade_ms: Crossfade length in milliseconds. 0 disables crossfade.
        sample_rate: Audio sample rate.

    Returns:
        A single stitched waveform.

    Raises:
        ValueError: If no chunks are provided or pause_mode is invalid.
    """
    if not audio_chunks:
        raise ValueError("audio_chunks must not be empty")
    if pause_mode not in {"none", "punctuation", "speaker-aware"}:
        raise ValueError(f"Invalid pause_mode: {pause_mode}")
    if crossfade_ms < 0:
        raise ValueError("crossfade_ms must be >= 0")

    prepared_chunks = [_ensure_1d_tensor(chunk) for chunk in audio_chunks]
    if len(prepared_chunks) == 1:
        return prepared_chunks[0]

    texts = list(chunk_texts) if chunk_texts is not None else [""] * len(prepared_chunks)
    if len(texts) != len(prepared_chunks):
        raise ValueError("chunk_texts length must match audio_chunks length")

    output = prepared_chunks[0]
    for idx in range(1, len(prepared_chunks)):
        pause_samples = _pause_samples_for_boundary(
            left_text=texts[idx - 1],
            right_text=texts[idx],
            pause_mode=pause_mode,
            sample_rate=sample_rate,
            device=output.device,
            dtype=output.dtype,
        )
        if pause_samples is not None:
            output = torch.cat([output, pause_samples], dim=-1)
        output = _append_with_crossfade(output, prepared_chunks[idx], crossfade_ms, sample_rate)

    return output.contiguous()


def _pause_samples_for_boundary(
    left_text: str,
    right_text: str,
    pause_mode: PauseMode,
    sample_rate: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    if pause_mode == "none":
        return None

    duration_ms = _punctuation_pause_ms(left_text)
    if pause_mode == "speaker-aware" and _is_speaker_boundary(left_text, right_text):
        duration_ms = max(duration_ms, _SPEAKER_PAUSE_MS)

    if duration_ms <= 0:
        return None

    silence_len = int(sample_rate * duration_ms / 1000)
    if silence_len <= 0:
        return None
    return torch.zeros(silence_len, device=device, dtype=dtype)


def _append_with_crossfade(
    left: torch.Tensor,
    right: torch.Tensor,
    crossfade_ms: int,
    sample_rate: int,
) -> torch.Tensor:
    if crossfade_ms <= 0:
        return torch.cat([left, right], dim=-1)

    requested = int(sample_rate * crossfade_ms / 1000)
    overlap = min(requested, left.shape[-1], right.shape[-1])
    if overlap <= 0:
        return torch.cat([left, right], dim=-1)

    fade_out = torch.linspace(1.0, 0.0, overlap, device=left.device, dtype=left.dtype)
    fade_in = torch.linspace(0.0, 1.0, overlap, device=right.device, dtype=right.dtype)
    blended = left[-overlap:] * fade_out + right[:overlap] * fade_in
    return torch.cat([left[:-overlap], blended, right[overlap:]], dim=-1)


def _ensure_1d_tensor(audio: torch.Tensor) -> torch.Tensor:
    if not isinstance(audio, torch.Tensor):
        raise TypeError("audio chunk must be a torch.Tensor")
    if audio.dim() == 0:
        return audio.reshape(1)
    if audio.dim() == 1:
        return audio
    return audio.squeeze()


def _punctuation_pause_ms(text: str) -> int:
    stripped = text.rstrip()
    if not stripped:
        return 0
    last_char = stripped[-1]
    return _PUNCTUATION_PAUSE_MS.get(last_char, 0)


def _is_speaker_boundary(left_text: str, right_text: str) -> bool:
    return bool(_SPEAKER_RE.match(left_text) or _SPEAKER_RE.match(right_text))
