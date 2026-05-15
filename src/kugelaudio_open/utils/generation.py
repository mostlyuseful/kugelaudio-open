"""High-level generation utilities for KugelAudio."""

from typing import Optional, Union

import torch

from kugelaudio_open.utils.chunking import split_text_into_chunks
from kugelaudio_open.utils.stitching import stitch_audio_chunks


def load_model_and_processor(
    model_name_or_path: str = "kugelaudio/kugelaudio-0-open",
    device: Optional[Union[str, torch.device]] = None,
    torch_dtype: Optional[torch.dtype] = None,
    use_flash_attention: bool = True,
):
    """Load KugelAudio model and processor.

    Args:
        model_name_or_path: HuggingFace model ID or local path
        device: Device to load model on (auto-detected if None)
        torch_dtype: Data type for model weights
        use_flash_attention: Whether to use flash attention if available

    Returns:
        Tuple of (model, processor)

    Example:
        >>> model, processor = load_model_and_processor("kugelaudio/kugelaudio-0-open")
    """
    from kugelaudio_open.models import KugelAudioForConditionalGenerationInference
    from kugelaudio_open.processors import KugelAudioProcessor

    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Auto-detect dtype
    if torch_dtype is None:
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Load model
    attn_impl = "flash_attention_2" if use_flash_attention else "sdpa"
    try:
        model = KugelAudioForConditionalGenerationInference.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn_impl,
        ).to(device)
    except Exception:
        # Fallback without flash attention
        model = KugelAudioForConditionalGenerationInference.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
        ).to(device)

    model.eval()

    # Load processor
    processor = KugelAudioProcessor.from_pretrained(model_name_or_path)

    return model, processor


def generate_speech(
    model,
    processor,
    text: str,
    voice: Optional[str] = None,
    voice_prompt: Optional[Union[str, torch.Tensor]] = None,
    cfg_scale: float = 3.0,
    max_new_tokens: int = 4096,
    device: Optional[Union[str, torch.device]] = None,
    max_words_per_chunk: Optional[int] = None,
    pause_mode: str = "punctuation",
    crossfade_ms: int = 30,
) -> torch.Tensor:
    """Generate speech from text with optional voice conditioning.

    Supports either pre-encoded voices (`voice`) or raw reference audio
    (`voice_prompt`). When chunking is enabled, the same voice conditioning is
    reused independently for each chunk and watermarking is applied once after
    stitching.

    Args:
        model: KugelAudio model
        processor: KugelAudio processor
        text: Text to synthesize
        voice: Name of a pre-encoded voice (from voices.json registry)
        voice_prompt: Raw voice prompt audio tensor or path
        cfg_scale: Classifier-free guidance scale
        max_new_tokens: Maximum number of tokens to generate per chunk
        device: Device for generation
        max_words_per_chunk: Enable chunking when set to a positive integer
        pause_mode: Pause insertion strategy for stitched multi-chunk output
        crossfade_ms: Crossfade duration between chunks in milliseconds

    Returns:
        Generated audio tensor (watermarked)
    """
    if device is None:
        device = next(model.parameters()).device

    if max_words_per_chunk is None:
        return _generate_single_pass(
            model=model,
            processor=processor,
            text=text,
            voice=voice,
            voice_prompt=voice_prompt,
            cfg_scale=cfg_scale,
            max_new_tokens=max_new_tokens,
            device=device,
        )

    plan = split_text_into_chunks(text, max_words_per_chunk)
    if len(plan.chunks) == 1:
        print("[Chunking] Chunking enabled but skipped: single chunk")
        return _generate_single_pass(
            model=model,
            processor=processor,
            text=text,
            voice=voice,
            voice_prompt=voice_prompt,
            cfg_scale=cfg_scale,
            max_new_tokens=max_new_tokens,
            device=device,
        )

    print(
        f"[Chunking] Enabled: {len(plan.chunks)} chunks, boundary_types={plan.boundary_types}, "
        f"pause_mode={pause_mode}, crossfade_ms={crossfade_ms}"
    )

    chunk_outputs = []
    for idx, chunk in enumerate(plan.chunks, start=1):
        print(f"[Chunking] Generating chunk {idx}/{len(plan.chunks)}")
        inputs = processor(text=chunk, voice=voice, voice_prompt=voice_prompt, return_tensors="pt")
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                cfg_scale=cfg_scale,
                max_new_tokens=max_new_tokens,
                do_watermark=False,
            )
        audio = outputs.speech_outputs[0] if outputs.speech_outputs else None
        if audio is None:
            raise RuntimeError(f"Generation failed for chunk {idx}")
        chunk_outputs.append(audio)

    stitched = stitch_audio_chunks(
        chunk_outputs,
        chunk_texts=plan.chunks,
        pause_mode=pause_mode,
        crossfade_ms=crossfade_ms,
        sample_rate=24000,
    )
    if hasattr(model, "_apply_watermark"):
        stitched = model._apply_watermark(stitched, sample_rate=24000)
    return stitched


def _generate_single_pass(
    model,
    processor,
    text: str,
    voice: Optional[str],
    voice_prompt: Optional[Union[str, torch.Tensor]],
    cfg_scale: float,
    max_new_tokens: int,
    device: Union[str, torch.device],
) -> torch.Tensor:
    inputs = processor(text=text, voice=voice, voice_prompt=voice_prompt, return_tensors="pt")
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            cfg_scale=cfg_scale,
            max_new_tokens=max_new_tokens,
        )

    audio = outputs.speech_outputs[0] if outputs.speech_outputs else None
    if audio is None:
        raise RuntimeError("Generation failed - no audio output")
    return audio
