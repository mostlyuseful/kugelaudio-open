#!/usr/bin/env python3
"""Minimal baseline script for cloning a voice from a reference audio file.

This script is intentionally verbose so it can be used as a trusted baseline
when debugging voice-prompt behavior.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from kugelaudio_open import KugelAudioForConditionalGenerationInference, KugelAudioProcessor


def describe_array(name, x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float().numpy()
    x = np.asarray(x)
    print(
        f"[{name}] shape={x.shape} dtype={x.dtype} "
        f"min={x.min():.4f} max={x.max():.4f} mean={x.mean():.4f} std={x.std():.4f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Baseline voice cloning script")
    parser.add_argument("voice_prompt", help="Path to reference audio file")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("-o", "--output", default="tts.wav", help="Output WAV path")
    parser.add_argument(
        "--model", default="kugelaudio/kugelaudio-0-open", help="Model ID or local path"
    )
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
    else:
        device = "cpu"
        dtype = torch.float32

    print(f"[Setup] model={args.model}")
    print(f"[Setup] device={device} dtype={dtype}")
    print(f"[Setup] voice_prompt={args.voice_prompt}")
    print(f"[Setup] output={args.output}")
    print(f"[Setup] text={args.text!r}")

    t0 = time.time()
    model = KugelAudioForConditionalGenerationInference.from_pretrained(
        args.model,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    print(f"[Load] model loaded in {time.time() - t0:.2f}s")

    has_semantic_path = (
        getattr(model.model, "semantic_tokenizer", None) is not None
        and getattr(model.model, "semantic_connector", None) is not None
    )
    print(f"[Load] semantic voice-conditioning path available={has_semantic_path}")
    print("[Load] not calling strip_encoders(); raw voice prompting requires encoders")

    processor = KugelAudioProcessor.from_pretrained(args.model)
    print("[Load] processor loaded")

    prompt_path = Path(args.voice_prompt)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Voice prompt not found: {prompt_path}")

    raw_prompt = processor.audio_processor._load_from_path(str(prompt_path))
    describe_array("Prompt/raw_loaded", raw_prompt)
    print(
        f"[Prompt] loaded via audio processor at target sample rate "
        f"{processor.audio_processor.sampling_rate} Hz"
    )

    prompt_audio = raw_prompt
    if processor.db_normalize and processor.audio_normalizer is not None:
        prompt_audio = processor.audio_normalizer(prompt_audio)
        describe_array("Prompt/db_normalized", prompt_audio)
    else:
        print("[Prompt] db normalization disabled")

    inputs = processor(text=args.text, voice_prompt=prompt_audio, return_tensors="pt")
    print(f"[Inputs] keys={sorted(inputs.keys())}")
    print(f"[Inputs] text_ids.shape={tuple(inputs['text_ids'].shape)}")
    print(
        f"[Inputs] speech_input_mask.shape={tuple(inputs['speech_input_mask'].shape)} "
        f"true_count={int(inputs['speech_input_mask'].sum().item())}"
    )
    if "speech_tensors" in inputs:
        print(f"[Inputs] speech_tensors.shape={tuple(inputs['speech_tensors'].shape)}")
    if "speech_masks" in inputs:
        print(
            f"[Inputs] speech_masks.shape={tuple(inputs['speech_masks'].shape)} "
            f"true_count={int(inputs['speech_masks'].sum().item())}"
        )
    if "speech_masks" in inputs:
        expected = int(inputs["speech_masks"].sum().item())
        actual = int(inputs["speech_input_mask"].sum().item())
        print(f"[Inputs] placeholder_count_matches_frames={expected == actual} ({actual} vs {expected})")

    model_inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

    t1 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **model_inputs,
            cfg_scale=args.cfg_scale,
            max_new_tokens=args.max_new_tokens,
        )
    print(f"[Generate] completed in {time.time() - t1:.2f}s")

    if not outputs.speech_outputs or outputs.speech_outputs[0] is None:
        raise RuntimeError("Generation failed: no speech output returned")

    audio = outputs.speech_outputs[0]
    describe_array("Output/audio", audio)

    processor.save_audio(audio, args.output)
    print(f"[Save] wrote {args.output}")


if __name__ == "__main__":
    main()
