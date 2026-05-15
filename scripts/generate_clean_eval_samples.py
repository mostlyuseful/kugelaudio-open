#!/usr/bin/env python3
"""Generate clean KugelAudio evaluation samples on a stronger VM.

This script is intended for remote execution on a machine with enough RAM/VRAM.
It loads a manifest of prompts, generates speech with `do_watermark=False`, and
writes clean WAVs that can later be analyzed by `scripts/evaluate_audioseal_quality.py`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from kugelaudio_open import KugelAudioForConditionalGenerationInference, KugelAudioProcessor


def detect_device_and_dtype() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def load_manifest(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("Manifest must be a JSON array")
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Manifest item {idx} must be an object")
        if "id" not in item or "text" not in item:
            raise ValueError(f"Manifest item {idx} must contain 'id' and 'text'")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/audioseal_eval_samples.json"),
        help="JSON manifest of samples to generate",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/clean_generated_samples"),
        help="Where to write clean generated WAVs",
    )
    parser.add_argument(
        "--model",
        default="kugelaudio/kugelaudio-0-open",
        help="Model ID or local path",
    )
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Execution device",
    )
    args = parser.parse_args()

    if args.device == "auto":
        device, dtype = detect_device_and_dtype()
    else:
        device = args.device
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

    manifest = load_manifest(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Setup] model={args.model}")
    print(f"[Setup] device={device} dtype={dtype}")
    print(f"[Setup] manifest={args.manifest}")
    print(f"[Setup] output_dir={args.output_dir}")
    print(f"[Setup] sample_count={len(manifest)}")

    processor = KugelAudioProcessor.from_pretrained(args.model)
    model = KugelAudioForConditionalGenerationInference.from_pretrained(
        args.model,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    metadata = []
    for item in manifest:
        sample_id = item["id"]
        text = item["text"]
        language = item.get("language")
        voice = item.get("voice")
        voice_prompt = item.get("voice_prompt")
        output_path = args.output_dir / f"{sample_id}.wav"

        print(f"[Generate] id={sample_id} language={language} voice={voice} voice_prompt={voice_prompt}")
        inputs = processor(
            text=text,
            voice=voice,
            voice_prompt=voice_prompt,
            language=language,
            return_tensors="pt",
        )
        inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                cfg_scale=args.cfg_scale,
                max_new_tokens=args.max_new_tokens,
                show_progress=False,
                do_watermark=False,
            )

        if not outputs.speech_outputs or outputs.speech_outputs[0] is None:
            raise RuntimeError(f"Generation failed for sample {sample_id}")

        audio = outputs.speech_outputs[0]
        processor.save_audio(audio, str(output_path))
        metadata.append(
            {
                "id": sample_id,
                "text": text,
                "language": language,
                "voice": voice,
                "voice_prompt": voice_prompt,
                "output": str(output_path),
            }
        )

    metadata_path = args.output_dir / "manifest_outputs.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"[Done] wrote {metadata_path}")


if __name__ == "__main__":
    main()
