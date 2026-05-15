#!/usr/bin/env python3
"""Command-line interface for KugelAudio."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="KugelAudio - Open-source text-to-speech",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch web interface
  kugelaudio ui
  
  # Launch with public share link
  kugelaudio ui --share
  
  # Generate speech from command line
  kugelaudio generate "Hello world!" -o output.wav
  
  # Generate with a specific pre-encoded voice
  kugelaudio generate "Hello world!" --voice default -o output.wav

  # Generate with reference audio prompt
  kugelaudio generate "Hello world!" --reference-audio ref.wav -o output.wav

  # Generate with syntax-aware chunk planning
  kugelaudio generate "Dr. Smith arrived. Then left." --max-words-per-chunk 20 --chunking-strategy syntax-aware -o output.wav

  # Check watermark in audio file
  kugelaudio verify audio.wav
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # UI command
    ui_parser = subparsers.add_parser("ui", help="Launch Gradio web interface")
    ui_parser.add_argument("--share", action="store_true", help="Create public share link")
    ui_parser.add_argument("--host", default="127.0.0.1", help="Server hostname")
    ui_parser.add_argument("--port", type=int, default=7860, help="Server port")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate speech from text")
    gen_parser.add_argument("text", help="Text to synthesize")
    gen_parser.add_argument("-o", "--output", default="output.wav", help="Output file path")
    gen_parser.add_argument(
        "-v", "--voice", help="Pre-encoded voice name (from voices.json registry)"
    )
    gen_parser.add_argument(
        "-r", "--reference-audio", help="Path to reference audio prompt for voice cloning"
    )
    gen_parser.add_argument("--model", default="kugelaudio/kugelaudio-0-open", help="Model ID")
    gen_parser.add_argument("--cfg-scale", type=float, default=3.0, help="Guidance scale")
    gen_parser.add_argument(
        "--max-words-per-chunk",
        type=int,
        default=0,
        help="Enable long-text chunking when greater than 0",
    )
    gen_parser.add_argument(
        "--overlap-sentences",
        type=int,
        default=0,
        help="Number of trailing completed sentences to reuse as context between chunks",
    )
    gen_parser.add_argument(
        "--chunking-strategy",
        choices=["heuristic", "syntax-aware"],
        default="heuristic",
        help="Chunk planning strategy to use when long-text chunking is enabled",
    )
    gen_parser.add_argument(
        "--pause-mode",
        choices=["none", "punctuation", "speaker-aware"],
        default="punctuation",
        help="Pause insertion mode for stitched chunked output",
    )
    gen_parser.add_argument(
        "--crossfade-ms",
        type=int,
        default=30,
        help="Crossfade duration between chunks in milliseconds",
    )

    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Check watermark in audio")
    verify_parser.add_argument("audio", help="Audio file to check")

    args = parser.parse_args()

    if args.command == "ui":
        from kugelaudio_open.ui import launch_app

        launch_app(
            share=args.share,
            server_name=args.host,
            server_port=args.port,
        )

    elif args.command == "generate":
        import torch

        from kugelaudio_open.utils import generate_speech, load_model_and_processor

        model, processor = load_model_and_processor(args.model)

        if args.crossfade_ms < 0:
            raise ValueError("--crossfade-ms must be >= 0")
        if args.overlap_sentences < 0:
            raise ValueError("--overlap-sentences must be >= 0")

        print("Generating speech...")
        audio = generate_speech(
            model=model,
            processor=processor,
            text=args.text,
            voice=args.voice,
            voice_prompt=args.reference_audio,
            cfg_scale=args.cfg_scale,
            max_new_tokens=4096,
            max_words_per_chunk=args.max_words_per_chunk if args.max_words_per_chunk > 0 else None,
            overlap_sentences=args.overlap_sentences,
            chunking_strategy=args.chunking_strategy,
            pause_mode=args.pause_mode,
            crossfade_ms=args.crossfade_ms,
        )

        # Save
        processor.save_audio(audio, args.output)
        print(f"Audio saved to {args.output}")

    elif args.command == "verify":
        import numpy as np
        import soundfile as sf

        from kugelaudio_open.watermark import AudioWatermark

        audio, sr = sf.read(args.audio)

        watermark = AudioWatermark()
        result = watermark.detect(audio, sample_rate=sr)

        if result.detected:
            print(f"✅ Watermark DETECTED (confidence: {result.confidence:.1%})")
            print("This audio was generated by KugelAudio.")
        else:
            print(f"❌ No watermark detected (confidence: {result.confidence:.1%})")
            print("This audio does not appear to be generated by KugelAudio.")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
