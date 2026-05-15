import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kugelaudio_open.utils.generation import generate_speech


class _FakeOutput:
    def __init__(self, audio):
        self.speech_outputs = [audio]


class _FakeProcessor:
    def __init__(self):
        self.calls = []

    def __call__(self, text, voice=None, voice_prompt=None, return_tensors=None):
        self.calls.append(
            {
                "text": text,
                "voice": voice,
                "voice_prompt": voice_prompt,
                "return_tensors": return_tensors,
            }
        )
        return {
            "input_ids": torch.tensor([[1]], dtype=torch.long),
            "text_value": text,
        }


class _FakeModel:
    def __init__(self):
        self._param = torch.nn.Parameter(torch.tensor(0.0))
        self.generate_calls = []
        self.watermark_calls = []

    def parameters(self):
        yield self._param

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        idx = len(self.generate_calls)
        return _FakeOutput(torch.tensor([float(idx), float(idx)]))

    def _apply_watermark(self, audio, sample_rate=24000):
        self.watermark_calls.append({"audio": audio.clone(), "sample_rate": sample_rate})
        return audio + 10


class GenerateSpeechTests(unittest.TestCase):
    def test_single_pass_when_chunking_disabled(self):
        model = _FakeModel()
        processor = _FakeProcessor()

        audio = generate_speech(model, processor, "Hello world.", max_words_per_chunk=None)

        self.assertTrue(torch.equal(audio, torch.tensor([1.0, 1.0])))
        self.assertEqual(len(processor.calls), 1)
        self.assertEqual(len(model.generate_calls), 1)
        self.assertEqual(model.watermark_calls, [])
        self.assertNotIn("do_watermark", model.generate_calls[0])

    def test_single_pass_when_chunking_produces_one_chunk(self):
        model = _FakeModel()
        processor = _FakeProcessor()

        audio = generate_speech(model, processor, "Hello world.", max_words_per_chunk=10)

        self.assertTrue(torch.equal(audio, torch.tensor([1.0, 1.0])))
        self.assertEqual(len(processor.calls), 1)
        self.assertEqual(len(model.generate_calls), 1)
        self.assertEqual(model.watermark_calls, [])

    def test_chunked_generation_reuses_voice_conditioning_and_watermarks_once(self):
        model = _FakeModel()
        processor = _FakeProcessor()

        audio = generate_speech(
            model,
            processor,
            "One two. Three four.",
            voice="default",
            voice_prompt="ref.wav",
            max_words_per_chunk=2,
            pause_mode="none",
            crossfade_ms=0,
        )

        self.assertEqual(len(processor.calls), 2)
        self.assertEqual([call["text"] for call in processor.calls], ["One two.", "Three four."])
        self.assertTrue(all(call["voice"] == "default" for call in processor.calls))
        self.assertTrue(all(call["voice_prompt"] == "ref.wav" for call in processor.calls))

        self.assertEqual(len(model.generate_calls), 2)
        self.assertTrue(all(call.get("do_watermark") is False for call in model.generate_calls))
        self.assertEqual(len(model.watermark_calls), 1)
        self.assertEqual(model.watermark_calls[0]["sample_rate"], 24000)
        self.assertTrue(torch.equal(model.watermark_calls[0]["audio"], torch.tensor([1.0, 1.0, 2.0, 2.0])))
        self.assertTrue(torch.equal(audio, torch.tensor([11.0, 11.0, 12.0, 12.0])))

    def test_chunked_generation_uses_overlap_prompt_context_but_stitches_new_text_boundaries(self):
        model = _FakeModel()
        processor = _FakeProcessor()

        with patch(
            "kugelaudio_open.utils.generation.stitch_audio_chunks",
            side_effect=lambda audios, chunk_texts, **kwargs: torch.cat(audios),
        ) as stitch:
            audio = generate_speech(
                model,
                processor,
                "One two. Three four. Five six.",
                max_words_per_chunk=2,
                overlap_sentences=1,
                chunking_strategy="heuristic",
                pause_mode="none",
                crossfade_ms=0,
            )

        self.assertEqual(
            [call["text"] for call in processor.calls],
            ["One two.", "One two. Three four.", "Three four. Five six."],
        )
        self.assertEqual(
            stitch.call_args.kwargs["chunk_texts"],
            ["One two.", "Three four.", "Five six."],
        )
        self.assertEqual(stitch.call_args.kwargs["pause_mode"], "none")
        self.assertEqual(stitch.call_args.kwargs["crossfade_ms"], 0)
        self.assertEqual(len(model.generate_calls), 3)
        self.assertEqual(len(model.watermark_calls), 1)
        self.assertTrue(torch.equal(audio, torch.tensor([11.0, 11.0, 12.0, 12.0, 13.0, 13.0])))

    def test_chunked_generation_forwards_chunking_strategy_to_planner(self):
        model = _FakeModel()
        processor = _FakeProcessor()

        with patch("kugelaudio_open.utils.generation.split_text_into_chunks") as planner:
            planner.return_value = type(
                "Plan",
                (),
                {
                    "chunks": ["Alpha.", "Beta."],
                    "new_texts": ["Alpha.", "Beta."],
                    "boundary_types": ["sentence", "sentence"],
                },
            )()
            generate_speech(
                model,
                processor,
                "Alpha. Beta.",
                max_words_per_chunk=1,
                chunking_strategy="syntax-aware",
                pause_mode="none",
                crossfade_ms=0,
            )

        self.assertEqual(planner.call_args.kwargs["chunking_strategy"], "syntax-aware")


if __name__ == "__main__":
    unittest.main()
