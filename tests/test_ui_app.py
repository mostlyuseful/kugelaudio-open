import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kugelaudio_open.ui import app as ui_app


class _FakeModel:
    def __init__(self):
        self._param = torch.nn.Parameter(torch.tensor(0.0))

    def parameters(self):
        yield self._param


class UIAppTests(unittest.TestCase):
    def test_generate_speech_forwards_chunking_args(self):
        model = _FakeModel()
        processor = object()
        fake_audio = torch.tensor([0.1, 0.2])

        with (
            patch.object(ui_app, "load_models", return_value=(model, processor, object())),
            patch("kugelaudio_open.utils.generate_speech", return_value=fake_audio) as gen,
        ):
            out = ui_app.generate_speech(
                text="Hello world.",
                voice_name="default",
                reference_audio=None,
                model_choice="kugelaudio-0-open",
                cfg_scale=3.0,
                max_tokens=2048,
                max_words_per_chunk=120,
                overlap_sentences=2,
                pause_mode="speaker-aware",
                crossfade_ms=45,
            )

        kwargs = gen.call_args.kwargs
        self.assertEqual(kwargs["text"], "Hello world.")
        self.assertEqual(kwargs["voice"], "default")
        self.assertEqual(kwargs["max_words_per_chunk"], 120)
        self.assertEqual(kwargs["overlap_sentences"], 2)
        self.assertEqual(kwargs["pause_mode"], "speaker-aware")
        self.assertEqual(kwargs["crossfade_ms"], 45)
        self.assertEqual(out[0], 24000)
        self.assertTrue(np.allclose(out[1], np.array([0.1, 0.2], dtype=np.float32)))

    def test_generate_speech_disables_chunking_for_non_positive_values(self):
        model = _FakeModel()
        processor = object()
        fake_audio = torch.tensor([0.1, 0.1])

        with (
            patch.object(ui_app, "load_models", return_value=(model, processor, object())),
            patch("kugelaudio_open.utils.generate_speech", return_value=fake_audio) as gen,
        ):
            ui_app.generate_speech(
                text="Hello world.",
                max_words_per_chunk=0,
            )

        self.assertIsNone(gen.call_args.kwargs["max_words_per_chunk"])
        self.assertEqual(gen.call_args.kwargs["overlap_sentences"], 0)

    def test_generate_speech_preserves_reference_audio_for_chunked_mode(self):
        model = _FakeModel()
        processor = object()
        fake_audio = torch.tensor([0.1, 0.1])
        reference_audio = (24000, np.array([0.0, 1.0, -1.0], dtype=np.float32))

        with (
            patch.object(ui_app, "load_models", return_value=(model, processor, object())),
            patch("kugelaudio_open.utils.generate_speech", return_value=fake_audio) as gen,
        ):
            ui_app.generate_speech(
                text="Hello world.",
                reference_audio=reference_audio,
                max_words_per_chunk=100,
            )

        voice_prompt = gen.call_args.kwargs["voice_prompt"]
        self.assertIsInstance(voice_prompt, np.ndarray)
        self.assertEqual(voice_prompt.ndim, 1)
        self.assertEqual(gen.call_args.kwargs["max_words_per_chunk"], 100)


if __name__ == "__main__":
    unittest.main()
