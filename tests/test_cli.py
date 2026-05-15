import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import kugelaudio_open.cli as cli


class _FakeProcessor:
    def __init__(self):
        self.saved = []

    def save_audio(self, audio, output):
        self.saved.append((audio, output))


class CLITests(unittest.TestCase):
    @patch("kugelaudio_open.cli.sys.argv", [
        "kugelaudio",
        "generate",
        "hello world",
        "--language",
        "de",
        "--max-words-per-chunk",
        "123",
        "--overlap-sentences",
        "2",
        "--chunking-strategy",
        "syntax-aware",
        "--pause-mode",
        "speaker-aware",
        "--crossfade-ms",
        "42",
        "--voice",
        "default",
        "--reference-audio",
        "ref.wav",
        "-o",
        "out.wav",
    ])
    def test_generate_passes_chunking_args_to_utility(self):
        processor = _FakeProcessor()
        with (
            patch("kugelaudio_open.utils.load_model_and_processor", return_value=(object(), processor)),
            patch("kugelaudio_open.utils.generate_speech", return_value="audio") as gen,
        ):
            cli.main()

        kwargs = gen.call_args.kwargs
        self.assertEqual(kwargs["text"], "hello world")
        self.assertEqual(kwargs["language"], "de")
        self.assertEqual(kwargs["voice"], "default")
        self.assertEqual(kwargs["voice_prompt"], "ref.wav")
        self.assertEqual(kwargs["max_words_per_chunk"], 123)
        self.assertEqual(kwargs["overlap_sentences"], 2)
        self.assertEqual(kwargs["chunking_strategy"], "syntax-aware")
        self.assertEqual(kwargs["pause_mode"], "speaker-aware")
        self.assertEqual(kwargs["crossfade_ms"], 42)
        self.assertEqual(processor.saved, [("audio", "out.wav")])

    @patch("kugelaudio_open.cli.sys.argv", [
        "kugelaudio",
        "generate",
        "hello world",
        "--max-words-per-chunk",
        "0",
    ])
    def test_generate_treats_non_positive_chunking_as_disabled(self):
        processor = _FakeProcessor()
        with (
            patch("kugelaudio_open.utils.load_model_and_processor", return_value=(object(), processor)),
            patch("kugelaudio_open.utils.generate_speech", return_value="audio") as gen,
        ):
            cli.main()

        self.assertIsNone(gen.call_args.kwargs["max_words_per_chunk"])

    @patch("kugelaudio_open.cli.sys.argv", [
        "kugelaudio",
        "generate",
        "hello world",
        "--crossfade-ms",
        "-1",
    ])
    def test_generate_rejects_negative_crossfade(self):
        processor = _FakeProcessor()
        with patch("kugelaudio_open.utils.load_model_and_processor", return_value=(object(), processor)):
            with self.assertRaisesRegex(ValueError, "crossfade-ms"):
                cli.main()

    @patch("kugelaudio_open.cli.sys.argv", [
        "kugelaudio",
        "generate",
        "hello world",
        "--overlap-sentences",
        "-1",
    ])
    def test_generate_rejects_negative_overlap_sentences(self):
        processor = _FakeProcessor()
        with patch("kugelaudio_open.utils.load_model_and_processor", return_value=(object(), processor)):
            with self.assertRaisesRegex(ValueError, "overlap-sentences"):
                cli.main()


if __name__ == "__main__":
    unittest.main()
