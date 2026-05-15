import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kugelaudio_open.processors.kugelaudio_processor import KugelAudioProcessor


class _FakeTokenizer:
    def __init__(self):
        self.calls = []

    def encode(self, text, add_special_tokens=False):
        self.calls.append(text)
        return [len(self.calls)]


class ProcessorLanguageTests(unittest.TestCase):
    def _make_processor(self):
        processor = object.__new__(KugelAudioProcessor)
        processor.tokenizer = _FakeTokenizer()
        processor.audio_processor = object()
        processor.audio_normalizer = None
        processor.db_normalize = False
        processor.voices_registry = {}
        processor.voices_dir = None
        processor._model_name_or_path = None
        processor.speech_compression_ratio = 3200
        return processor

    def test_language_hint_is_added_to_system_prompt(self):
        processor = self._make_processor()

        processor(text="Hello world.", language="de")

        self.assertIn("Output the speech in German (de).", processor.tokenizer.calls[0])

    def test_auto_language_keeps_default_prompt(self):
        processor = self._make_processor()

        processor(text="Hello world.", language=None)

        self.assertNotIn("Output the speech in", processor.tokenizer.calls[0])


if __name__ == "__main__":
    unittest.main()
