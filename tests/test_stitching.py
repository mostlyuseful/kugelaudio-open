import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kugelaudio_open.utils.stitching import stitch_audio_chunks


class StitchAudioChunksTests(unittest.TestCase):
    def test_rejects_empty_chunks(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            stitch_audio_chunks([])

    def test_rejects_invalid_pause_mode(self):
        with self.assertRaisesRegex(ValueError, "Invalid pause_mode"):
            stitch_audio_chunks([torch.ones(3), torch.ones(3)], pause_mode="weird")

    def test_single_chunk_bypass(self):
        chunk = torch.tensor([1.0, 2.0, 3.0])
        out = stitch_audio_chunks([chunk], pause_mode="punctuation", crossfade_ms=30)
        self.assertTrue(torch.equal(out, chunk))

    def test_none_pause_no_crossfade_is_plain_concat(self):
        out = stitch_audio_chunks(
            [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])],
            chunk_texts=["Hello", "World"],
            pause_mode="none",
            crossfade_ms=0,
            sample_rate=1000,
        )
        self.assertTrue(torch.equal(out, torch.tensor([1.0, 2.0, 3.0, 4.0])))

    def test_punctuation_pause_inserts_silence(self):
        out = stitch_audio_chunks(
            [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])],
            chunk_texts=["Hello,", "World"],
            pause_mode="punctuation",
            crossfade_ms=0,
            sample_rate=1000,
        )
        expected = torch.tensor([1.0, 2.0] + [0.0] * 120 + [3.0, 4.0])
        self.assertTrue(torch.equal(out, expected))

    def test_speaker_aware_falls_back_to_punctuation_when_no_marker(self):
        punct = stitch_audio_chunks(
            [torch.tensor([1.0]), torch.tensor([2.0])],
            chunk_texts=["Hello.", "World"],
            pause_mode="punctuation",
            crossfade_ms=0,
            sample_rate=1000,
        )
        speaker = stitch_audio_chunks(
            [torch.tensor([1.0]), torch.tensor([2.0])],
            chunk_texts=["Hello.", "World"],
            pause_mode="speaker-aware",
            crossfade_ms=0,
            sample_rate=1000,
        )
        self.assertTrue(torch.equal(punct, speaker))

    def test_speaker_aware_uses_stronger_pause_for_speaker_boundary(self):
        out = stitch_audio_chunks(
            [torch.tensor([1.0]), torch.tensor([2.0])],
            chunk_texts=["Narration.", "Speaker 1: Hello"],
            pause_mode="speaker-aware",
            crossfade_ms=0,
            sample_rate=1000,
        )
        expected = torch.tensor([1.0] + [0.0] * 360 + [2.0])
        self.assertTrue(torch.equal(out, expected))

    def test_crossfade_blends_overlap(self):
        out = stitch_audio_chunks(
            [torch.tensor([1.0, 1.0, 1.0]), torch.tensor([0.0, 0.0, 0.0])],
            chunk_texts=["A", "B"],
            pause_mode="none",
            crossfade_ms=2,
            sample_rate=1000,
        )
        self.assertTrue(torch.allclose(out, torch.tensor([1.0, 1.0, 0.0, 0.0]), atol=1e-6))

    def test_crossfade_clamps_for_short_chunks(self):
        out = stitch_audio_chunks(
            [torch.tensor([1.0]), torch.tensor([0.0])],
            chunk_texts=["A", "B"],
            pause_mode="none",
            crossfade_ms=50,
            sample_rate=1000,
        )
        self.assertTrue(torch.equal(out, torch.tensor([1.0])))


if __name__ == "__main__":
    unittest.main()
