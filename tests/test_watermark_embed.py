import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kugelaudio_open.watermark.watermark import AudioWatermark


class _FakeGenerator:
    def get_watermark(self, audio_16k, sample_rate, message=None):
        return torch.ones_like(audio_16k)


class WatermarkEmbedTests(unittest.TestCase):
    def test_embed_aligns_resampled_audio_and_watermark_lengths(self):
        watermark = object.__new__(AudioWatermark)
        watermark.device = "cpu"
        watermark.AUDIOSEAL_SAMPLE_RATE = 16000
        watermark.message = torch.zeros((1, 16), dtype=torch.int64)
        watermark._generator = _FakeGenerator()

        def fake_resample(audio, orig_sr, target_sr):
            if orig_sr == 24000 and target_sr == 16000:
                return audio[..., :-1]
            if orig_sr == 16000 and target_sr == 24000:
                if torch.all(audio == 1):
                    return torch.ones(*audio.shape[:-1], audio.shape[-1] + 1)
                return torch.zeros(*audio.shape[:-1], audio.shape[-1])
            return audio

        watermark._resample = fake_resample

        audio = torch.zeros(92799, dtype=torch.float32)
        out = watermark.embed(audio, sample_rate=24000)

        self.assertEqual(out.shape[-1], 92799)


if __name__ == "__main__":
    unittest.main()
