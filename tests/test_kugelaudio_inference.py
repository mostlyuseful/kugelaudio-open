import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kugelaudio_open.models.kugelaudio_inference import KugelAudioForConditionalGenerationInference


class _FakeAcousticTokenizer:
    def __init__(self):
        self.calls = []

    def decode(self, latents, use_cache=False):
        self.calls.append({"latents": latents.clone(), "use_cache": use_cache})
        # Return a waveform whose length depends on the full latent sequence.
        return torch.arange(latents.shape[-1] + 2, dtype=latents.dtype, device=latents.device)


class InferenceTailDecodeTests(unittest.TestCase):
    def test_decode_speech_latent_sequence_uses_full_non_streaming_decode(self):
        model = object.__new__(KugelAudioForConditionalGenerationInference)
        torch.nn.Module.__init__(model)
        model.register_parameter("_param", torch.nn.Parameter(torch.tensor(0.0)))
        model.model = type("Inner", (), {})()
        model.model.acoustic_tokenizer = _FakeAcousticTokenizer()

        latents = [
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([4.0, 5.0, 6.0]),
        ]

        audio = model._decode_speech_latent_sequence(latents)

        self.assertTrue(torch.equal(audio, torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])))
        self.assertEqual(len(model.acoustic_tokenizer.calls), 1)
        call = model.acoustic_tokenizer.calls[0]
        self.assertFalse(call["use_cache"])
        self.assertEqual(call["latents"].shape, (1, 2, 3))
        self.assertTrue(
            torch.equal(
                call["latents"],
                torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]),
            )
        )


if __name__ == "__main__":
    unittest.main()
