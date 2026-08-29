import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webapp"))

import transcribe_engines as engines  # noqa: E402


class MlxConfigurationTests(unittest.TestCase):
    def test_mlx_never_receives_the_unsupported_beam_search_option(self):
        calls = []

        class FakeMlx:
            @staticmethod
            def transcribe(_pcm, **options):
                calls.append(options)
                return {"segments": [], "language": "en"}

        engine = engines._MlxWhisperEngine.__new__(engines._MlxWhisperEngine)
        engine._mlx = FakeMlx()
        engine._model = "fake-model"
        engine._run(
            np.zeros(16000, dtype=np.float32), language="en",
            initial_prompt=None,
            spec=engines.PassSpec(0, temperature=0.0, beam_size=5),
        )
        self.assertIsNone(calls[0]["beam_size"])


if __name__ == "__main__":
    unittest.main()
