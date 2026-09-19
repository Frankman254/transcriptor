import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "webapp"))

import precision  # noqa: E402


class ApplyPriorsTests(unittest.TestCase):
    """El tracklist sabe quién canta; Whisper, sobre un stem con instrumental
    colado, responde "inglés" para casi cualquier cosa.

    Medido en una mezcla real de 58 minutos: de 289 líneas, el detector dio
    33 en japonés y 251 en inglés, y las "inglesas" eran japonés transcrito
    fonéticamente. Con los priors del tracklist son 126 en japonés.
    """

    def prior(self, start, end, language="ja"):
        return {"start": start, "end": end, "language": language}

    def test_a_span_the_detector_heard_only_in_english_gets_corrected(self):
        spans = [(0.0, 10.0, "en"), (10.0, 20.0, "en")]
        self.assertEqual(
            precision._apply_priors(spans, [self.prior(0.0, 20.0)]),
            [(0.0, 10.0, "ja"), (10.0, 20.0, "ja")])

    def test_a_minority_that_agrees_with_the_prior_takes_the_whole_span(self):
        # Decir "ja" exige oír fonética japonesa; decir "en" no exige nada.
        spans = [(0.0, 10.0, "en"), (10.0, 20.0, "en"), (20.0, 30.0, "ja")]
        self.assertEqual(
            {language for *_, language in
             precision._apply_priors(spans, [self.prior(0.0, 30.0)])},
            {"ja"})

    def test_a_third_language_the_detector_actually_heard_wins(self):
        # Oyó el audio; el prior solo leyó un nombre.
        spans = [(0.0, 10.0, "ko"), (10.0, 20.0, "ko"), (20.0, 30.0, "en")]
        self.assertEqual(
            precision._apply_priors(spans, [self.prior(0.0, 30.0)]), spans)

    def test_spans_outside_the_track_are_untouched(self):
        spans = [(0.0, 10.0, "en"), (100.0, 110.0, "en")]
        self.assertEqual(
            precision._apply_priors(spans, [self.prior(0.0, 20.0)]),
            [(0.0, 10.0, "ja"), (100.0, 110.0, "en")])

    def test_without_priors_nothing_changes(self):
        spans = [(0.0, 10.0, "en")]
        self.assertEqual(precision._apply_priors(spans, []), spans)


if __name__ == "__main__":
    unittest.main()
