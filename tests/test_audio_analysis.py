import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webapp"))

import audio_analysis as aa  # noqa: E402


SR = aa.SAMPLE_RATE


def tone(seconds: float, frequency: float = 220.0, amplitude: float = 0.35):
    t = np.arange(int(seconds * SR)) / SR
    # A little vibrato, so the frame energy is not a perfectly flat line.
    envelope = 1.0 + 0.25 * np.sin(2 * np.pi * 5.0 * t)
    return (amplitude * envelope * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def quiet(seconds: float, amplitude: float = 0.0008):
    rng = np.random.default_rng(7)
    return (amplitude * rng.standard_normal(int(seconds * SR))).astype(np.float32)


class VadTests(unittest.TestCase):
    def test_finds_the_sung_stretches_and_ignores_the_gaps(self):
        pcm = np.concatenate([quiet(3), tone(4), quiet(3), tone(5), quiet(2)])
        regions = aa.detect_speech_regions(pcm)
        self.assertEqual(len(regions), 2)
        # Padding widens each region slightly; it must not swallow the gap.
        self.assertAlmostEqual(regions[0].start, 3.0, delta=0.45)
        self.assertAlmostEqual(regions[0].end, 7.0, delta=0.45)
        self.assertAlmostEqual(regions[1].start, 10.0, delta=0.45)
        self.assertAlmostEqual(regions[1].end, 15.0, delta=0.45)

    def test_silence_produces_no_regions_at_all(self):
        self.assertEqual(aa.detect_speech_regions(quiet(20)), [])

    def test_empty_input_is_not_an_error(self):
        self.assertEqual(aa.detect_speech_regions(np.zeros(0, np.float32)), [])

    def test_a_blip_shorter_than_the_minimum_is_not_a_region(self):
        pcm = np.concatenate([quiet(2), tone(0.05), quiet(2)])
        self.assertEqual(aa.detect_speech_regions(pcm), [])

    def test_a_breath_does_not_split_one_phrase_in_two(self):
        pcm = np.concatenate([quiet(2), tone(2), quiet(0.15), tone(2), quiet(2)])
        regions = aa.detect_speech_regions(pcm)
        self.assertEqual(len(regions), 1)


class WindowTests(unittest.TestCase):
    def regions(self, *spans):
        return [aa.SpeechRegion(start, end) for start, end in spans]

    def test_a_short_region_becomes_exactly_one_window(self):
        windows = aa.plan_windows(self.regions((10.0, 22.0)))
        self.assertEqual(len(windows), 1)
        self.assertEqual((windows[0].start, windows[0].end), (10.0, 22.0))

    def test_a_long_region_is_split_into_windows_the_model_can_hold(self):
        windows = aa.plan_windows(self.regions((0.0, 300.0)))
        self.assertTrue(windows)
        self.assertTrue(all(w.duration <= aa.WINDOW_HARD_MAX for w in windows))
        self.assertAlmostEqual(windows[0].start, 0.0)
        self.assertAlmostEqual(windows[-1].end, 300.0)
        # Contiguous: no audio may fall between two windows.
        for previous, window in zip(windows, windows[1:]):
            self.assertAlmostEqual(previous.end, window.start, places=6)

    def test_a_different_phase_cuts_the_same_region_somewhere_else(self):
        base = aa.plan_windows(self.regions((0.0, 300.0)), phase=0.0)
        shifted = aa.plan_windows(self.regions((0.0, 300.0)), phase=0.5)
        self.assertNotEqual([w.end for w in base], [w.end for w in shifted])
        self.assertTrue(all(w.duration <= aa.WINDOW_HARD_MAX for w in shifted))

    def test_windows_are_numbered_in_order(self):
        windows = aa.plan_windows(self.regions((0.0, 120.0), (200.0, 260.0)))
        self.assertEqual([w.index for w in windows], list(range(len(windows))))

    def test_no_regions_means_no_work(self):
        self.assertEqual(aa.plan_windows([]), [])


class SeparationReportTests(unittest.TestCase):
    def test_a_clean_vocal_stem_passes(self):
        pcm = np.concatenate([quiet(2), tone(8), quiet(2), tone(6)])
        report = aa.assess_separation(pcm, None, len(pcm) / SR)
        self.assertEqual(report.verdict, "ok")
        self.assertGreater(report.speech_coverage, 0.5)

    def test_a_silent_stem_is_reported_as_silent(self):
        report = aa.assess_separation(quiet(20), None, 20.0)
        self.assertEqual(report.verdict, "silent")

    def test_an_empty_stem_is_reported_rather_than_crashing(self):
        report = aa.assess_separation(np.zeros(0, np.float32), None, 10.0)
        self.assertEqual(report.verdict, "empty")

    def test_leftover_bass_is_reported_as_bleed(self):
        # 45 Hz has no business in a vocal stem: that is the kick drum.
        bass = tone(12, frequency=45.0, amplitude=0.4)
        voice = tone(12, frequency=300.0, amplitude=0.05)
        report = aa.assess_separation(bass + voice, None, 12.0)
        self.assertEqual(report.verdict, "bleed")
        self.assertGreater(report.bleed_ratio, 0.55)

    def test_the_report_serializes_to_plain_json_types(self):
        report = aa.assess_separation(tone(6), None, 6.0)
        payload = report.as_dict()
        self.assertEqual(
            set(payload),
            {"vocalDbfs", "peakDbfs", "speechCoverage", "speechSeconds",
             "bleedRatio", "clipping", "verdict", "notes"},
        )
        self.assertIsInstance(payload["notes"], list)


class LanguageMapTests(unittest.TestCase):
    """A one-hour DJ set moves between languages; the map says where."""

    def setUp(self):
        import precision
        self.LanguageMap = precision.LanguageMap
        self.smooth = precision._smooth

    def single(self):
        return self.LanguageMap(dominant="ja", share=1.0,
                                spans=[(0.0, 600.0, "ja")], seconds={"ja": 600.0})

    def mixed(self):
        spans = [(0.0, 100.0, "ja"), (100.0, 200.0, "en"), (200.0, 260.0, "ja")]
        return self.LanguageMap(dominant="ja", share=0.61, spans=spans,
                                seconds={"ja": 160.0, "en": 100.0})

    def test_one_language_collapses_to_the_simple_case(self):
        mapping = self.single()
        self.assertFalse(mapping.multilingual)
        self.assertEqual(mapping.at(0.0, 30.0), "ja")
        self.assertEqual(mapping.at(9999.0, 10000.0), "ja")

    def test_a_mix_answers_with_the_language_of_that_moment(self):
        mapping = self.mixed()
        self.assertTrue(mapping.multilingual)
        self.assertEqual(mapping.at(10.0, 40.0), "ja")
        self.assertEqual(mapping.at(120.0, 150.0), "en")
        self.assertEqual(mapping.at(210.0, 240.0), "ja")

    def test_a_window_straddling_a_change_takes_the_side_it_mostly_covers(self):
        mapping = self.mixed()
        self.assertEqual(mapping.at(80.0, 105.0), "ja")
        self.assertEqual(mapping.at(95.0, 130.0), "en")

    def test_a_window_outside_every_span_still_gets_an_answer(self):
        self.assertEqual(self.mixed().at(5000.0, 5030.0), "ja")

    def test_an_empty_map_answers_nothing_rather_than_guessing(self):
        self.assertIsNone(self.LanguageMap().at(0.0, 30.0))


    def test_a_region_gets_one_language_no_matter_where_a_pass_cuts(self):
        # Every pass cuts the timeline at a different phase. If a window near a
        # language change could get a different answer per pass, the two passes
        # would transcribe the same audio in two languages and the vote would
        # have nothing to agree on.
        import precision
        regions = [aa.SpeechRegion(0.0, 60.0), aa.SpeechRegion(80.0, 140.0)]
        detected = [(0.0, 26.0, "ja"), (26.0, 52.0, "ja"), (52.0, 60.0, "en"),
                    (80.0, 106.0, "en"), (106.0, 140.0, "en")]
        spans = precision._by_region(detected, regions)
        self.assertEqual(spans, [(0.0, 60.0, "ja"), (80.0, 140.0, "en")])

        mapping = self.LanguageMap(dominant="en", share=0.5, spans=spans,
                                   seconds={"ja": 60.0, "en": 60.0})
        # Any window inside a region — whatever its phase — sees one answer.
        for start, end in [(0.0, 26.0), (13.0, 39.0), (40.0, 60.0), (52.0, 58.0)]:
            self.assertEqual(mapping.at(start, end), "ja", (start, end))
        for start, end in [(80.0, 106.0), (93.0, 119.0), (120.0, 140.0)]:
            self.assertEqual(mapping.at(start, end), "en", (start, end))

    def test_a_region_with_no_detection_is_left_out_rather_than_guessed(self):
        import precision
        regions = [aa.SpeechRegion(0.0, 30.0), aa.SpeechRegion(200.0, 230.0)]
        spans = precision._by_region([(0.0, 26.0, "ja")], regions)
        self.assertEqual(spans, [(0.0, 30.0, "ja")])

    def test_a_single_window_flip_between_agreeing_neighbours_is_noise(self):
        # One window of "oh oh oh" can be detected as anything.
        spans = [(0.0, 26.0, "ja"), (26.0, 52.0, "cy"), (52.0, 78.0, "ja")]
        self.assertEqual([s[2] for s in self.smooth(spans)], ["ja", "ja", "ja"])

    def test_a_real_language_change_survives_smoothing(self):
        spans = [(0.0, 26.0, "ja"), (26.0, 52.0, "en"),
                 (52.0, 78.0, "en"), (78.0, 104.0, "en")]
        self.assertEqual([s[2] for s in self.smooth(spans)],
                         ["ja", "en", "en", "en"])


if __name__ == "__main__":
    unittest.main()
