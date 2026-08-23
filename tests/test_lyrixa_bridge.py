import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webapp"))

from lyrixa_bridge import (  # noqa: E402
    HARD_MAX_DURATION,
    MIN_CLIP_DURATION,
    build_lyrixa_envelope,
    normalize_whisper_segments,
)


def make_segment(tokens, starts, ends, *, start=None, end=None):
    words = [
        {"word": token if i == 0 else f" {token}", "start": starts[i], "end": ends[i]}
        for i, token in enumerate(tokens)
    ]
    return {
        "start": starts[0] if start is None else start,
        "end": ends[-1] if end is None else end,
        "text": " ".join(tokens),
        "words": words,
    }


class LyrixaBridgeTests(unittest.TestCase):
    def test_normal_three_second_segment_is_not_split(self):
        segment = make_segment(
            ["Una", "línea", "normal."], [1.0, 2.0, 3.0], [1.7, 2.7, 4.0]
        )
        clips = normalize_whisper_segments([segment])
        self.assertEqual(len(clips), 1)
        self.assertEqual((clips[0]["startTime"], clips[0]["endTime"]), (1.0, 4.0))

    def test_23_second_fixture_becomes_multiple_reasonable_clips(self):
        fixture = json.loads(
            (ROOT / "tests/fixtures/whisper_long_segment.json").read_text()
        )
        clips = normalize_whisper_segments(fixture["segments"])
        self.assertGreater(len(clips), 1)
        self.assertTrue(all(c["endTime"] - c["startTime"] <= HARD_MAX_DURATION for c in clips))
        self.assertEqual(clips[0]["startTime"], 36.0)
        self.assertEqual(clips[-1]["endTime"], 52.4)

    def test_strong_punctuation_wins_over_pause_and_comma(self):
        tokens = ["Uno,", "dos", "tres", "cuatro.", "cinco", "seis", "siete", "ocho"]
        starts = [0, 1, 2.0, 3.0, 4.2, 5.0, 6.0, 7.0]
        ends = [0.7, 1.5, 2.5, 3.6, 4.7, 5.5, 6.5, 7.6]
        clips = normalize_whisper_segments([make_segment(tokens, starts, ends)])
        self.assertEqual(clips[0]["text"], "Uno, dos tres cuatro.")

    def test_pause_is_used_when_there_is_no_strong_punctuation(self):
        tokens = ["uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho"]
        starts = [0, 0.8, 1.6, 2.4, 4.0, 4.8, 5.6, 6.4]
        ends = [0.5, 1.3, 2.1, 2.9, 4.5, 5.3, 6.1, 6.9]
        clips = normalize_whisper_segments([make_segment(tokens, starts, ends)])
        self.assertEqual(clips[0]["text"], "uno dos tres cuatro")

    def test_hard_duration_cut_without_punctuation_pause_or_size_boundary(self):
        tokens = [f"w{i}" for i in range(8)]
        starts = [i * 1.0 for i in range(8)]
        ends = [value + 0.8 for value in starts]
        clips = normalize_whisper_segments([make_segment(tokens, starts, ends)])
        self.assertGreater(len(clips), 1)
        self.assertTrue(all(c["endTime"] - c["startTime"] <= HARD_MAX_DURATION for c in clips))

    def test_segment_without_words_uses_original_text_and_times(self):
        clips = normalize_whisper_segments([
            {"start": 10.0, "end": 15.0, "text": "No se pierde este texto"}
        ])
        self.assertEqual(len(clips), 1)
        self.assertEqual(clips[0]["text"], "No se pierde este texto")
        self.assertEqual((clips[0]["startTime"], clips[0]["endTime"]), (10.0, 15.0))
        self.assertNotIn("words", clips[0])

    def test_partially_invalid_words_fall_back_without_losing_text(self):
        segment = make_segment(
            ["texto", "completo", "aquí"], [2.0, 3.0, 4.0], [2.7, 3.7, 4.7]
        )
        del segment["words"][1]["end"]
        clips = normalize_whisper_segments([segment])
        self.assertEqual([c["text"] for c in clips], [segment["text"]])
        self.assertNotIn("words", clips[0])

    def test_text_words_order_timing_ids_and_no_overlap_are_preserved(self):
        fixture = json.loads(
            (ROOT / "tests/fixtures/whisper_long_segment.json").read_text()
        )
        original = copy.deepcopy(fixture)
        clips = normalize_whisper_segments(fixture["segments"])
        source_words = [w["word"].strip() for w in fixture["segments"][0]["words"]]
        output_words = [w["text"] for clip in clips for w in clip.get("words", [])]
        self.assertEqual(output_words, source_words)
        self.assertEqual(
            " ".join(" ".join(c["text"] for c in clips).split()),
            " ".join(fixture["segments"][0]["text"].split()),
        )
        self.assertEqual(fixture, original, "the raw Whisper JSON must not be mutated")
        self.assertEqual([c["sourceIndex"] for c in clips], list(range(len(clips))))
        self.assertEqual(len({c["id"] for c in clips}), len(clips))
        self.assertEqual(
            [c["id"] for c in clips],
            [c["id"] for c in normalize_whisper_segments(fixture["segments"])],
        )
        self.assertTrue(all(c["endTime"] > c["startTime"] for c in clips))
        self.assertTrue(all(c["endTime"] - c["startTime"] >= MIN_CLIP_DURATION for c in clips))
        self.assertTrue(all(a["endTime"] <= b["startTime"] for a, b in zip(clips, clips[1:])))

    def test_subminimum_word_clip_is_merged(self):
        segment = make_segment(["a", "continuación"], [0.0, 1.0], [0.1, 1.6])
        clips = normalize_whisper_segments([segment])
        self.assertEqual(len(clips), 1)
        self.assertGreaterEqual(clips[0]["endTime"] - clips[0]["startTime"], MIN_CLIP_DURATION)

    def test_project_envelope_matches_authoring_contract_and_audio_reference(self):
        fixture = json.loads(
            (ROOT / "tests/fixtures/whisper_long_segment.json").read_text()
        )
        envelope = build_lyrixa_envelope(
            fixture,
            title="Canción de prueba",
            duration=59.0,
            audio_file_name="cancion.mp3",
            model="medium",
            language="Spanish",
            exported_at="2026-08-22T00:00:00+00:00",
            project_stem="cancion",
        )
        self.assertEqual(envelope["app"], "Lyrixa")
        self.assertEqual(envelope["schemaVersion"], 1)
        self.assertNotIn("exportKind", envelope)
        project = envelope["project"]
        self.assertEqual(project["audioTracks"]["master"], {
            "fileName": "cancion.mp3", "duration": 59.0
        })
        self.assertEqual(project["normalizedLyrics"], [c["text"] for c in project["clips"]])
        self.assertEqual(project["rawLyricsText"], "\n".join(project["normalizedLyrics"]))
        self.assertGreater(len(project["clips"]), 1)

        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "fixture.lyrixa.json"
            target.write_text(json.dumps(envelope))
            loaded = json.loads(target.read_text())
        self.assertEqual(loaded["project"]["clips"], project["clips"])


if __name__ == "__main__":
    unittest.main()
