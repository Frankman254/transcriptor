"""What splitting a Whisper segment must never do.

The historic failure was a single 36 s → 59 s clip: 23 seconds of lyric in one
box, unreadable on screen and impossible to re-time by hand.  The splitter that
replaced it is a global optimizer, and an optimizer is exactly the kind of code
that can start losing a word or overlapping two lines while every individual
line still looks plausible.  These are the invariants, checked over the real
fixture and over a mix that strings several segments together.
"""

import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webapp"))

from lyrixa_bridge import (  # noqa: E402
    CLIP_TAIL_HOLD,
    HARD_MAX_DURATION,
    MIN_CLIP_DURATION,
    normalize_whisper_segments,
)
from textjoin import join_raw  # noqa: E402


PUNCTUATION = ".,;:!?…。、！？"


def segment(tokens, start, step=0.45, gap_after=1.5):
    """One Whisper segment on a regular grid, with a pause after it."""
    words, time = [], start
    for token in tokens:
        words.append({"word": token, "start": round(time, 3),
                      "end": round(time + step * 0.8, 3)})
        time += step
    return {
        "start": start,
        "end": round(words[-1]["end"], 3),
        "text": join_raw([w["word"] for w in words]).strip(),
        "words": words,
    }, round(time + gap_after, 3)


def build_mix():
    """Several segments in a row, the shape a real mix actually has."""
    phrases = [
        ["Esta", " noche", " seguimos", " bailando,", " porque", " la",
         " música", " nos", " llama", " otra", " vez."],
        ["Suena", " fuerte,", " suena", " claro,", " y", " no", " se", " apaga."],
        ["Una", " sola", " palabra"],
        ["Corremos", " sin", " mirar", " atrás", " mientras", " la", " ciudad",
         " duerme", " y", " nosotros", " no", " queremos", " parar", " todavía",
         " porque", " mañana", " ya", " será", " tarde", " para", " todo."],
    ]
    segments, cursor = [], 0.0
    for tokens in phrases:
        built, cursor = segment(tokens, cursor)
        segments.append(built)
    return segments


FIXTURE = json.loads(
    (ROOT / "tests/fixtures/whisper_long_segment.json").read_text(encoding="utf-8")
)


class SegmentationGuarantees(unittest.TestCase):
    """Each guarantee is checked against both corpora."""

    def corpora(self):
        yield "fixture 36→59s", copy.deepcopy(FIXTURE["segments"])
        yield "mezcla", build_mix()

    def source_words(self, segments):
        return [word for seg in segments for word in seg["words"]]

    # 1 --------------------------------------------------------------------
    def test_no_text_is_lost(self):
        for name, segments in self.corpora():
            with self.subTest(name):
                clips = normalize_whisper_segments(segments)
                rebuilt = " ".join(" ".join(c["text"] for c in clips).split())
                expected = " ".join(
                    " ".join(seg["text"] for seg in segments).split())
                self.assertEqual(rebuilt, expected)

    # 2 --------------------------------------------------------------------
    def test_no_word_is_duplicated_or_dropped(self):
        for name, segments in self.corpora():
            with self.subTest(name):
                clips = normalize_whisper_segments(segments)
                produced = [w["text"] for c in clips for w in c.get("words", [])]
                expected = [w["word"].strip()
                            for w in self.source_words(segments)]
                self.assertEqual(produced, expected)
                self.assertEqual(len(produced), len(expected))

    # 3 --------------------------------------------------------------------
    def test_every_timestamp_moves_forward(self):
        for name, segments in self.corpora():
            with self.subTest(name):
                clips = normalize_whisper_segments(segments)
                cursor = -1.0
                for clip in clips:
                    self.assertGreaterEqual(clip["startTime"], cursor)
                    self.assertGreater(clip["endTime"], clip["startTime"])
                    cursor = clip["startTime"]
                    inner = -1.0
                    for word in clip.get("words", []):
                        self.assertGreaterEqual(word["startTime"], inner)
                        self.assertGreater(word["endTime"], word["startTime"])
                        inner = word["endTime"]

    # 4 --------------------------------------------------------------------
    def test_two_lines_are_never_on_screen_at_once(self):
        for name, segments in self.corpora():
            with self.subTest(name):
                clips = normalize_whisper_segments(segments)
                for before, after in zip(clips, clips[1:]):
                    self.assertLessEqual(before["endTime"], after["startTime"],
                                         f"{before['text']!r} / {after['text']!r}")

    # 5 --------------------------------------------------------------------
    def test_every_line_is_a_readable_length(self):
        ceiling = HARD_MAX_DURATION + CLIP_TAIL_HOLD
        for name, segments in self.corpora():
            with self.subTest(name):
                clips = normalize_whisper_segments(segments)
                for clip in clips:
                    span = clip["endTime"] - clip["startTime"]
                    self.assertGreaterEqual(span, MIN_CLIP_DURATION, clip["text"])
                    self.assertLessEqual(span, ceiling, clip["text"])

    def test_the_historic_23_second_clip_cannot_come_back(self):
        clips = normalize_whisper_segments(copy.deepcopy(FIXTURE["segments"]))
        self.assertGreater(len(clips), 1)
        longest = max(clip["endTime"] - clip["startTime"] for clip in clips)
        self.assertLess(longest, 23.0 / 2)

    # 6 --------------------------------------------------------------------
    def test_most_cuts_land_where_the_phrase_actually_ends(self):
        for name, segments in self.corpora():
            with self.subTest(name):
                clips = normalize_whisper_segments(segments)
                inner = clips[:-1]
                if not inner:
                    continue
                natural = sum(1 for clip in inner
                              if clip["text"].rstrip()[-1:] in PUNCTUATION)
                self.assertGreaterEqual(
                    natural, len(inner) * 2 // 3,
                    [clip["text"] for clip in inner])

    # 7 --------------------------------------------------------------------
    def test_word_timings_survive_the_split_untouched(self):
        for name, segments in self.corpora():
            with self.subTest(name):
                original = copy.deepcopy(segments)
                clips = normalize_whisper_segments(segments)
                produced = [w for c in clips for w in c.get("words", [])]
                for word, source in zip(produced, self.source_words(original)):
                    self.assertAlmostEqual(word["startTime"], source["start"], 3)
                    self.assertAlmostEqual(word["endTime"], source["end"], 3)
                self.assertEqual(segments, original,
                                 "the raw Whisper JSON must not be mutated")

    def test_a_line_starts_on_its_first_word_and_never_ends_before_its_last(self):
        # The tail hold may push `endTime` past the last word — into silence,
        # never over the next line — but it may not move the start.
        for name, segments in self.corpora():
            with self.subTest(name):
                for clip in normalize_whisper_segments(segments):
                    words = clip.get("words")
                    if not words:
                        continue
                    self.assertAlmostEqual(clip["startTime"],
                                           words[0]["startTime"], 3)
                    self.assertGreaterEqual(clip["endTime"],
                                            words[-1]["endTime"])


if __name__ == "__main__":
    unittest.main()
