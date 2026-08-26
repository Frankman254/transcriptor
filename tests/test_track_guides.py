import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webapp"))

import track_guides as guides  # noqa: E402


TRACKLIST = """Original Tracklist:
00:00:00 Artist A - First Song (Club Mix)
00:04:00 Artist B - Second Song
00:08:00 Artist C - Third Song

Nightcore Tracklist:
00:00:00 Artist A - First Song (Nightcore)
00:03:00 Artist B - Second Song
00:06:00 Artist C - Third Song
"""


class TrackGuideTests(unittest.TestCase):
    def test_named_sections_and_artist_title_are_parsed(self):
        sections = guides.parse_tracklist(TRACKLIST)
        self.assertEqual([section.name for section in sections],
                         ["Original Tracklist", "Nightcore Tracklist"])
        self.assertEqual(sections[0].tracks[1].artist, "Artist B")
        self.assertEqual(sections[0].tracks[1].title, "Second Song")

    def test_duration_selects_the_nightcore_timestamps(self):
        selected = guides.choose_section(guides.parse_tracklist(TRACKLIST), 9 * 60)
        self.assertEqual(selected.name, "Nightcore Tracklist")

    def test_two_part_timestamp_means_minutes_and_seconds(self):
        section = guides.parse_tracklist("01:43 T-Pau – China in Your Hand")[0]
        self.assertEqual(section.tracks[0].start, 103)
        self.assertEqual(section.tracks[0].artist, "T-Pau")

    def test_candidate_matching_ignores_remix_suffix(self):
        track = guides.Track(0, "Rollergirl", "Dear Jessie (Green Court Remix)", "")
        candidates = [
            {"id": 1, "trackName": "Something Else", "artistName": "Other",
             "plainLyrics": "wrong"},
            {"id": 2, "trackName": "Dear Jessie", "artistName": "Rollergirl",
             "plainLyrics": "Running through the love parade"},
        ]
        self.assertEqual(guides.best_candidate(track, candidates)["id"], 2)

    def test_prompt_is_limited_to_the_song_at_that_time(self):
        guide = {"tracks": [
            {"start": 0, "end": 60, "artist": "A", "title": "One",
             "reference": {"plain": "alpha beta"}},
            {"start": 60, "end": 120, "artist": "B", "title": "Two",
             "reference": {"plain": "gamma delta"}},
        ]}
        spans = guides.prompt_spans(guide)
        self.assertIn("alpha beta", guides.prompt_at(spans, 10, 20))
        self.assertIn("gamma delta", guides.prompt_at(spans, 70, 80))

    def test_only_a_strong_match_gets_exact_reference_spelling(self):
        guide = {"tracks": [{
            "start": 0, "end": 60, "artist": "A", "title": "One",
            "reference": {"plain": "Pink elephants and lemonade\nDear Jessie hear the laughter"},
        }]}
        segments = [
            {"start": 2, "end": 5, "text": "pink elephant and lemonade"},
            {"start": 7, "end": 9, "text": "everybody everybody"},
        ]
        refined = guides.refine_segments(segments, guide)
        self.assertEqual(refined[0]["text"], "Pink elephants and lemonade")
        self.assertEqual(refined[1]["text"], "everybody everybody")


if __name__ == "__main__":
    unittest.main()
