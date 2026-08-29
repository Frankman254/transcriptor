import sys
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


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

    def test_duplicate_candidates_prefer_the_synced_copy(self):
        candidates = [
            {"id": 1, "trackName": "Dear Jessie", "artistName": "Rollergirl",
             "plainLyrics": "lyrics", "syncedLyrics": None},
            {"id": 2, "trackName": "Dear Jessie", "artistName": "Rollergirl",
             "plainLyrics": "lyrics", "syncedLyrics": "[00:01.00] lyrics"},
            {"id": 3, "trackName": "Another", "artistName": "Artist"},
        ]
        unique = guides.dedupe_candidates(candidates)
        self.assertEqual([candidate["id"] for candidate in unique], [2, 3])

    def test_network_error_is_reported_instead_of_no_match(self):
        track = guides.Track(0, "Artist", "Song", "Artist - Song")
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(
                guides, "_search_candidates", side_effect=OSError("offline")
            ):
                self.assertIsNone(
                    guides.fetch_reference(track, Path(directory)))
        self.assertEqual(track.lookup["status"], "error")
        self.assertIn("offline", track.lookup["error"])

    def test_legacy_url_error_cache_is_retried_immediately(self):
        track = guides.Track(0, "Artist", "Song", "Artist - Song")
        candidate = {
            "id": 9, "trackName": "Song", "artistName": "Artist",
            "plainLyrics": "real lyrics", "syncedLyrics": "[00:01.00] real lyrics",
        }
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            guides._cache_path(cache_dir, track).write_text(json.dumps({
                "reference": None, "fetchedAt": time.time(),
                "error": "URLError", "candidateCount": 0,
            }), encoding="utf-8")
            with mock.patch.object(
                guides, "_search_candidates", return_value=[candidate]
            ) as search:
                reference = guides.fetch_reference(track, cache_dir)
        search.assert_called_once()
        self.assertEqual(reference["id"], 9)

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

    def test_track_metadata_without_real_lyrics_never_becomes_a_prompt(self):
        guide = {"tracks": [{
            "start": 0, "end": 60, "artist": "T-Pau",
            "title": "China in Your Hand", "reference": None,
        }]}
        self.assertEqual(guides.prompt_spans(guide), [])

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
