import sys
import json
import tempfile
import time
import unittest
import urllib.error
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

    def test_service_unavailable_is_retried_before_marking_failure(self):
        failure = urllib.error.HTTPError(
            "https://lrclib.net/api/search", 503, "Service Unavailable",
            {"Retry-After": "0"}, None,
        )
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = b"[]"
        with mock.patch.object(
            guides.urllib.request, "urlopen", side_effect=[failure, response]
        ) as request, mock.patch.object(guides.time, "sleep") as sleep:
            self.assertEqual(guides._search_candidates("Song", 1.0), [])
        self.assertEqual(request.call_count, 2)
        sleep.assert_called_once_with(guides.SEARCH_RETRY_DELAYS[0])

    def test_explicit_retry_bypasses_a_fresh_error_cache(self):
        track = guides.Track(0, "Artist", "Song", "Artist - Song")
        candidate = {
            "id": 7, "trackName": "Song", "artistName": "Artist",
            "plainLyrics": "real lyrics", "syncedLyrics": None,
        }
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            guides._cache_path(cache, track).write_text(json.dumps({
                "reference": None, "fetchedAt": time.time(),
                "error": "HTTPError: Service Unavailable", "candidateCount": 0,
            }), encoding="utf-8")
            with mock.patch.object(
                guides, "_search_candidates", return_value=[candidate]
            ) as search:
                reference = guides.fetch_reference(
                    track, cache, retry_errors=True)
        search.assert_called_once()
        self.assertEqual(reference["id"], 7)

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

    def test_a_japanese_line_is_never_overwritten_by_the_first_lyric_line(self):
        # _fold borra el japonés: antes "" == "" pasaba por línea corta exacta
        # y cada verso se cambiaba por la primera línea de la referencia.
        guide = {"tracks": [{
            "start": 0, "end": 60, "artist": "A", "title": "One",
            "reference": {"plain": "教えて 教えてよ\n世界で君が笑う"},
        }]}
        segments = [
            {"start": 2, "end": 5, "text": "夢の続き追いかけていた",
             "words": [{"word": "夢の続き", "start": 2, "end": 3.5},
                       {"word": "追いかけていた", "start": 3.5, "end": 5}]},
            {"start": 7, "end": 9, "text": "♪"},
        ]
        refined = guides.refine_segments(segments, guide)
        self.assertEqual(refined[0]["text"], "夢の続き追いかけていた")
        self.assertEqual(refined[0]["words"][0]["word"], "夢の続き")
        self.assertNotIn("referenceScore", refined[0])
        self.assertEqual(refined[1]["text"], "♪")


class ReferencePlausibilityTests(unittest.TestCase):
    """Una referencia equivocada hace más daño que ninguna.

    Su letra acaba de `initial_prompt` sobre el tramo, y eso es exactamente
    cómo un tema japonés termina transcrito fonéticamente en inglés.
    """

    def track(self, artist, title, *, start=0.0, end=247.0):
        return guides.Track(start=start, artist=artist, title=title,
                            label=f"{artist} - {title}", end=end)

    def candidate(self, artist, title, duration, *, plain="letra"):
        return {"artistName": artist, "trackName": title,
                "duration": duration, "plainLyrics": plain}

    def test_the_same_title_by_another_artist_is_not_the_same_song(self):
        # Caso real: "KARUT feat. 紫崎雪 - Ariel" contra el "Ariel" de Dean
        # Friedman, 1977, en inglés.
        track = self.track("KARUT feat. 紫崎雪", "Ariel")
        self.assertFalse(guides.plausible_candidate(
            track, self.candidate("Dean Friedman", "Ariel", 262.0)))

    def test_a_guest_credit_does_not_break_a_real_match(self):
        # "lapix feat. YUC'e" y "lapix" son el mismo artista principal.
        track = self.track("lapix feat. YUC'e", "Blue Love", end=235.0)
        self.assertTrue(guides.plausible_candidate(
            track, self.candidate("lapix", "Blue Love", 230.0)))

    def test_a_wildly_different_duration_is_another_recording(self):
        track = self.track("Mameyudoufu", "Quality Control", end=214.0)
        self.assertFalse(guides.plausible_candidate(
            track, self.candidate("Mameyudoufu", "Quality Control", 32.0)))

    def test_an_artist_written_only_in_japanese_is_judged_by_duration(self):
        # `_fold` se queda con letras latinas, así que un nombre en japonés se
        # queda en blanco: ahí no hay juicio de artista posible y no se puede
        # descartar por eso.
        self.assertIsNone(guides.artist_agreement("中村さんそ", "Mameyudoufu"))
        track = self.track("中村さんそ", "Fluffy", end=149.0)
        self.assertTrue(guides.plausible_candidate(
            track, self.candidate("Mameyudoufu", "Fluffy", 194.0)))

    def test_best_candidate_refuses_a_list_of_impostors(self):
        track = self.track("PSYQUI", "Nervousness", end=128.0)
        self.assertIsNone(guides.best_candidate(
            track, [self.candidate("Joon", "nervousness", 215.0)]))


class LanguagePriorTests(unittest.TestCase):

    def guide(self, *labels):
        tracks = []
        for index, label in enumerate(labels):
            artist, _, title = label.partition(" - ")
            tracks.append({"start": index * 100.0, "end": (index + 1) * 100.0,
                           "artist": artist, "title": title, "label": label})
        return {"tracks": tracks}

    def test_a_japanese_credit_marks_the_span_as_japanese(self):
        priors = guides.language_priors(
            self.guide("KARUT feat. 紫崎雪 - Ariel", "braven - Start Again"))
        self.assertEqual(priors, [{"start": 0.0, "end": 100.0,
                                   "language": "ja"}])

    def test_a_latin_credit_claims_nothing(self):
        self.assertEqual(guides.language_priors(
            self.guide("lapix - Living Dead")), [])

    def test_the_script_of_the_reference_lyrics_also_counts(self):
        guide = self.guide("Some Artist - Some Title")
        guide["tracks"][0]["reference"] = {"plain": "きみの名前を呼ぶ"}
        self.assertEqual(guides.language_priors(guide)[0]["language"], "ja")

    def test_hangul_is_korean_not_japanese(self):
        self.assertEqual(guides.script_hint("아이유 - Love wins all"), "ko")


class CachedReferenceRevalidationTests(unittest.TestCase):
    """El caché guarda la elección, no los candidatos.

    Un acierto dudoso de una versión sin filtros volvería tal cual y se los
    saltaría enteros, así que se revalida al leer.
    """

    def test_a_cached_impostor_is_dropped_and_searched_again(self):
        track = guides.Track(start=0.0, artist="KARUT feat. 紫崎雪",
                             title="Ariel", label="KARUT - Ariel", end=247.0)
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            guides._cache_path(cache, track).write_text(json.dumps({
                "reference": {"provider": "LRCLIB", "artist": "Dean Friedman",
                              "title": "Ariel", "duration": 262.0,
                              "plain": "Way on the other side"},
                "fetchedAt": time.time(), "error": None, "candidateCount": 17,
            }), encoding="utf-8")
            with mock.patch.object(guides, "_search_candidates",
                                   return_value=[]) as search:
                self.assertIsNone(guides.fetch_reference(track, cache))
            self.assertTrue(search.called)

    def test_a_cached_good_match_is_served_without_a_search(self):
        track = guides.Track(start=0.0, artist="YUC'e", title="Pump",
                             label="YUC'e - Pump", end=117.0)
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            guides._cache_path(cache, track).write_text(json.dumps({
                "reference": {"provider": "LRCLIB", "artist": "YUC'e",
                              "title": "PUMP", "duration": 194.0,
                              "plain": "pump it up"},
                "fetchedAt": time.time(), "error": None, "candidateCount": 3,
            }), encoding="utf-8")
            with mock.patch.object(guides, "_search_candidates") as search:
                reference = guides.fetch_reference(track, cache)
            self.assertEqual(reference["artist"], "YUC'e")
            self.assertFalse(search.called)



if __name__ == "__main__":
    unittest.main()
