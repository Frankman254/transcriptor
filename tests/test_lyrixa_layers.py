import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webapp"))

from lyrixa_bridge import (  # noqa: E402
    HARD_MAX_DURATION,
    MAIN_LAYER_ID,
    FX_LAYER_ID,
    ROMAJI_LAYER_ID,
    TRANSLATION_LAYER_ID,
    build_lyrixa_envelope,
    clip_lines,
    normalize_whisper_segments,
)
from textjoin import display_width, join_raw, join_tokens  # noqa: E402


def segment(tokens, starts, ends, *, spaced=True):
    words = [
        {"word": (token if i == 0 or not spaced else f" {token}"),
         "start": starts[i], "end": ends[i]}
        for i, token in enumerate(tokens)
    ]
    return {
        "start": starts[0],
        "end": ends[-1],
        "text": join_raw([w["word"] for w in words]),
        "words": words,
    }


LONG = segment(
    ["Esta", "noche", "seguimos", "bailando,", "porque", "la", "música",
     "nos", "llama.", "Nadie", "quiere", "detenerse", "cuando", "vuelve",
     "el", "ritmo", "y", "la", "ciudad", "despierta."],
    [36.0, 36.65, 37.35, 38.15, 40.0, 40.65, 41.05, 41.85, 42.25,
     44.0, 44.7, 45.4, 46.5, 47.2, 47.9, 48.3, 50.0, 50.4, 50.9, 51.4],
    [36.6, 37.3, 38.1, 39.1, 40.6, 41.0, 41.8, 42.2, 43.0,
     44.6, 45.3, 46.4, 47.1, 47.8, 48.2, 49.2, 50.3, 50.8, 51.3, 52.4],
)


class TextJoinTests(unittest.TestCase):
    def test_latin_words_get_spaces_and_punctuation_hugs_the_word(self):
        self.assertEqual(join_tokens(["Tell", "me", "so,", "but,", "don't"]),
                         "Tell me so, but, don't")

    def test_japanese_is_joined_without_spaces(self):
        self.assertEqual(join_tokens(["君", "の", "名", "は", "。"]), "君の名は。")

    def test_a_script_change_inside_one_line_gets_exactly_one_space(self):
        self.assertEqual(
            join_tokens(["終わらない", "Future", "Some", "more", "music"]),
            "終わらない Future Some more music")
        # ...in both directions: a kanji after a Latin word starts a new word.
        self.assertEqual(
            join_tokens(["let", "you", "go", "誰", "の", "目"]),
            "let you go 誰の目")

    def test_full_width_punctuation_still_hugs_the_word_before_it(self):
        self.assertEqual(join_tokens(["君", "の", "名", "は", "。"]), "君の名は。")
        self.assertEqual(join_tokens(["終わり", "。", "Future"]), "終わり。 Future")

    def test_whisper_style_tokens_that_carry_their_own_space_are_respected(self):
        self.assertEqual(join_raw(["Esta", " noche", " normal."]),
                         "Esta noche normal.")

    def test_space_free_tokens_are_not_forced_apart(self):
        self.assertEqual(join_raw(["君", "の", "名"]), "君の名")

    def test_a_kanji_counts_as_two_columns(self):
        self.assertEqual(display_width("abc"), 3)
        self.assertEqual(display_width("君の名"), 6)


class BalancedSplitTests(unittest.TestCase):
    def clips(self):
        return normalize_whisper_segments([LONG])

    def test_a_long_line_is_cut_at_its_commas_and_periods(self):
        texts = [clip["text"] for clip in self.clips()]
        self.assertGreater(len(texts), 1)
        # Every line but the last ends on punctuation or at a musical rest.
        self.assertTrue(texts[0].endswith(","))
        self.assertTrue(texts[1].endswith("."))

    def test_no_line_is_left_with_scraps_while_another_carries_everything(self):
        clips = self.clips()
        lengths = [len(clip["text"]) for clip in clips]
        # The balance term exists precisely to stop this ratio from exploding.
        self.assertLess(max(lengths) / min(lengths), 3.0)
        durations = [c["endTime"] - c["startTime"] for c in clips]
        self.assertTrue(all(d <= HARD_MAX_DURATION for d in durations))

    def test_the_split_loses_no_text_and_no_word(self):
        clips = self.clips()
        rebuilt = " ".join(" ".join(c["text"] for c in clips).split())
        self.assertEqual(rebuilt, " ".join(LONG["text"].split()))
        emitted = [w["text"] for c in clips for w in c["words"]]
        self.assertEqual(emitted, [w["word"].strip() for w in LONG["words"]])

    def test_clip_lines_matches_what_the_envelope_will_contain(self):
        envelope = build_lyrixa_envelope(
            {"segments": [LONG]}, title="T", duration=60.0)
        main = [c["text"] for c in envelope["project"]["clips"]
                if c["layerId"] == MAIN_LAYER_ID]
        self.assertEqual(clip_lines({"segments": [LONG]}), main)

    def test_a_no_space_script_prefers_cutting_where_the_script_changes(self):
        tokens = ["誰", "の", "メモ", "気", "に", "しない", "で",
                  "You'll", "be", "getting", "high", "so", "let", "me", "go"]
        starts = [i * 0.62 for i in range(len(tokens))]
        ends = [s + 0.55 for s in starts]
        clips = normalize_whisper_segments([segment(tokens, starts, ends,
                                                    spaced=False)])
        self.assertGreater(len(clips), 1)
        self.assertEqual(clips[0]["text"], "誰のメモ気にしないで")


class TwoLayerEnvelopeTests(unittest.TestCase):
    def envelope(self, **kwargs):
        lines = clip_lines({"segments": [LONG]})
        defaults = {
            "title": "Canción",
            "duration": 60.0,
            "audio_file_name": "cancion.mp3",
            "language": "ja",
            "exported_at": "2026-08-25T00:00:00+00:00",
            "translations": [f"traducción {i}" for i in range(len(lines))],
            "translation_language": "es",
        }
        defaults.update(kwargs)
        return build_lyrixa_envelope({"segments": [LONG]}, **defaults)

    def test_the_translation_lands_on_its_own_declared_layer(self):
        project = self.envelope()["project"]
        layers = {layer["id"]: layer for layer in project["layers"]}
        self.assertEqual(layers[MAIN_LAYER_ID]["role"], "primary")
        self.assertEqual(layers[MAIN_LAYER_ID]["language"], "ja")
        self.assertEqual(layers[TRANSLATION_LAYER_ID]["role"], "translation")
        self.assertEqual(layers[TRANSLATION_LAYER_ID]["language"], "es")

    def test_the_two_text_layers_never_share_an_anchor(self):
        # Both renderers place a layer's clips absolutely at its preset, so two
        # text layers on the same preset draw on top of each other.
        layers = self.envelope()["project"]["layers"]
        anchors = [layer["renderSettings"]["positionPreset"] for layer in layers
                   if layer["id"] in {MAIN_LAYER_ID, TRANSLATION_LAYER_ID}]
        self.assertEqual(len(anchors), 2)
        self.assertEqual(len(set(anchors)), 2, anchors)

    def test_a_translation_shares_its_line_timing_including_the_hold(self):
        clips = self.envelope()["project"]["clips"]
        pairs = list(zip([c for c in clips if c["layerId"] == MAIN_LAYER_ID],
                         [c for c in clips if c["layerId"] == TRANSLATION_LAYER_ID]))
        self.assertTrue(pairs)
        for source, target in pairs:
            self.assertEqual(target["endTime"], source["endTime"])
        # And no two lines of the same layer are ever on screen at once.
        for layer_id in (MAIN_LAYER_ID, TRANSLATION_LAYER_ID):
            layer_clips = [c for c in clips if c["layerId"] == layer_id]
            self.assertTrue(all(a["endTime"] <= b["startTime"]
                                for a, b in zip(layer_clips, layer_clips[1:])))

    def test_every_lyric_line_gets_exactly_one_translation_at_the_same_time(self):
        clips = self.envelope()["project"]["clips"]
        main = [c for c in clips if c["layerId"] == MAIN_LAYER_ID]
        translated = [c for c in clips if c["layerId"] == TRANSLATION_LAYER_ID]
        self.assertEqual(len(main), len(translated))
        for source, target in zip(main, translated):
            # The shared identity is the *line*, not the main clip's id.
            self.assertEqual(target["sourceId"], source["sourceId"])
            self.assertNotEqual(target["id"], source["id"])
            self.assertEqual(target["startTime"], source["startTime"])
            self.assertEqual(target["endTime"], source["endTime"])

    def test_without_a_translation_the_channel_exists_but_is_empty(self):
        # Las cuatro capas salen siempre: un canal vacío se puede rellenar a
        # mano en Lyrixa, y que estén siempre las cuatro mantiene el `order`
        # estable entre trabajos, que es de lo que depende una escena de
        # Vibrix montada sobre una capa concreta.
        project = self.envelope(translations=None,
                                translation_language=None)["project"]
        by_id = {layer["id"]: layer for layer in project["layers"]}
        self.assertEqual(set(by_id), {MAIN_LAYER_ID, TRANSLATION_LAYER_ID,
                                      ROMAJI_LAYER_ID, FX_LAYER_ID})
        self.assertEqual([by_id[i]["order"] for i in
                          (MAIN_LAYER_ID, TRANSLATION_LAYER_ID,
                           ROMAJI_LAYER_ID, FX_LAYER_ID)],
                         [0, 1, 2, 3])
        # Vacío y sin idioma, así que no dice ser una traducción de nada.
        self.assertEqual(by_id[TRANSLATION_LAYER_ID]["role"], "backing")
        self.assertIsNone(by_id[TRANSLATION_LAYER_ID]["language"])
        self.assertTrue(all(c["layerId"] == MAIN_LAYER_ID
                            for c in project["clips"]))

    def test_the_fx_channel_is_a_command_track_not_a_lyric_one(self):
        # Su texto no se dibuja: un clip que diga "glitch 0.8" es una orden
        # para el renderer.  Ver docs/contrato-capa-fx.md.
        project = self.envelope()["project"]
        fx = next(l for l in project["layers"] if l["id"] == FX_LAYER_ID)
        self.assertEqual(fx["role"], "fx")
        self.assertTrue(fx["renderSettings"]["suppressClipText"])
        # Transcriptor nunca autora efectos: no sabe qué debería brillar.
        self.assertEqual([c for c in project["clips"]
                          if c["layerId"] == FX_LAYER_ID], [])

    def test_romaji_is_its_own_layer_and_never_touches_the_original(self):
        # Leer una mezcla japonesa no puede obligar a elegir entre los kanji y
        # cómo suenan: las dos lecturas conviven, cada una en su capa.
        lines = clip_lines({"segments": [LONG]})
        romaji = [f"romaji {i}" for i in range(len(lines))]
        project = self.envelope(romaji=romaji)["project"]
        main = [c for c in project["clips"] if c["layerId"] == MAIN_LAYER_ID]
        reading = [c for c in project["clips"]
                   if c["layerId"] == ROMAJI_LAYER_ID]
        self.assertEqual([c["text"] for c in main], lines)
        self.assertEqual([c["text"] for c in reading], romaji)
        self.assertNotIn("originalText", main[0])
        # La letra que Lyrixa lista sigue siendo la original.
        self.assertEqual(project["normalizedLyrics"], lines)

    def test_the_romaji_layer_pairs_with_the_original_by_source_id(self):
        # Mismo `sourceId`, ids distintos: así siguen emparejadas aunque
        # alguien recorte o sustituya un clip en Lyrixa.
        lines = clip_lines({"segments": [LONG]})
        romaji = [f"romaji {i}" for i in range(len(lines))]
        project = self.envelope(romaji=romaji, translations=None,
                                translation_language=None)["project"]
        by_layer = {MAIN_LAYER_ID: [], ROMAJI_LAYER_ID: []}
        for clip in project["clips"]:
            by_layer.setdefault(clip["layerId"], []).append(clip)
        main, reading = by_layer[MAIN_LAYER_ID], by_layer[ROMAJI_LAYER_ID]
        self.assertEqual([c["sourceId"] for c in main],
                         [c["sourceId"] for c in reading])
        self.assertTrue(set(c["id"] for c in main)
                        .isdisjoint(c["id"] for c in reading))

    def test_a_short_romaji_list_never_truncates_the_lyric(self):
        lines = clip_lines({"segments": [LONG]})
        project = self.envelope(romaji=["solo la primera"])["project"]
        main = [c for c in project["clips"] if c["layerId"] == MAIN_LAYER_ID]
        reading = [c for c in project["clips"]
                   if c["layerId"] == ROMAJI_LAYER_ID]
        self.assertEqual([c["text"] for c in main], lines)
        self.assertEqual([c["text"] for c in reading], ["solo la primera"])

    def test_a_line_already_in_latin_script_gets_no_romaji_clip(self):
        # Su romaji es idéntico al original: duplicarlo sería una capa de ruido.
        lines = clip_lines({"segments": [LONG]})
        project = self.envelope(romaji=list(lines))["project"]
        self.assertEqual([c for c in project["clips"]
                          if c["layerId"] == ROMAJI_LAYER_ID], [])

    def test_metadata_survives_a_round_trip_through_a_file(self):
        envelope = self.envelope(precision={"passes": 5, "converged": True})
        restored = json.loads(json.dumps(envelope, ensure_ascii=False))
        meta = restored["transcriptor"]
        self.assertEqual(meta["translationLanguage"], "es")
        self.assertIs(meta["romanized"], False)
        self.assertEqual(meta["precision"]["passes"], 5)
        self.assertEqual(restored["project"]["clips"],
                         envelope["project"]["clips"])


if __name__ == "__main__":
    unittest.main()
