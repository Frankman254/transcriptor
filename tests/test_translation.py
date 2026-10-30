"""Tests for the parts of the translator that do not need a model running.

Route planning and batch parsing are where a bug silently desynchronizes the
translation layer from the lyrics, so they are covered here; the actual
language quality is the local model's job.
"""

import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webapp"))

import translation as T  # noqa: E402

JA_CASES = json.loads(
    (ROOT / "tests" / "fixtures" / "japanese_precision_cases.json")
    .read_text(encoding="utf-8")
)


class LanguageCodeTests(unittest.TestCase):
    def test_iso_codes_pass_through(self):
        self.assertEqual(T.normalize_code("es"), "es")
        self.assertEqual(T.normalize_code("pt-BR"), "pt")

    def test_whisper_style_names_are_recognized(self):
        self.assertEqual(T.normalize_code("Spanish"), "es")
        self.assertEqual(T.normalize_code("japanese"), "ja")

    def test_nothing_maps_to_nothing(self):
        self.assertIsNone(T.normalize_code(""))
        self.assertIsNone(T.normalize_code(None))

    def test_labels_are_in_spanish_and_never_blank(self):
        self.assertEqual(T.language_label("ja"), "Japonés")
        self.assertEqual(T.language_label(None), "desconocido")
        self.assertEqual(T.language_label("xx"), "xx")


class RouteTests(unittest.TestCase):
    def test_russian_to_spanish_goes_through_english(self):
        self.assertEqual(T.plan_route("ru", "es"), [("ru", "en"), ("en", "es")])
        self.assertEqual(T.route_label(T.plan_route("ru", "es")),
                         "Ruso → Inglés → Español")

    def test_japanese_to_spanish_goes_through_english(self):
        self.assertEqual(T.plan_route("Japanese", "es"),
                         [("ja", "en"), ("en", "es")])

    def test_a_latin_script_source_translates_directly(self):
        self.assertEqual(T.plan_route("en", "es"), [("en", "es")])
        self.assertEqual(T.plan_route("fr", "pt"), [("fr", "pt")])

    def test_english_as_the_target_is_a_single_hop_even_from_russian(self):
        self.assertEqual(T.plan_route("ru", "en"), [("ru", "en")])

    def test_same_language_needs_no_work(self):
        self.assertEqual(T.plan_route("es", "es"), [])
        self.assertEqual(T.plan_route("Spanish", "es"), [])
        self.assertEqual(T.route_label([]), "sin traducción")

    def test_without_a_target_there_is_no_route_but_an_unknown_source_still_routes(self):
        # Un origen desconocido sí se traduce: devolver el texto tal cual
        # sería dejar que quien llama crea que eso es la traducción.
        self.assertEqual(T.plan_route(None, "es"), [(T.AUTO_SOURCE, "es")])
        self.assertEqual(T.plan_route("auto", "es"), [(T.AUTO_SOURCE, "es")])
        self.assertEqual(T.plan_route("es", None), [])
        self.assertEqual(T.plan_route("ru", ""), [])


class BatchParsingTests(unittest.TestCase):
    def test_the_plain_numbered_format_is_read_back(self):
        parsed = T.parse_numbered("1| Hola\n2| Adiós", 2)
        self.assertEqual(parsed, {1: "Hola", 2: "Adiós"})

    def test_the_shapes_models_actually_emit_are_tolerated(self):
        reply = "```\n1. Hola\n[2] Adiós\n3) Hasta luego\n4: Otra\n```"
        parsed = T.parse_numbered(reply, 4)
        self.assertEqual(parsed,
                         {1: "Hola", 2: "Adiós", 3: "Hasta luego", 4: "Otra"})

    def test_chatter_around_the_lines_is_ignored(self):
        reply = "Sure, here you go:\n1| Hola\n2| Adiós\nHope this helps!"
        self.assertEqual(T.parse_numbered(reply, 2), {1: "Hola", 2: "Adiós"})

    def test_a_line_number_beyond_the_batch_is_refused(self):
        # Accepting index 9 for a 2-line batch would shift the whole layer.
        self.assertEqual(T.parse_numbered("1| Hola\n9| Qué", 2), {1: "Hola"})

    def test_a_missing_line_is_reported_as_missing_not_invented(self):
        parsed = T.parse_numbered("1| Hola\n3| Tres", 3)
        self.assertEqual(set(parsed), {1, 3})

    def test_an_empty_translation_is_kept_as_an_empty_string(self):
        self.assertEqual(T.parse_numbered("1| \n2| Algo", 2), {1: "", 2: "Algo"})


class OfflineRomanizationTests(unittest.TestCase):
    def test_the_offline_fallback_produces_latin_script(self):
        result = T.romanize_offline(["夜に駆ける"])
        if result is None:
            self.skipTest("pykakasi no está instalado")
        self.assertTrue(result[0].isascii(), result[0])
        self.assertEqual(len(result), 1)

    def test_it_never_changes_the_number_of_lines(self):
        lines = ["夜に駆ける", "", "Hello"]
        result = T.romanize_offline(lines)
        if result is None:
            self.skipTest("pykakasi no está instalado")
        self.assertEqual(len(result), len(lines))

    def test_romaji_validator_rejects_unconverted_or_empty_japanese(self):
        self.assertFalse(T._valid_romaji("少しずつ", "少しずつ"))
        self.assertFalse(T._valid_romaji("少しずつ", ""))
        self.assertTrue(T._valid_romaji("少しずつ", "sukoshi zutsu"))
        self.assertTrue(T._valid_romaji("Hello", "Hello"))

    def test_invalid_llm_lines_fall_back_individually(self):
        with mock.patch.object(T, "pick_model", return_value="qwen3:8b"), \
             mock.patch.object(T, "_run_lines",
                               return_value=["mada narenai kedo", "少しずつ"]), \
             mock.patch.object(T, "romanize_offline",
                               return_value=["mada nare nai kedo", "sukoshi zutsu"]):
            result, backend = T.romanize(["まだ慣れないけど", "少しずつ"])
        self.assertEqual(result, ["mada narenai kedo", "sukoshi zutsu"])
        self.assertEqual(backend, "ollama:qwen3:8b+pykakasi")


class NoBackendTests(unittest.TestCase):
    def test_translating_nothing_is_not_an_error_even_without_a_model(self):
        result = T.translate([], source="ru", target="es")
        self.assertEqual(result["lines"], [])

    def test_a_no_op_route_returns_the_lines_untouched(self):
        lines = ["ya está en español"]
        result = T.translate(lines, source="es", target="es")
        self.assertEqual(result["lines"], lines)
        self.assertEqual(result["backend"], "none")

    def test_a_missing_model_is_an_explicit_actionable_error(self):
        original = T.pick_model
        T.pick_model = lambda models=None: None
        try:
            with self.assertRaises(T.TranslationUnavailable) as caught:
                T.translate(["привет"], source="ru", target="es")
        finally:
            T.pick_model = original
        self.assertIn("ollama", str(caught.exception).lower())


class ModelChoiceTests(unittest.TestCase):
    def test_a_modern_mid_size_model_wins_over_a_slow_27b(self):
        installed = ["codellama:7b", "qwen3:8b",
                     "hf.co/ggml-org/Qwen3.8-27B-GGUF:Q4_K_M"]
        self.assertEqual(T.pick_model(installed), "qwen3:8b")

    def test_a_code_model_is_still_better_than_refusing_to_translate(self):
        self.assertEqual(T.pick_model(["some-unknown-model"]),
                         "some-unknown-model")

    def test_no_models_means_no_choice(self):
        self.assertIsNone(T.pick_model([]))


class TranslationCacheTests(unittest.TestCase):
    def test_repeated_chorus_is_translated_once_and_then_loaded_from_disk(self):
        calls = []
        original = T._run_lines

        def fake_run(model, system, lines, progress=None):
            calls.append(list(lines))
            return [f"ES:{line}" for line in lines]

        with tempfile.TemporaryDirectory() as folder:
            T._run_lines = fake_run
            try:
                first = T._run_cached_lines(
                    "qwen3:8b", "prompt", ["hello", "hello", "goodbye"],
                    source="en", target="es", cache_dir=Path(folder))
                second = T._run_cached_lines(
                    "qwen3:8b", "prompt", ["hello", "goodbye"],
                    source="en", target="es", cache_dir=Path(folder))
            finally:
                T._run_lines = original
        self.assertEqual(calls, [["hello", "goodbye"]])
        self.assertEqual(first, ["ES:hello", "ES:hello", "ES:goodbye"])
        self.assertEqual(second, ["ES:hello", "ES:goodbye"])

    def test_same_text_with_different_context_does_not_collide(self):
        calls = []

        def fake_run(model, system, lines, progress=None):
            calls.append(list(lines))
            return [f"ES:{line}:{index}" for index, line in enumerate(lines)]

        with tempfile.TemporaryDirectory() as folder, \
             mock.patch.object(T, "_run_lines", side_effect=fake_run):
            result = T._run_cached_lines(
                "qwen3:8b", "prompt", ["same", "same"],
                source="en", target="es", cache_dir=Path(folder),
                contexts=["left-a|right-a", "left-b|right-b"],
            )
        self.assertEqual(calls, [["same", "same"]])
        self.assertEqual(result, ["ES:same:0", "ES:same:1"])

    def test_equivalent_context_reuses_a_repeated_chorus(self):
        calls = []

        def fake_run(model, system, lines, progress=None):
            calls.append(list(lines))
            return [f"ES:{line}" for line in lines]

        with tempfile.TemporaryDirectory() as folder, \
             mock.patch.object(T, "_run_lines", side_effect=fake_run):
            result = T._run_cached_lines(
                "qwen3:8b", "prompt", ["chorus", "chorus"],
                source="en", target="es", cache_dir=Path(folder),
                contexts=["verse|end", "verse|end"],
            )
        self.assertEqual(calls, [["chorus"]])
        self.assertEqual(result, ["ES:chorus", "ES:chorus"])



class MultilingualGroupingTests(unittest.TestCase):
    """A one-hour DJ set is not one language; routing is per line."""

    def setUp(self):
        self.group = T.group_by_language

    def test_lines_are_grouped_by_the_language_they_were_sung_in(self):
        groups = self.group(["ja", "ja", "en", "ja", "en"], None)
        self.assertEqual(groups["ja"], [0, 1, 3])
        self.assertEqual(groups["en"], [2, 4])

    def test_a_line_with_no_language_falls_back_to_the_track_language(self):
        groups = self.group([None, "ja", None], "en")
        self.assertEqual(groups["en"], [0, 2])
        self.assertEqual(groups["ja"], [1])

    def test_grouping_covers_every_line_exactly_once(self):
        languages = ["ja", "en", None, "ru", "en", "ja"]
        groups = self.group(languages, "en")
        covered = sorted(i for indices in groups.values() for i in indices)
        self.assertEqual(covered, list(range(len(languages))))

    def test_each_group_gets_the_route_its_own_language_needs(self):
        # This is the bug that produced English text for a Japanese track:
        # one global route cannot serve a mix.
        groups = self.group(["ja", "en"], None)
        routes = {code: T.plan_route(code, "es") for code in groups}
        self.assertEqual(routes["ja"], [("ja", "en"), ("en", "es")])
        self.assertEqual(routes["en"], [("en", "es")])

    def test_contiguous_groups_never_cross_a_track_boundary(self):
        rows = JA_CASES["trackGroups"]
        groups = T._contiguous_groups(
            [row["language"] for row in rows], None,
            [row["trackId"] for row in rows],
        )
        self.assertEqual(
            [(group[0], group[1], group[2]) for group in groups],
            [("ja", "track-a", [0, 1]), ("ja", "track-b", [2])],
        )


class ClipRomanizationTests(unittest.TestCase):
    """El romaji se pedía y no salía, sin un solo error visible.

    El parámetro se llamaba `romanize` igual que la función del módulo, así
    que la tapaba: la llamada reventaba con "'bool' object is not callable",
    el `except` se lo tragaba y el trabajo terminaba con la capa vacía y la
    etapa "romanizing" en 0.0 s.
    """

    def test_japanese_lines_come_back_romanized(self):
        with mock.patch.object(T, "romanize",
                               return_value=(["kimi no na wa"], "test")):
            report = T.translate_clip_lines(
                ["君の名は", "hello"], ["ja", "en"], source=None, target="",
                romaji=True, emit=lambda *a, **k: None)
        self.assertEqual(report["romaji"], ["kimi no na wa", "hello"])
        self.assertEqual(report["romajiBackend"], "test")
        self.assertIsNone(report["error"])

    def test_asking_for_no_romaji_leaves_the_channel_empty(self):
        report = T.translate_clip_lines(
            ["君の名は"], ["ja"], source=None, target="", romaji=False,
            emit=lambda *a, **k: None)
        self.assertIsNone(report["romaji"])

    def test_a_romanizer_that_fails_says_so_instead_of_staying_silent(self):
        with mock.patch.object(T, "romanize", side_effect=RuntimeError("boom")):
            report = T.translate_clip_lines(
                ["君の名は"], ["ja"], source=None, target="", romaji=True,
                emit=lambda *a, **k: None)
        self.assertIsNone(report["romaji"])
        self.assertIn("boom", report["error"])


if __name__ == "__main__":
    unittest.main()
