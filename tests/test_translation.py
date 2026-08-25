"""Tests for the parts of the translator that do not need a model running.

Route planning and batch parsing are where a bug silently desynchronizes the
translation layer from the lyrics, so they are covered here; the actual
language quality is the local model's job.
"""

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webapp"))

import translation as T  # noqa: E402


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

    def test_a_missing_end_of_the_route_means_no_route(self):
        self.assertEqual(T.plan_route(None, "es"), [])
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
    def test_the_strongest_multilingual_model_present_wins(self):
        installed = ["codellama:7b", "qwen3:8b",
                     "hf.co/ggml-org/Qwen3.8-27B-GGUF:Q4_K_M"]
        self.assertEqual(T.pick_model(installed),
                         "hf.co/ggml-org/Qwen3.8-27B-GGUF:Q4_K_M")

    def test_a_code_model_is_still_better_than_refusing_to_translate(self):
        self.assertEqual(T.pick_model(["some-unknown-model"]),
                         "some-unknown-model")

    def test_no_models_means_no_choice(self):
        self.assertIsNone(T.pick_model([]))


if __name__ == "__main__":
    unittest.main()
