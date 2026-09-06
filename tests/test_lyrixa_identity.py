"""Line identity: one logical line, one id, shared by every layer.

Lyrixa resolves clips with `clip.layerId === layerId && clip.sourceId === line.sourceId`
(`src/core/timeline/tapSync.ts`) and derives clip ids from `sourceId`
(`tapSyncClipId`).  A `sourceId` repeated inside a layer therefore resolves to
the wrong clip and mints duplicate ids, and a `sourceId` that differs between
the lyric and its translation loses the pairing entirely.
"""

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webapp"))

from lyrixa_bridge import (  # noqa: E402
    MAIN_LAYER_ID,
    TRANSLATION_LAYER_ID,
    build_lyrixa_envelope,
    channel_slug,
    clip_id,
    line_id,
    normalize_whisper_segments,
    translation_clips,
)


def words(tokens, start=0.0, step=0.42):
    out = []
    time = start
    for token in tokens:
        out.append({"word": token, "start": round(time, 3),
                    "end": round(time + step * 0.85, 3)})
        time += step
    return out


# One Whisper segment long enough that the splitter must cut it several times.
LONG_TOKENS = ["Esta", " noche", " seguimos", " bailando,", " porque", " la",
               " música", " nos", " llama", " otra", " vez,", " y", " nadie",
               " quiere", " irse", " a", " casa", " todavía."]
LONG = {
    "start": 0.0, "end": 7.6,
    "text": "".join(LONG_TOKENS).strip(),
    "words": words(LONG_TOKENS),
}
SECOND_TOKENS = ["Suena", " fuerte,", " suena", " claro."]
SECOND = {
    "start": 12.0, "end": 13.7,
    "text": "".join(SECOND_TOKENS).strip(),
    "words": words(SECOND_TOKENS, start=12.0),
}
SEGMENTS = {"segments": [LONG, SECOND]}


class IdentityHelperTests(unittest.TestCase):
    def test_line_ids_are_zero_padded_and_ordered(self):
        self.assertEqual(line_id(0), "line-000000")
        self.assertEqual(line_id(42), "line-000042")
        self.assertLess(line_id(9), line_id(10))     # padding keeps sort order

    def test_a_channel_slug_is_always_safe_in_an_id(self):
        self.assertEqual(channel_slug("es", "tr"), "es")
        self.assertEqual(channel_slug("pt-BR", "tr"), "pt-br")
        self.assertEqual(channel_slug(None, "tr"), "tr")
        self.assertEqual(channel_slug("  ", "tr"), "tr")
        self.assertEqual(channel_slug("zh/Hant", "tr"), "zh-hant")

    def test_clip_ids_name_their_channel(self):
        self.assertEqual(clip_id("main", 7), "clip-main-000007")
        self.assertEqual(clip_id("es", 7), "clip-es-000007")


class MainLayerIdentityTests(unittest.TestCase):
    def setUp(self):
        self.clips = normalize_whisper_segments(SEGMENTS["segments"])

    def test_one_segment_splits_into_several_lines(self):
        # Otherwise this fixture would not exercise the bug at all.
        self.assertGreater(len(self.clips), 2)

    def test_every_line_has_its_own_identity(self):
        ids = [clip["sourceId"] for clip in self.clips]
        self.assertEqual(len(set(ids)), len(ids), ids)

    def test_identity_matches_position_and_never_the_whisper_segment(self):
        for index, clip in enumerate(self.clips):
            self.assertEqual(clip["sourceId"], line_id(index))
            self.assertEqual(clip["sourceIndex"], index)
            self.assertFalse(clip["sourceId"].startswith("segment-"))

    def test_clip_ids_are_unique_too(self):
        ids = [clip["id"] for clip in self.clips]
        self.assertEqual(len(set(ids)), len(ids), ids)

    def test_whisper_provenance_survives_in_its_own_field(self):
        origins = [clip.get("segmentIndex") for clip in self.clips]
        self.assertEqual(origins[0], 0)
        self.assertEqual(origins[-1], 1)
        self.assertEqual(sorted(set(origins)), [0, 1])

    def test_a_derived_tap_sync_id_stays_unique(self):
        # Lyrixa builds `tap-<layer>-<sourceId>`; duplicated identities used to
        # collapse several clips onto one id.
        derived = [f"tap-{MAIN_LAYER_ID}-{clip['sourceId']}"
                   for clip in self.clips]
        self.assertEqual(len(set(derived)), len(derived))


class CrossLayerIdentityTests(unittest.TestCase):
    def envelope(self, **kwargs):
        clips = normalize_whisper_segments(SEGMENTS["segments"])
        options = {
            "title": "Prueba", "duration": 20.0,
            "translations": [f"traducción {i}" for i in range(len(clips))],
            "translation_language": "es",
        }
        options.update(kwargs)
        return build_lyrixa_envelope(SEGMENTS, **options)

    def by_layer(self, project, layer_id):
        return [clip for clip in project["clips"] if clip["layerId"] == layer_id]

    def test_the_two_layers_are_paired_by_the_same_identity(self):
        project = self.envelope()["project"]
        main = self.by_layer(project, MAIN_LAYER_ID)
        translated = self.by_layer(project, TRANSLATION_LAYER_ID)
        self.assertEqual(len(main), len(translated))
        self.assertEqual([clip["sourceId"] for clip in main],
                         [clip["sourceId"] for clip in translated])

    def test_a_translation_never_borrows_the_main_clips_id(self):
        project = self.envelope()["project"]
        main_ids = {clip["id"] for clip in self.by_layer(project, MAIN_LAYER_ID)}
        for clip in self.by_layer(project, TRANSLATION_LAYER_ID):
            self.assertNotIn(clip["id"], main_ids)
            self.assertNotIn(clip["sourceId"], main_ids)

    def test_the_translation_channel_is_named_after_its_language(self):
        project = self.envelope(translation_language="pt")["project"]
        for clip in self.by_layer(project, TRANSLATION_LAYER_ID):
            self.assertTrue(clip["id"].startswith("clip-pt-"), clip["id"])

    def test_a_translation_without_a_declared_language_still_gets_an_id(self):
        clips = normalize_whisper_segments(SEGMENTS["segments"])
        mirrored = translation_clips(clips, [f"x{i}" for i in range(len(clips))])
        self.assertTrue(all(clip["id"].startswith("clip-tr-") for clip in mirrored))
        self.assertEqual(len({clip["id"] for clip in mirrored}), len(mirrored))

    def test_every_clip_id_in_the_project_is_unique(self):
        project = self.envelope()["project"]
        ids = [clip["id"] for clip in project["clips"]]
        self.assertEqual(len(set(ids)), len(ids))

    def test_an_empty_translation_line_drops_the_clip_without_shifting_the_rest(self):
        clips = normalize_whisper_segments(SEGMENTS["segments"])
        texts = [f"t{i}" for i in range(len(clips))]
        texts[1] = "   "
        mirrored = translation_clips(clips, texts, language="es")
        self.assertEqual(len(mirrored), len(clips) - 1)
        # The survivors still point at their own line, not at the next one.
        for clip in mirrored:
            twin = next(c for c in clips if c["sourceId"] == clip["sourceId"])
            self.assertEqual(clip["startTime"], twin["startTime"])
            self.assertEqual(clip["text"], texts[twin["sourceIndex"]])

    def test_the_schema_version_announces_the_new_contract(self):
        meta = self.envelope()["transcriptor"]
        self.assertGreaterEqual(meta["bridgeSchemaVersion"], 3)


class RomanizationDeclarationTests(unittest.TestCase):
    def test_a_romanized_project_says_so_on_the_layer(self):
        clips = normalize_whisper_segments(SEGMENTS["segments"])
        envelope = build_lyrixa_envelope(
            SEGMENTS, title="P", duration=20.0,
            romaji=[f"romaji {i}" for i in range(len(clips))])
        main = next(layer for layer in envelope["project"]["layers"]
                    if layer["id"] == MAIN_LAYER_ID)
        self.assertTrue(main["romanized"])
        self.assertEqual(main["role"], "primary")

    def test_an_untouched_project_does_not_claim_romanization(self):
        envelope = build_lyrixa_envelope(SEGMENTS, title="P", duration=20.0)
        main = next(layer for layer in envelope["project"]["layers"]
                    if layer["id"] == MAIN_LAYER_ID)
        self.assertFalse(main["romanized"])


if __name__ == "__main__":
    unittest.main()
