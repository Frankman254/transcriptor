"""Pure Whisper -> Lyrixa project conversion.

The raw Whisper JSON is an input only.  This module never mutates it and has
no dependency on FastAPI, Whisper, or the filesystem, which keeps the timing
normalization deterministic and easy to test.

Line splitting is a global optimization, not a left-to-right walk.  A greedy
cutter fills the first line to the brim and leaves the last one with two words;
minimizing a cost over *all* cut sets at once is what produces paragraphs of
comparable weight, which is the whole point of splitting a long line.
"""

from __future__ import annotations

# Interfaz externa de este módulo; lo que no está aquí es seam interno
# y solo lo tocan sus propias pruebas (docs/arquitectura.md).
__all__ = [
    "BRIDGE_SCHEMA_VERSION", "LAYER_ROLES", "build_lyrixa_envelope",
    "clip_languages", "clip_lines",
]

import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from textjoin import display_width, is_no_space_char, join_raw


# Lyrixa's current hard floor (src/core/types/clip.ts).
MIN_CLIP_DURATION = 0.25

# A lyric line should normally live in this window.  Natural boundaries may
# stretch to HARD_MAX_DURATION, but never beyond it when a word boundary exists.
TARGET_MIN_DURATION = 2.0
TARGET_DURATION = 4.5
TARGET_MAX_DURATION = 6.0
HARD_MAX_DURATION = 8.0
MAX_WORDS = 12
MAX_CHARACTERS = 72
TARGET_CHARACTERS = 42
PAUSE_THRESHOLD = 0.45

# --- Cost model -------------------------------------------------------------
# Running past the ideal length is far worse than falling short of it: a long
# line is unreadable on screen, a short one is just a short lyric line.
OVER_DURATION_WEIGHT = 2.0
UNDER_DURATION_WEIGHT = 0.55
# Character balance.  Symmetric on purpose — this is the term that stops one
# paragraph from carrying most of the text while its neighbour carries three
# words.
CHARACTER_WEIGHT = 0.25
# Paid once per line, so an unnecessary extra cut has to earn its place.
BREAK_COST = 0.25
# Boundary bonuses (negative = preferred cut).  Punctuation ranks above a bare
# pause, and a comma that also lands on a pause gets both.
STRONG_PUNCTUATION_BONUS = -0.30
WEAK_PUNCTUATION_BONUS = -0.22
PAUSE_BONUS = -0.18
# Cutting mid-phrase with no signal at all.  High enough that it only happens
# when the length limits leave no choice.
NO_BOUNDARY_COST = 2.0
# Japanese and Chinese have no spaces, and Whisper's `words[]` splits them into
# sub-word pieces — so a cut between two such pieces usually lands *inside* a
# word.  Costlier than a plain unmarked cut, which pushes the optimizer toward
# the script change a few tokens away, where a real word does end.
MID_WORD_COST = 3.2
SCRIPT_CHANGE_COST = 0.9

# Whisper's word-level end timestamp marks where the *token* alignment stops,
# and in singing that is clearly before the held vowel does.  A line that
# disappears on that timestamp reads as changing while the voice is still
# going, so each line lingers past its last word — into the silence that
# follows it, never into the next line.
CLIP_TAIL_HOLD = 0.7
# Two lines must never be on screen at once, so a line always stops short of
# its successor by this much.
CLIP_HANDOVER_GAP = 0.05

# v3 replaced the overloaded `sourceId` with a real line identity.  A
# consumer that only reads text and times does not care; one that pairs layers
# does, and needs to know which contract it is looking at.
BRIDGE_SCHEMA_VERSION = 4

MAIN_LAYER_ID = "layer-main"
TRANSLATION_LAYER_ID = "layer-backing"
ROMAJI_LAYER_ID = "layer-romaji"
FX_LAYER_ID = "layer-fx"

# Lyrixa's own vocabulary (src/core/types/layer.ts).  It says
# `transliteration`, not `romanization`; an unknown role is dropped on import,
# so this list is a mirror and not a place to invent a seventh value.
LAYER_ROLES = ("primary", "translation", "transliteration", "backing", "fx",
               "annotation")

# The channel a clip belongs to, as it appears in its id.  Only cosmetic —
# `layerId` is what actually assigns a clip to a layer — but a project opened
# in an editor is read by people, and `clip-es-000042` says what
# `clip-0042-t` did not.
MAIN_CHANNEL = "main"
TRANSLATION_CHANNEL_FALLBACK = "tr"
ROMAJI_CHANNEL = "romaji"

_CHANNEL_UNSAFE = re.compile(r"[^a-z0-9]+")


def line_id(index: int) -> str:
    """The stable identity of one lyric line, shared by every layer.

    Minted here and nowhere else, and only *after* segmentation has decided
    what the lines are: a line born from splitting Whisper segment 7 in three
    is three lines, and numbering them by their origin would give all three the
    same identity.  Lyrixa looks clips up by `(layerId, sourceId)`
    (`src/core/timeline/tapSync.ts`), so a repeated value there silently
    resolves to the wrong clip.
    """
    return f"line-{index:06d}"


def channel_slug(language: str | None, fallback: str) -> str:
    slug = _CHANNEL_UNSAFE.sub("-", (language or "").strip().lower()).strip("-")
    return slug or fallback


def clip_id(channel: str, index: int) -> str:
    return f"clip-{channel}-{index:06d}"

_STRONG_PUNCTUATION = re.compile(r"[.!?…。！？][\"'’”)\]}]*$")
_WEAK_PUNCTUATION = re.compile(r"[,;:،؛，、—–-][\"'’”)\]}]*$")
_NO_SPACE_BEFORE = re.compile(r"\s+([,.;:!?…)\]}])")


def _finite_number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _round_time(value: float) -> float:
    return round(value, 3)


def _canonical_text(text: str) -> str:
    return _NO_SPACE_BEFORE.sub(r"\1", " ".join(text.split())).strip()


def _join_tokens(tokens: Iterable[str]) -> str:
    # Whisper normally includes the leading space in every token after the
    # first.  Hand-authored fixtures and other Whisper frontends often do not,
    # and a Japanese line legitimately has no spaces at all — `textjoin` is the
    # single place that decides which case this is.
    return join_raw(list(tokens))


def _fallback_clip(segment: dict[str, Any], segment_index: int) -> dict[str, Any] | None:
    text = str(segment.get("text") or "").strip()
    if not text:
        return None
    start = max(0.0, _finite_number(segment.get("start")) or 0.0)
    end = _finite_number(segment.get("end"))
    if end is None or end <= start:
        end = start + MIN_CLIP_DURATION
    clip = {
        "text": text,
        "startTime": _round_time(start),
        "endTime": _round_time(end),
        # Which Whisper segment this came from.  Provenance, not identity:
        # several lines legitimately share one segment.
        "segmentIndex": segment_index,
    }
    if segment.get("language"):
        clip["language"] = segment["language"]
    return clip


def _validated_words(segment: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Return a complete, monotonic word list or ``None`` for safe fallback.

    We deliberately do not partially use a damaged word list: doing so could
    silently drop the invalid token or require inventing its timestamps.
    """
    source = segment.get("words")
    if not isinstance(source, list) or not source:
        return None

    words: list[dict[str, Any]] = []
    previous_end = -math.inf
    raw_tokens: list[str] = []
    for raw in source:
        if not isinstance(raw, dict):
            return None
        token = raw.get("word", raw.get("text", ""))
        token = token if isinstance(token, str) else str(token or "")
        if not token.strip():
            continue
        start = _finite_number(raw.get("start"))
        end = _finite_number(raw.get("end"))
        if start is None or end is None or start < 0 or end <= start:
            return None
        # A cut between overlapping word windows would create overlapping clips.
        if start < previous_end:
            return None
        raw_tokens.append(token)
        entry = {
            "text": token.strip(),
            "_raw": token,
            "startTime": _round_time(start),
            "endTime": _round_time(end),
        }
        # Consensus output carries how sure the passes were; keep it so the
        # editor can highlight the lines worth a human ear.
        for key in ("agreement", "votes", "probability"):
            value = raw.get(key)
            if value is not None:
                entry[key] = value
        words.append(entry)
        previous_end = end

    if not words:
        return None

    segment_text = str(segment.get("text") or "")
    if _canonical_text(_join_tokens(raw_tokens)) != _canonical_text(segment_text):
        # The word list is incomplete, so segmenting it would lose source text.
        return None
    return words


# --------------------------------------------------------------------------- #
# Segmentación balanceada
# --------------------------------------------------------------------------- #
def _boundary_bonus(words: list[dict[str, Any]], end: int) -> float:
    """What a cut after ``words[end]`` is worth.

    Additive by design: a comma that also lands on a musical rest is a better
    place to break than either signal alone, and the cost model should say so.
    """
    if end >= len(words) - 1:
        return 0.0                      # end of the segment: free
    token = words[end]["text"]
    bonus = 0.0
    if _STRONG_PUNCTUATION.search(token):
        bonus += STRONG_PUNCTUATION_BONUS
    elif _WEAK_PUNCTUATION.search(token):
        bonus += WEAK_PUNCTUATION_BONUS
    if words[end + 1]["startTime"] - words[end]["endTime"] >= PAUSE_THRESHOLD:
        bonus += PAUSE_BONUS
    if bonus < 0:
        return bonus

    # Nothing marks this boundary, so fall back to what the scripts say.
    left, right = words[end]["text"], words[end + 1]["text"]
    if not left or not right:
        return NO_BOUNDARY_COST
    left_joined = is_no_space_char(left[-1])
    right_joined = is_no_space_char(right[0])
    if left_joined and right_joined:
        return MID_WORD_COST
    if left_joined != right_joined:
        return SCRIPT_CHANGE_COST
    return NO_BOUNDARY_COST


def _piece_cost(words: list[dict[str, Any]], start: int, end: int) -> float:
    duration = words[end]["endTime"] - words[start]["startTime"]
    characters = display_width(_join_tokens(w["_raw"] for w in words[start:end + 1]))

    if duration > TARGET_DURATION:
        length = OVER_DURATION_WEIGHT * (
            (duration - TARGET_DURATION) / TARGET_DURATION) ** 2
    else:
        length = UNDER_DURATION_WEIGHT * (
            (TARGET_DURATION - duration) / TARGET_DURATION) ** 2

    balance = CHARACTER_WEIGHT * (
        (characters - TARGET_CHARACTERS) / TARGET_CHARACTERS) ** 2
    return BREAK_COST + length + balance + _boundary_bonus(words, end)


def _feasible(words: list[dict[str, Any]], start: int, end: int,
              max_duration: float, max_characters: float,
              max_words: float) -> bool:
    if end - start + 1 > max_words:
        return False
    if words[end]["endTime"] - words[start]["startTime"] > max_duration:
        return False
    if end == start:
        return True                     # a single word can never be split
    width = display_width(_join_tokens(w["_raw"] for w in words[start:end + 1]))
    return width <= max_characters


def _solve_cuts(words: list[dict[str, Any]], max_duration: float,
                max_characters: float, max_words: float) -> list[int] | None:
    """Minimum-cost partition of ``words`` into lines, or None if impossible.

    O(n·k) where k is how many words fit in one line, so a 12-word cap keeps
    this linear in practice even on a segment with hundreds of words.
    """
    count = len(words)
    best = [math.inf] * (count + 1)
    origin = [0] * (count + 1)
    best[0] = 0.0
    for end in range(1, count + 1):
        for start in range(end - 1, -1, -1):
            if not _feasible(words, start, end - 1, max_duration,
                             max_characters, max_words):
                # Constraints only tighten as the line grows, so once this
                # start is infeasible every earlier one is too.
                break
            if best[start] == math.inf:
                continue
            candidate = best[start] + _piece_cost(words, start, end - 1)
            if candidate < best[end]:
                best[end] = candidate
                origin[end] = start
    if best[count] == math.inf:
        return None

    cuts: list[int] = []
    cursor = count
    while cursor > 0:
        cuts.append(origin[cursor])
        cursor = origin[cursor]
    return sorted(cuts)


def split_words(words: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Split one word list into balanced lines.

    The limits relax in steps: first the comfortable target, then the hard
    ceiling, and only then anything goes.  A single word that lasts nine
    seconds must still end up in a line, and it must not drag its neighbours
    into an unreadable one.
    """
    if not words:
        return []
    ladder = (
        (TARGET_MAX_DURATION, MAX_CHARACTERS, MAX_WORDS),
        (HARD_MAX_DURATION, MAX_CHARACTERS * 1.5, MAX_WORDS * 2),
        (math.inf, math.inf, math.inf),
    )
    for max_duration, max_characters, max_words in ladder:
        cuts = _solve_cuts(words, max_duration, max_characters, max_words)
        if cuts is not None:
            bounds = [*cuts, len(words)]
            return [words[bounds[i]:bounds[i + 1]] for i in range(len(cuts))]
    return [words]


def segment_whisper_segment(
    segment: dict[str, Any], segment_index: int = 0,
) -> list[dict[str, Any]]:
    """Convert one Whisper segment into one or more provisional lyric clips."""
    fallback = _fallback_clip(segment, segment_index)
    if fallback is None:
        return []
    words = _validated_words(segment)
    if words is None:
        return [fallback]

    spoken = segment.get("language")
    clips: list[dict[str, Any]] = []
    for chunk in split_words(words):
        clip = {
            "text": _join_tokens(word["_raw"] for word in chunk),
            "startTime": chunk[0]["startTime"],
            "endTime": chunk[-1]["endTime"],
            "segmentIndex": segment_index,
            "words": [
                {
                    "text": word["text"],
                    "startTime": word["startTime"],
                    "endTime": word["endTime"],
                    **({"agreement": word["agreement"]} if "agreement" in word else {}),
                }
                for word in chunk
            ],
        }
        agreements = [w["agreement"] for w in chunk if "agreement" in w]
        if agreements:
            clip["agreement"] = round(sum(agreements) / len(agreements), 4)
        if spoken:
            clip["language"] = spoken
        clips.append(clip)
    return clips


def _merge_text(left: str, right: str) -> str:
    return _NO_SPACE_BEFORE.sub(r"\1", f"{left.rstrip()} {right.lstrip()}").strip()


def _merge_clips(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = {
        **left,
        "text": _merge_text(left["text"], right["text"]),
        "startTime": min(left["startTime"], right["startTime"]),
        "endTime": max(left["endTime"], right["endTime"]),
    }
    if left.get("segmentIndex") != right.get("segmentIndex"):
        # A line stitched from two Whisper segments belongs to neither.
        merged.pop("segmentIndex", None)
    words = [*(left.get("words") or []), *(right.get("words") or [])]
    if words:
        merged["words"] = words
    agreements = [c["agreement"] for c in (left, right) if "agreement" in c]
    if agreements:
        merged["agreement"] = round(sum(agreements) / len(agreements), 4)
    return merged


def _merge_subminimum_clips(clips: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = list(clips)
    i = 0
    while i < len(result):
        clip = result[i]
        if clip["endTime"] - clip["startTime"] >= MIN_CLIP_DURATION:
            i += 1
            continue
        if i + 1 < len(result):
            result[i:i + 2] = [_merge_clips(clip, result[i + 1])]
        elif i > 0:
            result[i - 1:i + 1] = [_merge_clips(result[i - 1], clip)]
            i -= 1
        else:
            clip["endTime"] = _round_time(clip["startTime"] + MIN_CLIP_DURATION)
            i += 1
    return result


def hold_clip_tails(clips: list[dict[str, Any]],
                    duration: float | None = None) -> list[dict[str, Any]]:
    """Let each line stay on screen until the voice has actually stopped.

    The extension only ever consumes the silence *after* a line: it stops
    ``CLIP_HANDOVER_GAP`` short of the next line, so a line followed
    immediately by another still hands over cleanly instead of overlapping it.
    """
    for index, clip in enumerate(clips):
        limit = clip["endTime"] + CLIP_TAIL_HOLD
        if index + 1 < len(clips):
            limit = min(limit, clips[index + 1]["startTime"] - CLIP_HANDOVER_GAP)
        elif duration and duration > 0:
            limit = min(limit, duration)
        if limit > clip["endTime"]:
            clip["endTime"] = _round_time(limit)
    return clips


def normalize_whisper_segments(segments: Any,
                               duration: float | None = None) -> list[dict[str, Any]]:
    """Normalize all segments and assign final deterministic Lyrixa fields."""
    provisional: list[dict[str, Any]] = []
    if isinstance(segments, list):
        for segment_index, segment in enumerate(segments):
            if isinstance(segment, dict):
                provisional.extend(segment_whisper_segment(segment, segment_index))

    normalized = hold_clip_tails(_merge_subminimum_clips(provisional), duration)
    final: list[dict[str, Any]] = []
    for index, clip in enumerate(normalized):
        final.append({
            **clip,
            "id": clip_id(MAIN_CHANNEL, index),
            "sourceId": line_id(index),
            "sourceIndex": index,
            "layerId": MAIN_LAYER_ID,
            "createdBy": "import",
            "transitionIn": "fade",
            "transitionOut": "fade",
            "position": "center",
        })
    return final


# --------------------------------------------------------------------------- #
# Capas
# --------------------------------------------------------------------------- #
def clip_lines(whisper_data: dict[str, Any]) -> list[str]:
    """The exact lines the main layer will carry, before any translation.

    Callers translate *these*, not the raw Whisper segments: the split happens
    here, so translating anything else would produce a different number of
    lines than there are clips.  `normalize_whisper_segments` is deterministic,
    so calling it twice gives the same partition.
    """
    return [clip["text"] for clip
            in normalize_whisper_segments(whisper_data.get("segments"))]


def clip_languages(whisper_data: dict[str, Any]) -> list[str | None]:
    """The source language of each line, parallel to `clip_lines`.

    A one-hour DJ set moves between languages, so the translator routes each
    line by the language it was sung in rather than by one value for the file.
    """
    return [clip.get("language") for clip
            in normalize_whisper_segments(whisper_data.get("segments"))]


def derived_clips(clips: list[dict[str, Any]], texts: list[str], *,
                  layer_id: str, channel: str,
                  skip_identical: bool = False) -> list[dict[str, Any]]:
    """Duplica los clips principales en otro canal, sin tocar los tiempos.

    Una línea derivada —traducción o transliteración— lleva *el mismo*
    `sourceId` que la original, no su id: el id dice qué caja de la línea de
    tiempo es esto, el `sourceId` dice qué verso representa.  Emparejar las
    capas por el id funcionaba solo hasta que alguien recortaba o sustituía un
    clip en Lyrixa.

    No se copian `words[]`: el texto derivado es por línea, y fingir que
    también es por palabra inventaría fronteras que nadie midió.
    """
    out: list[dict[str, Any]] = []
    for clip, text in zip(clips, texts):
        text = (text or "").strip()
        if not text or (skip_identical and text == clip["text"]):
            continue
        out.append({
            "id": clip_id(channel, clip["sourceIndex"]),
            "text": text,
            "startTime": clip["startTime"],
            "endTime": clip["endTime"],
            "sourceId": clip["sourceId"],
            "sourceIndex": clip["sourceIndex"],
            "layerId": layer_id,
            "createdBy": "import",
            "transitionIn": "fade",
            "transitionOut": "fade",
            "position": "center",
        })
    return out


def romaji_clips(clips: list[dict[str, Any]],
                 romaji: list[str]) -> list[dict[str, Any]]:
    """La transliteración, en su propia capa.

    Antes el romaji sustituía el texto del clip principal y guardaba el
    original en `originalText`.  Leer una mezcla japonesa así obliga a elegir:
    o ves los kanji o ves cómo suenan, nunca las dos cosas.  Como capa propia
    las tres lecturas —original, traducción y romaji— conviven en pantalla y
    cada una se puede ocultar por separado.

    Las líneas que ya estaban en alfabeto latino no generan clip: su romaji es
    idéntico al original y sería una capa duplicada sin información.
    """
    return derived_clips(clips, romaji, layer_id=ROMAJI_LAYER_ID,
                         channel=ROMAJI_CHANNEL, skip_identical=True)


def translation_clips(clips: list[dict[str, Any]], translations: list[str],
                      *, language: str | None = None,
                      ) -> list[dict[str, Any]]:
    """La traducción, un clip por cada línea principal."""
    return derived_clips(
        clips, translations, layer_id=TRANSLATION_LAYER_ID,
        channel=channel_slug(language, TRANSLATION_CHANNEL_FALLBACK))


def build_layers(*, source_language: str | None, translation_language: str | None,
                 translation_name: str | None = None,
                 romanized: bool = False) -> list[dict[str, Any]]:
    """Las cuatro capas de Lyrixa, con el papel que les asigna este puente.

    Tres son de lectura —original, traducción y romaji— y la cuarta es el
    canal de efectos.  Siempre están las cuatro, incluso cuando el trabajo no
    tiene traducción ni japonés.  Un canal vacío al que arrastrar clips es más
    útil que un canal ausente, y sobre todo mantiene el `order` estable
    (0/1/2/3) en todas las exportaciones: una escena de Vibrix montada sobre
    una capa no se desplaza porque el siguiente trabajo no llevara traducción.

    `role` y `language` son campos aditivos: Lyrixa deja pasar las claves de
    capa que no conoce hasta el bundle de letras, así que LiveWallpaper puede
    distinguir el canal de traducción de uno de coros sin adivinarlo por el
    nombre.
    """
    layers: list[dict[str, Any]] = [{
        "id": MAIN_LAYER_ID,
        "name": "Letra original",
        "layerType": "lyrics",
        # `primary` y en su escritura original: los kanji no se sustituyen
        # nunca.  Cómo suenan lo dice la capa de romaji, que va aparte
        # justamente para que se puedan leer las dos a la vez.
        "role": "primary",
        "romanized": False,
        "language": source_language,
        "color": "#2e7afb",
        "visible": True,
        "locked": False,
        "order": 0,
        "renderSettings": {"positionPreset": "bottom", "textAlign": "center",
                           "zIndex": 20},
        "styleDefaults": {"fontSize": "2.8rem", "fontWeight": "800",
                          "textColor": "#ffffff"},
    }]
    layers.append({
            "id": TRANSLATION_LAYER_ID,
            "name": (translation_name
                     or (f"Traducción · {translation_language}"
                         if translation_language else "Voz secundaria")),
            "layerType": "backing",
            # Sin traducción el canal existe igual, pero no miente sobre lo
            # que lleva: `translation` activa en Lyrixa la detección de líneas
            # derivadas desfasadas, y un canal vacío no deriva de nada.
            "role": "translation" if translation_language else "backing",
            "language": translation_language,
            "color": "#8a5cf6",
            "visible": True,
            "locked": False,
            "order": 1,
            # Opposite anchor from the main layer, and never the same one.
            # Both renderers place a layer's clips absolutely at its preset, so
            # two text layers sharing `bottom` draw exactly on top of each
            # other. Smaller and dimmer besides: a translation is a reading
            # aid, not a second lead vocal.
            "renderSettings": {"positionPreset": "top", "textAlign": "center",
                               "zIndex": 18},
            "styleDefaults": {"fontSize": "1.6rem", "fontWeight": "600",
                              "textColor": "rgba(206, 220, 255, 0.9)",
                              "opacity": 0.9},
    })
    layers.append({
        "id": ROMAJI_LAYER_ID,
        "name": "Romaji",
        "layerType": "lyrics",
        # `transliteration` es el papel que Lyrixa ya conoce: un renderer que
        # oculte los canales de transliteración apaga este y deja la letra.
        "role": "transliteration",
        "romanized": True,
        "language": source_language,
        "color": "#39c2a3",
        "visible": romanized,
        "locked": False,
        "order": 2,
        # Tercer anclaje distinto: original abajo, traducción arriba y romaji
        # justo debajo del original, que es donde se lee sin tapar nada.
        "renderSettings": {"positionPreset": "bottom", "textAlign": "center",
                           "zIndex": 19, "offsetY": "-3.4rem"},
        "styleDefaults": {"fontSize": "1.5rem", "fontWeight": "600",
                          "textColor": "rgba(196, 244, 232, 0.92)",
                          "opacity": 0.9},
    })
    layers.append({
        "id": FX_LAYER_ID,
        "name": "FX / Adlibs",
        "layerType": "fx",
        "role": "fx",
        "color": "#f6a25c",
        "visible": True,
        "locked": False,
        "order": 3,
        # `suppressClipText` es lo que convierte esta capa en un canal de
        # comandos: su texto no se dibuja, así que un clip que diga
        # "glitch 0.8" es una orden para el renderer y no una línea de letra.
        # Ver docs/contrato-capa-fx.md.  Transcriptor no escribe estos clips:
        # los autora la persona en Lyrixa y los lee LiveWallpaper.
        "renderSettings": {"positionPreset": "top-right", "textAlign": "right",
                           "zIndex": 30, "suppressClipText": True},
    })
    return layers


def build_lyrixa_envelope(
    whisper_data: dict[str, Any], *, title: str, duration: float,
    audio_file_name: str | None = None, model: str | None = None,
    language: str | None = None, exported_at: str | None = None,
    project_stem: str = "transcription",
    translations: list[str] | None = None,
    translation_language: str | None = None,
    romaji: list[str] | None = None,
    precision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the schema-v1 Lyrixa authoring-project envelope."""
    safe_duration = max(0.0, _finite_number(duration) or 0.0)
    clips = normalize_whisper_segments(whisper_data.get("segments"),
                                       safe_duration or None)
    lines = [clip["text"] for clip in clips]
    translated = (translation_clips(clips, translations,
                                    language=translation_language)
                  if translations else [])
    transliterated = romaji_clips(clips, romaji) if romaji else []
    timestamp = exported_at or datetime.now(timezone.utc).isoformat()
    detected_language = language or whisper_data.get("language") or None
    audio_tracks = {
        "master": ({"fileName": audio_file_name, "duration": safe_duration}
                   if audio_file_name else None)
    }
    return {
        "schemaVersion": 1,
        "app": "Lyrixa",
        "exportedAt": timestamp,
        # Unknown envelope fields are intentionally ignored by Lyrixa's v1
        # parser.  Keeping bridge metadata here cannot be mistaken for project
        # runtime state or for the separate lyrics-bundle contract.
        "transcriptor": {
            "source": "transcriptor",
            "bridgeSchemaVersion": BRIDGE_SCHEMA_VERSION,
            "whisperModel": model,
            "language": detected_language,
            "title": title,
            "duration": safe_duration,
            "audioFileName": audio_file_name,
            "generatedAt": timestamp,
            "romanized": bool(romaji),
            "translationLanguage": translation_language,
            "precision": precision,
        },
        "project": {
            "id": f"transcriptor-{Path(project_stem).stem}",
            "name": title or Path(project_stem).stem,
            "lyricMode": "single",
            "audioTracks": audio_tracks,
            "rawLyricsText": "\n".join(lines),
            "normalizedLyrics": lines,
            "layers": build_layers(
                source_language=detected_language,
                translation_language=translation_language,
                romanized=bool(romaji),
            ),
            "clips": [*clips, *translated, *transliterated],
            "currentTime": 0,
            "renderMode": "editor",
        },
    }
