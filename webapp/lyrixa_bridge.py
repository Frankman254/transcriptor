"""Pure Whisper -> Lyrixa project conversion.

The raw Whisper JSON is an input only.  This module never mutates it and has
no dependency on FastAPI, Whisper, or the filesystem, which keeps the timing
normalization deterministic and easy to test.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


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
PAUSE_THRESHOLD = 0.45

BRIDGE_SCHEMA_VERSION = 1

_STRONG_PUNCTUATION = re.compile(r"[.!?…][\"'’”)\]}]*$")
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
    raw = list(tokens)
    if not raw:
        return ""
    # Whisper normally includes the leading space in every token after the
    # first.  Hand-authored fixtures and other Whisper frontends often do not.
    if any(token[:1].isspace() for token in raw[1:]):
        return "".join(raw).strip()
    return _NO_SPACE_BEFORE.sub(r"\1", " ".join(t.strip() for t in raw)).strip()


def _fallback_clip(segment: dict[str, Any], segment_index: int) -> dict[str, Any] | None:
    text = str(segment.get("text") or "").strip()
    if not text:
        return None
    start = max(0.0, _finite_number(segment.get("start")) or 0.0)
    end = _finite_number(segment.get("end"))
    if end is None or end <= start:
        end = start + MIN_CLIP_DURATION
    return {
        "text": text,
        "startTime": _round_time(start),
        "endTime": _round_time(end),
        "sourceId": f"segment-{segment_index}",
    }


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
        words.append({
            "text": token.strip(),
            "_raw": token,
            "startTime": _round_time(start),
            "endTime": _round_time(end),
        })
        previous_end = end

    if not words:
        return None

    segment_text = str(segment.get("text") or "")
    if _canonical_text(_join_tokens(raw_tokens)) != _canonical_text(segment_text):
        # The word list is incomplete, so segmenting it would lose source text.
        return None
    return words


def _boundary_priority(words: list[dict[str, Any]], start: int, end: int) -> int:
    token = words[end]["text"]
    if _STRONG_PUNCTUATION.search(token):
        return 4
    gap = words[end + 1]["startTime"] - words[end]["endTime"]
    if gap >= PAUSE_THRESHOLD:
        return 3
    if _WEAK_PUNCTUATION.search(token):
        return 2

    count = end - start + 1
    characters = len(_join_tokens(w["_raw"] for w in words[start:end + 1]))
    if count >= MAX_WORDS or characters >= MAX_CHARACTERS:
        return 1
    return 0


def _choose_cut(words: list[dict[str, Any]], start: int) -> int:
    last = len(words) - 1
    remaining_duration = words[last]["endTime"] - words[start]["startTime"]
    remaining_text = _join_tokens(w["_raw"] for w in words[start:])
    if (remaining_duration <= TARGET_MAX_DURATION
            and last - start + 1 <= MAX_WORDS
            and len(remaining_text) <= MAX_CHARACTERS):
        return last

    candidates: list[tuple[int, float, int]] = []
    for end in range(start, last):
        duration = words[end]["endTime"] - words[start]["startTime"]
        if duration > HARD_MAX_DURATION:
            break
        if duration < TARGET_MIN_DURATION:
            continue
        priority = _boundary_priority(words, start, end)
        if priority:
            candidates.append((priority, abs(duration - TARGET_DURATION), end))

    if candidates:
        # Natural-boundary class wins first, then proximity to the ideal length.
        priority = max(item[0] for item in candidates)
        return min(
            (item for item in candidates if item[0] == priority),
            key=lambda item: (item[1], item[2]),
        )[2]

    # No linguistic/size signal was available: this is the hard-duration cut.
    # Prefer the latest real boundary within the 6 s target and only stretch to
    # 8 s when no such boundary exists.  An individual long word cannot split.
    cut = start
    found_target = False
    for end in range(start, last):
        duration = words[end]["endTime"] - words[start]["startTime"]
        if duration <= TARGET_MAX_DURATION:
            cut = end
            found_target = True
        elif duration <= HARD_MAX_DURATION and not found_target:
            cut = end
        else:
            break
    return cut


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

    clips: list[dict[str, Any]] = []
    start = 0
    while start < len(words):
        end = _choose_cut(words, start)
        chunk = words[start:end + 1]
        clips.append({
            "text": _join_tokens(word["_raw"] for word in chunk),
            "startTime": chunk[0]["startTime"],
            "endTime": chunk[-1]["endTime"],
            "sourceId": f"segment-{segment_index}",
            "words": [
                {
                    "text": word["text"],
                    "startTime": word["startTime"],
                    "endTime": word["endTime"],
                }
                for word in chunk
            ],
        })
        start = end + 1
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
    if left.get("sourceId") != right.get("sourceId"):
        merged.pop("sourceId", None)
    words = [*(left.get("words") or []), *(right.get("words") or [])]
    if words:
        merged["words"] = words
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


def normalize_whisper_segments(segments: Any) -> list[dict[str, Any]]:
    """Normalize all segments and assign final deterministic Lyrixa fields."""
    provisional: list[dict[str, Any]] = []
    if isinstance(segments, list):
        for segment_index, segment in enumerate(segments):
            if isinstance(segment, dict):
                provisional.extend(segment_whisper_segment(segment, segment_index))

    normalized = _merge_subminimum_clips(provisional)
    final: list[dict[str, Any]] = []
    for index, clip in enumerate(normalized):
        final.append({
            "id": f"clip-{index:04d}",
            **clip,
            "sourceIndex": index,
            "layerId": "layer-main",
            "createdBy": "import",
            "transitionIn": "fade",
            "transitionOut": "fade",
            "position": "center",
        })
    return final


def build_lyrixa_envelope(
    whisper_data: dict[str, Any], *, title: str, duration: float,
    audio_file_name: str | None = None, model: str | None = None,
    language: str | None = None, exported_at: str | None = None,
    project_stem: str = "transcription",
) -> dict[str, Any]:
    """Build the schema-v1 Lyrixa authoring-project envelope."""
    clips = normalize_whisper_segments(whisper_data.get("segments"))
    lines = [clip["text"] for clip in clips]
    timestamp = exported_at or datetime.now(timezone.utc).isoformat()
    detected_language = language or whisper_data.get("language") or None
    safe_duration = max(0.0, _finite_number(duration) or 0.0)
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
        },
        "project": {
            "id": f"transcriptor-{Path(project_stem).stem}",
            "name": title or Path(project_stem).stem,
            "lyricMode": "single",
            "audioTracks": audio_tracks,
            "rawLyricsText": "\n".join(lines),
            "normalizedLyrics": lines,
            "clips": clips,
            "currentTime": 0,
            "renderMode": "editor",
        },
    }
