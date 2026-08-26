"""Tracklists and optional lyric references for long DJ mixes.

The timestamps supplied by a DJ are the strongest signal we have: they tell us
which vocabulary belongs in every part of a multi-hour recording.  LRCLIB is a
useful second signal, but never the authority; remixes can cut, repeat or reorder
the source song.  Consequently this module only replaces a Whisper line when
the audio-derived text already agrees strongly with the reference.
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
import time
import unicodedata
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable


LRCLIB_SEARCH = "https://lrclib.net/api/search"
USER_AGENT = "TranscriptorLocal/1.0 (local lyrics reference)"
_TIMED_LINE = re.compile(
    r"^\s*((?:\d{1,2}:){1,2}\d{2}(?:[.,]\d{1,3})?)\s+(.+?)\s*$"
)
_LRC_LINE = re.compile(r"^\[(\d{1,3}):(\d{2}(?:\.\d+)?)\]\s*(.*)$")
_CREDIT_SPLIT = re.compile(r"\s+[-–—]\s+", re.UNICODE)
_VERSION = re.compile(
    r"\s*[\[(](?:[^\])]*(?:remix|rmx|mix|edit|version|vocal|club|extended|"
    r"radio|original|promo|nightcore)[^\])]*)[\])]",
    re.IGNORECASE,
)


@dataclass
class Track:
    start: float
    artist: str
    title: str
    label: str
    end: float = 0.0
    reference: dict | None = None


@dataclass
class Section:
    name: str
    tracks: list[Track] = field(default_factory=list)


def parse_timestamp(value: str) -> float:
    parts = value.replace(",", ".").split(":")
    if len(parts) == 2:
        hours, minutes, seconds = 0, int(parts[0]), float(parts[1])
    elif len(parts) == 3:
        hours, minutes, seconds = int(parts[0]), int(parts[1]), float(parts[2])
    else:
        raise ValueError(f"Timestamp desconocido: {value}")
    return hours * 3600 + minutes * 60 + seconds


def split_credit(label: str) -> tuple[str, str]:
    parts = _CREDIT_SPLIT.split(label.strip(), maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return "", label.strip()


def parse_tracklist(text: str) -> list[Section]:
    """Parse YouTube/Pastebin-style tracklists, including named sections."""
    sections: list[Section] = []
    current: Section | None = None
    pending_name = "Tracklist"
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        match = _TIMED_LINE.match(line)
        if not match:
            if line.endswith(":") and len(line) <= 100:
                pending_name = line[:-1].strip() or "Tracklist"
                current = None
            continue
        if current is None:
            current = Section(pending_name)
            sections.append(current)
        start = parse_timestamp(match.group(1))
        label = match.group(2).strip()
        artist, title = split_credit(label)
        current.tracks.append(Track(start, artist, title, label))

    # A repeated timestamp normally means the paste contains a new unnamed
    # section. Split it instead of producing negative track durations.
    normalized: list[Section] = []
    for section in sections:
        part = Section(section.name)
        for track in section.tracks:
            if part.tracks and track.start <= part.tracks[-1].start:
                normalized.append(part)
                part = Section(f"{section.name} {len(normalized) + 1}")
            part.tracks.append(track)
        if part.tracks:
            normalized.append(part)
    return normalized


def _estimated_end(section: Section) -> float:
    starts = [track.start for track in section.tracks]
    gaps = [b - a for a, b in zip(starts, starts[1:]) if 20 <= b - a <= 900]
    tail = statistics.median(gaps) if gaps else 240.0
    return starts[-1] + tail if starts else 0.0


def choose_section(sections: list[Section], duration: float) -> Section | None:
    if not sections:
        return None
    if duration <= 0:
        return max(sections, key=lambda section: len(section.tracks))

    def score(section: Section) -> tuple[float, int]:
        if not section.tracks:
            return (float("inf"), 0)
        last = section.tracks[-1].start
        # A section whose final song begins after the audio ends cannot be the
        # right speed/version. This cleanly separates Original from Nightcore.
        impossible = duration * 2 if last >= duration else 0.0
        return (abs(_estimated_end(section) - duration) + impossible,
                -len(section.tracks))

    return min(sections, key=score)


def _fold(value: str) -> str:
    value = unicodedata.normalize("NFKD", value or "")
    value = "".join(char for char in value if not unicodedata.combining(char))
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _search_title(value: str) -> str:
    cleaned = _VERSION.sub("", value or "")
    return re.sub(r"\s+", " ", cleaned).strip(" -–—") or (value or "")


def _similarity(left: str, right: str) -> float:
    a, b = _fold(left), _fold(right)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def candidate_score(track: Track, candidate: dict) -> float:
    title = candidate.get("trackName") or candidate.get("name") or ""
    artist = candidate.get("artistName") or ""
    title_score = max(
        _similarity(track.title, title),
        _similarity(_search_title(track.title), _search_title(title)),
    )
    artist_score = _similarity(track.artist, artist) if track.artist else 0.7
    content = 0.06 if candidate.get("plainLyrics") else 0.0
    synced = 0.04 if candidate.get("syncedLyrics") else 0.0
    return title_score * 0.72 + artist_score * 0.18 + content + synced


def best_candidate(track: Track, candidates: list[dict]) -> dict | None:
    usable = [candidate for candidate in candidates if not candidate.get("instrumental")]
    if not usable:
        return None
    best = max(usable, key=lambda candidate: candidate_score(track, candidate))
    score = candidate_score(track, best)
    if score < 0.62:
        return None
    return {
        "provider": "LRCLIB",
        "id": best.get("id"),
        "artist": best.get("artistName") or "",
        "title": best.get("trackName") or best.get("name") or "",
        "duration": float(best.get("duration") or 0.0),
        "plain": best.get("plainLyrics") or "",
        "synced": best.get("syncedLyrics") or "",
        "score": round(score, 4),
    }


def _cache_path(cache_dir: Path, track: Track) -> Path:
    key = hashlib.sha256(
        f"{_fold(track.artist)}\0{_fold(_search_title(track.title))}".encode()
    ).hexdigest()[:24]
    return cache_dir / f"{key}.json"


def fetch_reference(track: Track, cache_dir: Path, timeout: float = 12.0) -> dict | None:
    """Fetch and cache one public lyric reference; failures stay non-fatal."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = _cache_path(cache_dir, track)
    if target.is_file():
        try:
            cached = json.loads(target.read_text(encoding="utf-8"))
            return cached or None
        except Exception:  # noqa: BLE001
            pass

    query = " ".join(part for part in (track.artist, _search_title(track.title)) if part)
    url = LRCLIB_SEARCH + "?" + urllib.parse.urlencode({"q": query})
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            candidates = json.loads(response.read().decode("utf-8"))
        reference = best_candidate(track, candidates if isinstance(candidates, list) else [])
    except Exception:  # noqa: BLE001
        reference = None
    target.write_text(json.dumps(reference, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    return reference


def build_guide(
    text: str,
    duration: float,
    *,
    fetch_lyrics: bool,
    cache_dir: Path,
    progress: Callable[[int, int, Track], None] | None = None,
) -> dict | None:
    sections = parse_tracklist(text)
    selected = choose_section(sections, duration)
    if selected is None:
        return None
    tracks = selected.tracks
    for index, track in enumerate(tracks):
        track.end = tracks[index + 1].start if index + 1 < len(tracks) else duration
        if track.end <= track.start:
            track.end = track.start + 240.0
        if fetch_lyrics:
            track.reference = fetch_reference(track, cache_dir)
            # LRCLIB is a community service; serialize requests instead of
            # hammering it with 50 songs at once. Cache hits skip this delay.
            time.sleep(0.18)
        if progress:
            progress(index + 1, len(tracks), track)
    found = sum(1 for track in tracks if track.reference)
    synced = sum(1 for track in tracks if track.reference and track.reference.get("synced"))
    return {
        "version": 1,
        "section": selected.name,
        "duration": duration,
        "sectionCandidates": [
            {"name": section.name, "tracks": len(section.tracks),
             "estimatedEnd": _estimated_end(section)}
            for section in sections
        ],
        "tracks": [asdict(track) for track in tracks],
        "references": {"provider": "LRCLIB", "found": found,
                       "synced": synced, "total": len(tracks)},
    }


def reference_lines(reference: dict | None) -> list[str]:
    if not reference:
        return []
    synced = reference.get("synced") or ""
    lines = []
    for raw in synced.splitlines():
        match = _LRC_LINE.match(raw.strip())
        if match and match.group(3).strip():
            lines.append(match.group(3).strip())
    if not lines:
        lines = [line.strip() for line in (reference.get("plain") or "").splitlines()
                 if line.strip()]
    return lines


def prompt_spans(guide: dict | None, max_chars: int = 1400) -> list[dict]:
    spans = []
    for track in (guide or {}).get("tracks", []):
        seen: set[str] = set()
        lines = []
        for line in reference_lines(track.get("reference")):
            key = _fold(line)
            if key and key not in seen:
                seen.add(key)
                lines.append(line)
        heading = f"{track.get('artist', '')} — {track.get('title', '')}. ".strip()
        prompt = heading + " " + " | ".join(lines)
        spans.append({
            "start": float(track.get("start") or 0.0),
            "end": float(track.get("end") or 0.0),
            "prompt": prompt[:max_chars],
        })
    return spans


def prompt_at(spans: list[dict], start: float, end: float) -> str | None:
    if not spans:
        return None
    overlap = lambda span: max(0.0, min(end, span["end"]) - max(start, span["start"]))
    winner = max(spans, key=overlap)
    return winner.get("prompt") if overlap(winner) > 0 else None


def _line_score(heard: str, reference: str) -> float:
    heard_tokens = _fold(heard).split()
    ref_tokens = _fold(reference).split()
    if not heard_tokens or not ref_tokens:
        return 0.0
    common = sum((Counter(heard_tokens) & Counter(ref_tokens)).values())
    coverage = common / len(heard_tokens)
    precision = common / len(ref_tokens)
    sequence = SequenceMatcher(None, " ".join(heard_tokens),
                               " ".join(ref_tokens)).ratio()
    return 0.45 * sequence + 0.35 * coverage + 0.20 * precision


def refine_segments(segments: list[dict], guide: dict | None) -> list[dict]:
    """Use exact lyric spelling only for lines Whisper already corroborated."""
    tracks = (guide or {}).get("tracks", [])
    if not tracks:
        return segments
    out = []
    for segment in segments:
        item = dict(segment)
        midpoint = (float(item.get("start", 0)) + float(item.get("end", 0))) / 2
        track = next((track for track in tracks
                      if float(track.get("start", 0)) <= midpoint <
                      float(track.get("end", float("inf")))), None)
        if not track:
            out.append(item)
            continue
        lines = reference_lines(track.get("reference"))
        candidates = list(lines)
        candidates.extend(
            f"{left} {right}" for left, right in zip(lines, lines[1:])
            if len(left) + len(right) < 220
        )
        if candidates:
            best = max(candidates, key=lambda line: _line_score(item.get("text", ""), line))
            score = _line_score(item.get("text", ""), best)
            heard_count = len(_fold(item.get("text", "")).split())
            ref_count = len(_fold(best).split())
            balanced = ref_count <= heard_count * 2 + 2 and heard_count <= ref_count * 2 + 2
            exact_short = heard_count <= 2 and _fold(item.get("text", "")) == _fold(best)
            if (score >= 0.76 and balanced and heard_count >= 3) or exact_short:
                item["text"] = best
                item["referenceScore"] = round(score, 4)
                item["referenceProvider"] = "LRCLIB"
        item["track"] = {
            "artist": track.get("artist", ""), "title": track.get("title", ""),
        }
        out.append(item)
    return out

