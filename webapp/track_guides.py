"""Tracklists and optional lyric references for long DJ mixes.

The timestamps supplied by a DJ are the strongest signal we have: they tell us
which vocabulary belongs in every part of a multi-hour recording.  LRCLIB is a
useful second signal, but never the authority; remixes can cut, repeat or reorder
the source song.  Consequently this module only replaces a Whisper line when
the audio-derived text already agrees strongly with the reference.
"""

from __future__ import annotations

# Interfaz externa de este módulo; lo que no está aquí es seam interno
# y solo lo tocan sus propias pruebas (docs/arquitectura.md).
__all__ = [
    "Track", "build_guide", "choose_section", "language_priors",
    "parse_tracklist", "prompt_spans", "refine_segments",
]

import hashlib
import json
import re
import ssl
import statistics
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable
from textjoin import is_no_space_char, space_prefixes


LRCLIB_SEARCH = "https://lrclib.net/api/search"
USER_AGENT = "TranscriptorLocal/1.0 (local lyrics reference)"
# Un fallo de red o una búsqueda vacía no es una verdad permanente. Los
# aciertos se conservan; los negativos se reintentan al día siguiente.
NEGATIVE_CACHE_TTL = 24 * 60 * 60
ERROR_CACHE_TTL = 5 * 60
# LRCLIB es un servicio comunitario y a veces devuelve 429/503 durante unos
# segundos. Un fallo transitorio no debe dejar media mezcla sin referencias.
SEARCH_RETRY_DELAYS = (0.8, 2.0)
RETRYABLE_HTTP_STATUS = frozenset({429, 500, 502, 503, 504})
_TIMED_LINE = re.compile(
    r"^\s*((?:\d{1,2}:){1,2}\d{2}(?:[.,]\d{1,3})?)\s+(.+?)\s*$"
)
_LRC_LINE = re.compile(r"^\[(\d{1,3}):(\d{2}(?:\.\d+)?)\]\s*(.*)$")
_CREDIT_SPLIT = re.compile(r"\s+[-–—]\s+", re.UNICODE)
# Créditos de invitados: lo que va después no es el artista principal.
_CREDIT_GUEST = re.compile(
    r"\s*(?:\bfeat\b\.?|\bft\b\.?|\bfeaturing\b|\bwith\b|\bvs\b\.?|"
    r"[,&×]|\bx\b)\s*", re.IGNORECASE)
# Umbral de coincidencia de artista. Calibrado sobre una mezcla real de 19
# temas: los aciertos quedaron en 0.81–1.00 y los falsos positivos en 0.00–0.47.
ARTIST_MIN = 0.62
# Horquilla de duración referencia/hueco en la mezcla.
DURATION_MIN_RATIO = 0.5
DURATION_MAX_RATIO = 2.5
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
    lookup: dict = field(default_factory=dict)


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


def _lyric_char(char: str) -> str:
    """Normalize one lyric character without erasing its writing system."""
    # Katakana and hiragana are alternate spellings of the same Japanese
    # reading.  Folding the common katakana block makes a sung loanword
    # comparable without transliterating it or confusing it with Hangul.
    code = ord(char)
    if 0x30A1 <= code <= 0x30F6:
        return chr(code - 0x60)
    return char.casefold()


def _lyric_units(value: str) -> list[str]:
    """Comparable lyric units that retain kana, Han and Hangul.

    Metadata matching deliberately remains Latin-only in :func:`_fold`.
    Lyrics need a different representation: CJK characters are useful units,
    while adjacent Latin letters/digits remain words so the established
    English/Spanish scoring does not become character-based overnight.
    """
    normalized = unicodedata.normalize("NFKC", value or "")
    units: list[str] = []
    latin_run: list[str] = []

    def flush() -> None:
        if latin_run:
            units.append("".join(latin_run))
            latin_run.clear()

    for char in normalized:
        category = unicodedata.category(char)
        if char.isascii() and char.isalnum():
            latin_run.append(char.casefold())
        elif category.startswith(("L", "N")):
            flush()
            units.append(_lyric_char(char))
        else:
            flush()
    flush()
    return units


def _lyric_key(value: str) -> str:
    """Stable Unicode key for lyric text; punctuation alone has no key."""
    return "\u2063".join(_lyric_units(value))


def _search_title(value: str) -> str:
    cleaned = _VERSION.sub("", value or "")
    return re.sub(r"\s+", " ", cleaned).strip(" -–—") or (value or "")


def _similarity(left: str, right: str) -> float:
    a, b = _fold(left), _fold(right)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _primary_artist(value: str) -> str:
    """El nombre principal, sin los invitados.

    "lapix feat. YUC'e" y "lapix" son el mismo artista, pero como cadenas se
    parecen poco: comparar también los nombres principales evita descartar un
    acierto por culpa de un crédito más largo.
    """
    return _CREDIT_GUEST.split(value or "", maxsplit=1)[0].strip() or (value or "")


def artist_agreement(track_artist: str, candidate_artist: str) -> float | None:
    """Cuánto coinciden dos créditos, o `None` si no se pueden comparar.

    `_fold` se queda solo con letras latinas, así que un artista escrito
    entero en japonés queda en blanco.  Ahí no hay juicio posible y devolvemos
    `None`: mejor dejar decidir a la duración que inventar un cero.
    """
    if not _fold(track_artist) or not _fold(candidate_artist):
        return None
    return max(
        _similarity(track_artist, candidate_artist),
        _similarity(_primary_artist(track_artist), _primary_artist(candidate_artist)),
    )


def duration_ratio(track: Track, candidate: dict) -> float | None:
    """Duración de la referencia partido por la del hueco en la mezcla."""
    slot = float(track.end or 0.0) - float(track.start or 0.0)
    reference = float(candidate.get("duration") or 0.0)
    if slot <= 0 or reference <= 0:
        return None
    return reference / slot


def candidate_score(track: Track, candidate: dict) -> float:
    title = candidate.get("trackName") or candidate.get("name") or ""
    artist = candidate.get("artistName") or ""
    title_score = max(
        _similarity(track.title, title),
        _similarity(_search_title(track.title), _search_title(title)),
    )
    agreement = artist_agreement(track.artist, artist) if track.artist else None
    artist_score = 0.7 if agreement is None else agreement
    content = 0.06 if candidate.get("plainLyrics") else 0.0
    synced = 0.04 if candidate.get("syncedLyrics") else 0.0
    return title_score * 0.72 + artist_score * 0.18 + content + synced


def plausible_candidate(track: Track, candidate: dict) -> bool:
    """Filtros duros antes de aceptar una referencia.

    El título solo no basta.  "Ariel" existe en LRCLIB como canción de Dean
    Friedman (1977, en inglés) y ese texto acabaría de `initial_prompt` sobre
    un tema japonés, que es exactamente cómo Whisper empieza a transcribir
    fonéticamente en inglés algo que se canta en japonés.  Una referencia
    equivocada hace más daño que ninguna, así que en la duda se descarta.
    """
    agreement = artist_agreement(track.artist, candidate.get("artistName") or "")
    if track.artist and agreement is not None and agreement < ARTIST_MIN:
        return False
    ratio = duration_ratio(track, candidate)
    # Un remix corta o alarga, pero no convierte un tema de 3:30 en uno de 32
    # segundos: fuera de esta horquilla no es la misma grabación.
    if ratio is not None and not (DURATION_MIN_RATIO <= ratio <= DURATION_MAX_RATIO):
        return False
    return True


def still_plausible(track: Track, reference: dict | None) -> bool:
    """Revalida una referencia que ya estaba en caché.

    El caché guarda la elección, no los candidatos, así que un acierto
    dudoso de una versión anterior volvería tal cual y se saltaría los
    filtros.  Comprobarlo al leer cuesta nada y repara también los trabajos
    ya procesados, sin pedir a nadie que borre la caché a mano.
    """
    if not reference:
        return True
    return plausible_candidate(track, {
        "artistName": reference.get("artist") or "",
        "trackName": reference.get("title") or "",
        "duration": reference.get("duration"),
    })


def best_candidate(track: Track, candidates: list[dict]) -> dict | None:
    usable = [candidate for candidate in candidates
              if not candidate.get("instrumental")
              and plausible_candidate(track, candidate)]
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


def dedupe_candidates(candidates: list[dict]) -> list[dict]:
    """Collapse LRCLIB duplicates, preferring an entry with synced lyrics.

    Adapted from the useful part of search-play-lyrics: LRCLIB commonly returns
    the same upload several times and selection should not count that as
    independent evidence.
    """
    chosen: dict[tuple[str, str], dict] = {}
    order: list[tuple[str, str]] = []
    for candidate in candidates:
        key = (_fold(candidate.get("trackName") or candidate.get("name") or ""),
               _fold(candidate.get("artistName") or ""))
        if key not in chosen:
            chosen[key] = candidate
            order.append(key)
        elif (not chosen[key].get("syncedLyrics")
              and candidate.get("syncedLyrics")):
            chosen[key] = candidate
    return [chosen[key] for key in order]


def _https_context() -> ssl.SSLContext:
    """Use certifi on Python.org macOS builds whose OpenSSL store is empty."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001
        return ssl.create_default_context()


def _search_candidates(query: str, timeout: float) -> list[dict]:
    url = LRCLIB_SEARCH + "?" + urllib.parse.urlencode({"q": query})
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(len(SEARCH_RETRY_DELAYS) + 1):
        try:
            with urllib.request.urlopen(
                request, timeout=timeout, context=_https_context()) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as error:
            retryable = error.code in RETRYABLE_HTTP_STATUS
            if not retryable or attempt >= len(SEARCH_RETRY_DELAYS):
                raise
            retry_after = error.headers.get("Retry-After") if error.headers else None
            try:
                delay = float(retry_after) if retry_after is not None else 0.0
            except ValueError:
                delay = 0.0
            # Nunca martillar el servicio, pero tampoco congelar el pipeline
            # por una cabecera Retry-After absurda.
            time.sleep(min(15.0, max(SEARCH_RETRY_DELAYS[attempt], delay)))
    return dedupe_candidates(payload if isinstance(payload, list) else [])


def _cache_path(cache_dir: Path, track: Track) -> Path:
    key = hashlib.sha256(
        f"{_fold(track.artist)}\0{_fold(_search_title(track.title))}".encode()
    ).hexdigest()[:24]
    return cache_dir / f"{key}.json"


def fetch_reference(track: Track, cache_dir: Path, timeout: float = 12.0,
                    *, retry_errors: bool = False) -> dict | None:
    """Fetch and cache one public lyric reference; failures stay non-fatal."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = _cache_path(cache_dir, track)
    if target.is_file():
        try:
            cached = json.loads(target.read_text(encoding="utf-8"))
            # Acepta el formato histórico (dict/null) y el formato actual, que
            # distingue resultado vacío de error temporal y les pone caducidad.
            if isinstance(cached, dict) and "reference" in cached:
                reference = cached.get("reference")
                fetched_at = float(cached.get("fetchedAt") or 0.0)
                age = time.time() - fetched_at
                ttl = ERROR_CACHE_TTL if cached.get("error") else NEGATIVE_CACHE_TTL
                # La versión afectada por el certificado solo guardaba el
                # nombre genérico "URLError". Ese valor no sirve para decidir
                # nada y debe reintentarse inmediatamente.
                legacy_ssl_failure = cached.get("error") == "URLError"
                if reference and not still_plausible(track, reference):
                    reference = None    # elección de una versión sin filtros
                elif reference or (
                    age < ttl
                    and not legacy_ssl_failure
                    and not (retry_errors and cached.get("error"))
                ):
                    track.lookup = {
                        "status": "found" if reference else
                                  "error" if cached.get("error") else "no_match",
                        "candidates": int(cached.get("candidateCount") or 0),
                        "error": cached.get("error"),
                    }
                    return reference or None
            elif cached and still_plausible(track, cached):
                track.lookup = {"status": "found", "candidates": 1,
                                "error": None}
                return cached
            # Un `null` del formato viejo no dice si fue "sin resultados" o
            # "sin internet"; se reintenta una vez y queda migrado.
        except Exception:  # noqa: BLE001
            pass

    title = _search_title(track.title)
    queries = list(dict.fromkeys(filter(None, (
        " ".join(part for part in (track.artist, title) if part),
        title,
    ))))
    error = None
    candidates: list[dict] = []
    reference = None
    try:
        for query in queries:
            candidates = _search_candidates(query, timeout)
            reference = best_candidate(track, candidates)
            if reference:
                break
    except Exception as caught:  # noqa: BLE001
        reference = None
        reason = getattr(caught, "reason", caught)
        error = f"{type(caught).__name__}: {reason}"
    track.lookup = {
        "status": "found" if reference else "error" if error else "no_match",
        "candidates": len(candidates), "error": error,
    }
    target.write_text(json.dumps({
        "reference": reference,
        "fetchedAt": time.time(),
        "error": error,
        "candidateCount": len(candidates),
        "queries": queries,
    }, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    return reference


def build_guide(
    text: str,
    duration: float,
    *,
    fetch_lyrics: bool,
    cache_dir: Path,
    progress: Callable[[int, int, Track], None] | None = None,
    retry_errors: bool = False,
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
            track.reference = fetch_reference(
                track, cache_dir, retry_errors=retry_errors)
            # LRCLIB is a community service; serialize requests instead of
            # hammering it with 50 songs at once. Cache hits skip this delay.
            time.sleep(0.18)
        if progress:
            progress(index + 1, len(tracks), track)
    found = sum(1 for track in tracks if track.reference)
    synced = sum(1 for track in tracks if track.reference and track.reference.get("synced"))
    failed = sum(1 for track in tracks if track.lookup.get("status") == "error")
    no_match = sum(1 for track in tracks if track.lookup.get("status") == "no_match")
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
                       "synced": synced, "total": len(tracks),
                       "failed": failed, "noMatch": no_match},
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


# --------------------------------------------------------------------------- #
# Pistas de idioma a partir del tracklist
# --------------------------------------------------------------------------- #
_KANA = re.compile(r"[\u3040-\u309f\u30a0-\u30ff]")
_HAN = re.compile(r"[\u4e00-\u9fff]")
_HANGUL = re.compile(r"[\uac00-\ud7af\u1100-\u11ff]")


def script_hint(text: str) -> str | None:
    """El idioma que delata la escritura de un crédito o de una letra.

    Kana es japonés sin discusión.  Han a secas sería ambiguo entre japonés y
    chino, pero en este material (doujin, core, Vocaloid) el chino es raro y el
    japonés es la norma, así que se resuelve a japonés; quien tenga una mezcla
    china fija el idioma a mano.
    """
    if not text:
        return None
    if _HANGUL.search(text):
        return "ko"
    if _KANA.search(text) or _HAN.search(text):
        return "ja"
    return None


def language_priors(guide: dict | None) -> list[dict]:
    """Qué idioma se espera en cada tramo, según quién firma la pista.

    El tracklist lo escribe una persona que sabe qué puso: si el crédito está
    en japonés, ese tramo se canta en japonés.  Es una evidencia más firme que
    el detector de Whisper, que sobre un stem con instrumental colado tiende a
    responder "inglés" para casi cualquier cosa — y una vez que responde
    inglés, transcribe el japonés fonéticamente y la letra se pierde entera.
    """
    tracks = (guide or {}).get("tracks") or []
    out: list[dict] = []
    for track in tracks:
        label = track.get("label") or ""
        reference = track.get("reference") or {}
        hint = script_hint(label) or script_hint(reference.get("plain") or "")
        if not hint:
            continue
        start, end = float(track.get("start") or 0.0), float(track.get("end") or 0.0)
        if end > start:
            out.append({"start": start, "end": end, "language": hint})
    return out


def prompt_spans(guide: dict | None, max_chars: int = 1400) -> list[dict]:
    spans = []
    for track in (guide or {}).get("tracks", []):
        seen: set[str] = set()
        lines = []
        for line in reference_lines(track.get("reference")):
            key = _lyric_key(line)
            if key and key not in seen:
                seen.add(key)
                lines.append(line)
        # `initial_prompt` es texto que Whisper puede repetir. Artista y título
        # son metadatos, no letra; si LRCLIB no halló versos no hay prompt.
        if not lines:
            continue
        prompt = " | ".join(lines)
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
    heard_tokens = _lyric_units(heard)
    ref_tokens = _lyric_units(reference)
    if not heard_tokens or not ref_tokens:
        return 0.0
    common = sum((Counter(heard_tokens) & Counter(ref_tokens)).values())
    coverage = common / len(heard_tokens)
    precision = common / len(ref_tokens)
    sequence = SequenceMatcher(None, heard_tokens, ref_tokens).ratio()
    score = 0.45 * sequence + 0.35 * coverage + 0.20 * precision
    # A short substitution inside an otherwise identical CJK line often
    # represents Whisper spelling a reading phonetically.  Reward matching
    # context at both edges very slightly; unrelated lines still score far
    # below the acceptance gate.
    if (heard_tokens[0] == ref_tokens[0]
            and heard_tokens[-1] == ref_tokens[-1]
            and min(coverage, precision) >= 0.70):
        score += 0.02
    return min(score, 1.0)


def _reference_tokens(reference: str, words: list[dict]) -> list[str]:
    """Split a reference without turning a whole Japanese line into one word.

    Whisper already measured useful acoustic intervals.  For no-space scripts
    we partition the corrected surface across those intervals, weighted by
    how much source text each interval carried.  This preserves the exact
    reference string and the measured first/last boundaries.
    """
    spaced = [token for token in reference.split() if token.strip()]
    if (len(spaced) > 1 or not words
            or not any(is_no_space_char(char) for char in reference)):
        return spaced
    chars = list(reference.strip())
    if not chars:
        return []
    count = min(len(words), len(chars))
    if count == 1:
        return ["".join(chars)]
    weights = [max(len(_lyric_units(word.get("word", ""))), 1)
               for word in words[:count]]
    total_weight = sum(weights)
    cuts = [0]
    consumed = 0
    for index, weight in enumerate(weights[:-1], start=1):
        consumed += weight
        target = round(len(chars) * consumed / total_weight)
        minimum = cuts[-1] + 1
        maximum = len(chars) - (count - index)
        cuts.append(min(max(target, minimum), maximum))
    cuts.append(len(chars))
    return ["".join(chars[left:right]) for left, right in zip(cuts, cuts[1:])]


def _resync_words(item: dict, reference: str) -> None:
    """Rewrite `words[]` so its tokens are the reference's, at the heard times.

    A corrected line is worthless to Lyrixa if `words[]` still holds the old
    tokens: the bridge builds clips from words, so the export would carry the
    superseded text.  Tokens are aligned on their folded form — the same
    comparison `_line_score` already accepted — so matched reference words
    inherit the time of the word they replace, unmatched ones interpolate
    between their nearest anchors, and heard words the reference rejects are
    dropped: the acceptance gate already decided the reference is the better
    surface.  A line whose script `_fold` cannot transliterate (kana, Cyrillic)
    folds to nothing, and gets its times redistributed proportionally instead.
    """
    words = item.get("words") or []
    tokens = _reference_tokens(reference, words)
    if not words or not tokens:
        return

    heard_start = [float(w.get("start", 0.0)) for w in words]
    heard_end = [float(w.get("end", 0.0)) for w in words]

    heard_keys = [_fold(word.get("word", "")) for word in words]
    ref_keys = [_fold(token) for token in tokens]
    anchor_of = (_align_keys(heard_keys, ref_keys)
                 if any(heard_keys) and any(ref_keys) else {})
    m = len(tokens)
    token_start = [0.0] * m
    token_end = [0.0] * m
    if not anchor_of:
        if m == len(words):
            token_start[:] = heard_start
            token_end[:] = heard_end
        else:
            full_start, full_end = heard_start[0], heard_end[-1]
            step = max(full_end - full_start, 0.02 * m) / m
            for index in range(m):
                token_start[index] = full_start + step * index
                token_end[index] = (full_end if index == m - 1
                                    else full_start + step * (index + 1))
    for ref_index, heard_index in anchor_of.items():
        token_start[ref_index] = heard_start[heard_index]
        token_end[ref_index] = heard_end[heard_index]

    # Unmatched reference words split the gap between their nearest anchors.
    if anchor_of:
        last = -1
        for ref_index in range(m):
            if ref_index in anchor_of:
                last = ref_index
                continue
            nxt = next((k for k in range(ref_index + 1, m) if k in anchor_of), None)
            left_end = (heard_end[anchor_of[last]] if last >= 0
                        else heard_start[0])
            right_start = (heard_start[anchor_of[nxt]] if nxt is not None
                           else heard_end[-1])
            run = (nxt if nxt is not None else m) - last
            offset = ref_index - last
            span = max(right_start - left_end, 0.02 * run)
            token_start[ref_index] = left_end + span * (offset - 1) / run
            token_end[ref_index] = left_end + span * offset / run

    # Keep the timeline strictly increasing: interpolation can make neighbours
    # touch, and downstream consumers expect a monotone word stream.
    for index in range(1, m):
        if token_start[index] < token_end[index - 1]:
            token_start[index] = token_end[index - 1]
        if token_end[index] <= token_start[index]:
            token_end[index] = token_start[index] + 0.02

    new_words = []
    for index, (prefix, token) in enumerate(zip(space_prefixes(tokens), tokens)):
        source = words[anchor_of[index]] if index in anchor_of else None
        new_words.append({
            "word": f"{prefix}{token}",
            "start": round(token_start[index], 3),
            "end": round(token_end[index], 3),
            # A word the reference supplied but no pass matched keeps average
            # evidence, never full confidence: it stays contestable downstream.
            "probability": (source.get("probability", 0.5) if source
                            else float(item.get("confidence", 0.5))),
            "agreement": (source.get("agreement", 0.5) if source
                          else float(item.get("agreement", 0.5))),
            "votes": source.get("votes", 0) if source else 0,
        })
    item["words"] = new_words


def _align_keys(heard: list[str], reference: list[str]) -> dict[int, int]:
    """Monotone matching ref-index → heard-index over folded keys.

    Longest increasing chain anchored by exact key pairs: a word aligns only
    if it is *the same word*, never a similar-but-different one, so a
    correction like goodbay→goodbye deliberately finds no anchor and gets an
    interpolated time instead of inheriting a mismatched word's confidence.
    Empty keys (pure punctuation, non-Latin scripts) match nothing.
    """
    by_key: dict[str, list[int]] = {}
    for index, key in enumerate(heard):
        if key:
            by_key.setdefault(key, []).append(index)
    anchors: dict[int, int] = {}
    chain_end = -1
    for ref_index, key in enumerate(reference):
        if not key:
            continue
        heard_index = next((h for h in by_key.get(key, ()) if h > chain_end), None)
        if heard_index is not None:
            anchors[ref_index] = heard_index
            chain_end = heard_index
    return anchors


def refine_segments(segments: list[dict], guide: dict | None) -> list[dict]:
    """Use exact lyric spelling only for lines Whisper already corroborated."""
    tracks = (guide or {}).get("tracks", [])
    if not tracks:
        return segments
    out = []
    for segment in segments:
        item = dict(segment)
        midpoint = (float(item.get("start", 0)) + float(item.get("end", 0))) / 2
        track_index = next((index for index, candidate in enumerate(tracks)
                            if float(candidate.get("start", 0)) <= midpoint <
                            float(candidate.get("end", float("inf")))), None)
        if track_index is None:
            out.append(item)
            continue
        track = tracks[track_index]
        lines = reference_lines(track.get("reference"))
        candidates = list(lines)
        candidates.extend(
            f"{left} {right}" for left, right in zip(lines, lines[1:])
            if len(left) + len(right) < 220
        )
        if candidates:
            best = max(candidates, key=lambda line: _line_score(item.get("text", ""), line))
            score = _line_score(item.get("text", ""), best)
            heard_count = len(_lyric_units(item.get("text", "")))
            ref_count = len(_lyric_units(best))
            balanced = ref_count <= heard_count * 2 + 2 and heard_count <= ref_count * 2 + 2
            # Una línea que `_fold` deja vacía (japonés, coreano…) no es
            # "corta y exacta": "" == "" la cambiaba por la primera línea.
            exact_short = (1 <= heard_count <= 2
                           and _lyric_key(item.get("text", "")) == _lyric_key(best))
            if (score >= 0.76 and balanced and heard_count >= 3) or exact_short:
                item["text"] = best
                item["referenceScore"] = round(score, 4)
                item["referenceProvider"] = "LRCLIB"
                # The words are what Lyrixa cuts clips from: replacing only
                # `text` used to export the superseded tokens at the new
                # line's boundaries, silently.
                _resync_words(item, best)
        item["track"] = {
            "id": (track.get("id") or
                   f"track-{track_index:04d}-{round(float(track.get('start', 0)) * 1000)}"),
            "artist": track.get("artist", ""), "title": track.get("title", ""),
        }
        out.append(item)
    return out
