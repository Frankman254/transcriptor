"""Local translation and romanization for the second lyric layer.

Everything runs on this machine: Ollama for the language work, pykakasi as the
offline fallback for Japanese readings.  Nothing is sent to a cloud API, which
is the same promise the transcription side makes.

Two rules shape the whole module:

1. **Line count is sacred.**  Every lyric line has a clip with its own
   millisecond timing.  A translation that merges or splits lines silently
   desynchronizes the layer, so a batch whose line count does not come back
   exactly is retried, then split, and finally falls back to the original text
   for the specific lines that failed — never dropped, never merged.
2. **Non-Latin sources go through English.**  Russian → English → Spanish beats
   Russian → Spanish on every local model small enough to run here, because
   English is where the pivot's training mass is.
"""

from __future__ import annotations

# Interfaz externa de este módulo; lo que no está aquí es seam interno
# y solo lo tocan sus propias pruebas (docs/arquitectura.md).
__all__ = [
    "DEFAULT_CACHE_DIR", "LANGUAGE_NAMES", "ROMANIZABLE",
    "TranslationUnavailable", "language_label", "normalize_code", "pick_model",
    "plan_route", "romanize", "route_label", "translate",
    "translate_clip_lines", "translation_available",
]

import hashlib
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

import paths

OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_TIMEOUT = 600

# Preference order.  Bigger multilingual instruct models first; the list is
# matched by prefix so a local tag like "qwen3.5:9b" still resolves.
MODEL_PREFERENCE = (
    "qwen3.8", "qwen3.5", "qwen3", "qwen2.5", "gemma3", "llama3.3", "llama3.2",
    "llama3.1", "mistral", "gpt-4o", "devstral",
)

# Lines per request.  Small enough that a model reliably keeps the numbering,
# large enough that a 600-line lyric is ~40 requests instead of 600.
BATCH_SIZE = 32

LANGUAGE_NAMES = {
    "es": "Spanish", "en": "English", "pt": "Portuguese", "fr": "French",
    "it": "Italian", "de": "German", "ja": "Japanese", "ko": "Korean",
    "zh": "Chinese", "ru": "Russian", "uk": "Ukrainian", "ar": "Arabic",
    "he": "Hebrew", "hi": "Hindi", "th": "Thai", "el": "Greek",
    "tr": "Turkish", "pl": "Polish", "nl": "Dutch", "sv": "Swedish",
    "id": "Indonesian", "vi": "Vietnamese", "fa": "Persian", "ro": "Romanian",
    "ca": "Catalan", "fi": "Finnish", "da": "Danish", "no": "Norwegian",
    "cs": "Czech", "hu": "Hungarian", "bg": "Bulgarian", "sr": "Serbian",
    "ta": "Tamil", "yue": "Cantonese",
}

# Spanish labels for the UI.
LANGUAGE_LABELS_ES = {
    "es": "Español", "en": "Inglés", "pt": "Portugués", "fr": "Francés",
    "it": "Italiano", "de": "Alemán", "ja": "Japonés", "ko": "Coreano",
    "zh": "Chino", "ru": "Ruso", "uk": "Ucraniano", "ar": "Árabe",
    "he": "Hebreo", "hi": "Hindi", "th": "Tailandés", "el": "Griego",
    "tr": "Turco", "pl": "Polaco", "nl": "Neerlandés", "sv": "Sueco",
    "id": "Indonesio", "vi": "Vietnamita", "fa": "Persa", "ro": "Rumano",
    "ca": "Catalán", "fi": "Finés", "da": "Danés", "no": "Noruego",
    "cs": "Checo", "hu": "Húngaro", "bg": "Búlgaro", "sr": "Serbio",
    "ta": "Tamil", "yue": "Cantonés",
}

# Sources whose script is not Latin: these pivot through English.
NON_LATIN_SOURCES = {
    "ru", "uk", "bg", "sr", "mk", "be", "ja", "zh", "yue", "ko", "ar", "he",
    "fa", "ur", "hi", "bn", "ta", "te", "th", "lo", "km", "my", "el", "hy",
    "ka", "am", "yi", "ps", "sd", "ne", "si", "mr", "pa", "gu", "kn", "ml",
}

# Languages that need transliteration to be readable in Latin script.
ROMANIZABLE = {"ja"}

# The caller does not know what language the text is in.  Routine once
# translation is a service: the transcription knew the language of every line,
# an editor asking to re-translate one edited line usually does not.
AUTO_SOURCE = "auto"
# What the codes nobody should treat as a language look like in the wild.
_UNKNOWN_CODES = {"auto", "und", "unknown", "detect"}

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
# `1| text`, and the shapes models drift into: `1. text`, `1) text`,
# `[1] text`, `(1) text`.  A bare number needs a separator after it so that a
# lyric line beginning with a year is not mistaken for a numbered row.
_NUMBERED = re.compile(
    r"^\s*(?:"
    r"[\[(]\s*(?P<bracketed>\d{1,4})\s*[\])]\s*[|.:\-–]?"   # [1] / (1) / [1]:
    r"|(?P<bare>\d{1,4})\s*[|.):\-–]"                        # 1| / 1. / 1)
    r")\s?(?P<text>.*)$"
)
_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*$")


class TranslationUnavailable(RuntimeError):
    """No local translation backend could be reached."""


# --------------------------------------------------------------------------- #
# Ollama
# --------------------------------------------------------------------------- #
def _post(path: str, payload: dict, timeout: int = OLLAMA_TIMEOUT) -> dict:
    request = urllib.request.Request(
        f"{OLLAMA_HOST}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def available_models() -> list[str]:
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=4) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception:  # noqa: BLE001
        return []
    return [str(model.get("name", "")) for model in data.get("models", [])
            if model.get("name")]


def pick_model(models: list[str] | None = None) -> str | None:
    """Choose a capable, responsive local model for short lyric lines.

    A 27B model is not automatically the best choice here: most of the wall
    clock is autoregressive decoding, while the task is tightly constrained.
    Prefer a modern 4B–14B multilingual model and keep a larger one as fallback.
    """
    models = available_models() if models is None else models
    if not models:
        return None

    def score(name: str) -> tuple[int, int, float, str]:
        lower = name.lower()
        family = next(
            (index for index, prefix in enumerate(MODEL_PREFERENCE)
             if prefix in lower),
            len(MODEL_PREFERENCE),
        )
        match = re.search(r"(?:^|[-_:])([0-9]+(?:\.[0-9]+)?)b(?:[-_:]|$)", lower)
        size = float(match.group(1)) if match else 0.0
        if 4 <= size <= 14:
            size_class = 0
        elif 0 < size < 20:
            size_class = 1
        elif size == 0:
            size_class = 2
        else:
            size_class = 3
        return size_class, family, abs(size - 8.0) if size else 99.0, lower

    return min(models, key=score)


def translation_available() -> bool:
    return bool(available_models())


def _chat(model: str, system: str, user: str, *, temperature: float = 0.0) -> str:
    data = _post("/api/chat", {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        # Reasoning models otherwise spend the whole budget thinking about a
        # four-word lyric line.
        "think": False,
        "keep_alive": "30m",
        "options": {"temperature": temperature, "num_ctx": 8192},
    })
    content = (data.get("message") or {}).get("content", "")
    return _THINK.sub("", str(content)).strip()


# --------------------------------------------------------------------------- #
# Parseo estricto de lotes
# --------------------------------------------------------------------------- #
def parse_numbered(reply: str, expected: int) -> dict[int, str]:
    """Read back a `1| text` block, tolerating the ways models decorate it."""
    out: dict[int, str] = {}
    for raw in reply.splitlines():
        if _FENCE.match(raw):
            continue
        match = _NUMBERED.match(raw)
        if not match:
            continue
        index = int(match.group("bracketed") or match.group("bare"))
        # An index outside the batch would shift every line after it, so it is
        # discarded and the caller retries that line on its own.
        if 1 <= index <= expected:
            out[index] = match.group("text").strip()
    return out


def _language_name(code: str) -> str:
    """The name the prompt uses for a language code."""
    if code == AUTO_SOURCE:
        return "their original language"
    return LANGUAGE_NAMES.get(code, code)


def _system_prompt(source_name: str, target_name: str) -> str:
    return (
        f"You translate song lyrics from {source_name} to {target_name}.\n"
        "Rules you must follow exactly:\n"
        f"1. The input has N numbered lines. Output exactly N lines, "
        "each formatted as `<number>| <translation>`.\n"
        "2. Never merge, split, reorder, add, or omit a line. Line 3 of the "
        "output must be the translation of line 3 of the input.\n"
        "3. Translate meaning, not word order. Keep it singable and natural; "
        "these are lyrics, not a manual.\n"
        "4. Keep proper names, and keep an untranslatable interjection "
        "(oh, yeah, la la) as it is.\n"
        f"5. If a line is already {target_name}, repeat it unchanged.\n"
        "6. Output only the numbered lines. No preface, no notes, no code "
        "fences, no explanations."
    )


_ROMAJI_SYSTEM = (
    "You transliterate Japanese song lyrics into Hepburn romaji.\n"
    "Rules you must follow exactly:\n"
    "1. The input has N numbered lines. Output exactly N lines, each formatted "
    "as `<number>| <romaji>`.\n"
    "2. Use the reading the words actually have in context: 君 as `kimi` when "
    "it means 'you', 名 as `na` in 名前. Particles は/へ/を are written "
    "`wa`, `e`, `o`.\n"
    "3. Latin words already in the line stay as they are.\n"
    "4. Lowercase except for proper names. Keep the punctuation.\n"
    "5. Output only the numbered lines. No preface, no notes, no explanations."
)


def _render_batch(lines: list[str]) -> str:
    return "\n".join(f"{i}| {line}" for i, line in enumerate(lines, start=1))


def _run_batch(model: str, system: str, lines: list[str]) -> list[str | None]:
    """One batch, with a stricter retry.  ``None`` marks a line still missing."""
    for attempt, temperature in enumerate((0.0, 0.2)):
        user = _render_batch(lines)
        if attempt:
            user = (f"{user}\n\nReturn exactly {len(lines)} lines, numbered 1 "
                    f"to {len(lines)}, in the format `<number>| <text>`.")
        try:
            reply = _chat(model, system, user, temperature=temperature)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            continue
        parsed = parse_numbered(reply, len(lines))
        if len(parsed) == len(lines):
            return [parsed[i] for i in range(1, len(lines) + 1)]
        if attempt:  # keep the partial result from the last try
            return [parsed.get(i) for i in range(1, len(lines) + 1)]
    return [None] * len(lines)


def _run_lines(model: str, system: str, lines: list[str],
               progress=None) -> list[str]:
    """Translate a list of lines, preserving its length no matter what.

    Batches that come back malformed are retried once per line, so a single
    stubborn line costs one extra request instead of corrupting the alignment
    of everything after it.
    """
    results: list[str] = []
    done = 0
    for start in range(0, len(lines), BATCH_SIZE):
        batch = lines[start:start + BATCH_SIZE]
        produced = _run_batch(model, system, batch)
        for original, value in zip(batch, produced):
            if value:
                results.append(value)
                continue
            single = _run_batch(model, system, [original])[0]
            # Last resort: keep the source line.  A visible untranslated line
            # is recoverable; a shifted layer is not.
            results.append(single or original)
        done += len(batch)
        if progress:
            progress(done, len(lines))
    return results


def _translation_cache_path(cache_dir: Path, *, model: str, source: str,
                            target: str, text: str) -> Path:
    key = hashlib.sha256(
        f"v1\0{model}\0{source}\0{target}\0{text}".encode("utf-8")
    ).hexdigest()
    return cache_dir / f"{key}.json"


def _run_cached_lines(model: str, system: str, lines: list[str], *,
                      source: str, target: str, cache_dir: Path | None,
                      progress=None) -> list[str]:
    """Translate each distinct lyric once and reuse persistent chorus hits."""
    if not cache_dir:
        return _run_lines(model, system, lines, progress)
    cache_dir.mkdir(parents=True, exist_ok=True)
    unique = list(dict.fromkeys(lines))
    values: dict[str, str] = {}
    missing: list[str] = []
    for line in unique:
        path = _translation_cache_path(
            cache_dir, model=model, source=source, target=target, text=line)
        try:
            value = json.loads(path.read_text(encoding="utf-8")).get("text")
        except Exception:  # noqa: BLE001
            value = None
        if isinstance(value, str) and value:
            values[line] = value
        else:
            missing.append(line)

    def missing_progress(done: int, total: int) -> None:
        if progress:
            progress(len(unique) - len(missing) + done, len(unique))

    produced = (_run_lines(model, system, missing, missing_progress)
                if missing else [])
    for original, translated in zip(missing, produced):
        values[original] = translated
        path = _translation_cache_path(
            cache_dir, model=model, source=source, target=target, text=original)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"text": translated}, ensure_ascii=False),
                             encoding="utf-8")
        temporary.replace(path)
    if progress and not missing:
        progress(len(unique), len(unique))
    return [values.get(line, line) for line in lines]


# --------------------------------------------------------------------------- #
# Rutas
# --------------------------------------------------------------------------- #
def normalize_code(language: str | None) -> str | None:
    if not language:
        return None
    code = language.strip().lower().replace("_", "-")
    if not code or code in _UNKNOWN_CODES:
        return None
    # Whisper reports both "es" and "spanish" depending on the runtime.
    if len(code) > 3:
        for iso, name in LANGUAGE_NAMES.items():
            if name.lower() == code:
                return iso
        return code[:2]
    return code.split("-")[0]


def language_label(code: str | None) -> str:
    if code == AUTO_SOURCE:
        return "Idioma detectado"
    code = normalize_code(code)
    if not code:
        return "desconocido"
    return LANGUAGE_LABELS_ES.get(code, LANGUAGE_NAMES.get(code, code))


def plan_route(source: str | None, target: str | None) -> list[tuple[str, str]]:
    """The hops needed to get from `source` to `target`.

    An unknown source still gets translated — returning "nothing to do" would
    hand the caller its own text back and let it believe that was the
    translation.  It gets a single hop: pivoting is a decision about the
    script, and guessing the script wrong is worse than not pivoting.
    """
    source = normalize_code(source)
    target = normalize_code(target)
    if not target:
        return []
    if not source:
        return [(AUTO_SOURCE, target)]
    if source == target:
        return []
    if source in NON_LATIN_SOURCES and target != "en":
        return [(source, "en"), ("en", target)]
    return [(source, target)]


def route_label(route: list[tuple[str, str]]) -> str:
    if not route:
        return "sin traducción"
    names = [language_label(route[0][0])] + [language_label(hop[1]) for hop in route]
    return " → ".join(names)


# --------------------------------------------------------------------------- #
# API pública
# --------------------------------------------------------------------------- #
def romanize(lines: list[str], *, model: str | None = None,
             progress=None) -> tuple[list[str], str]:
    """Japanese → Hepburn romaji, in context.

    Prefers the local LLM because dictionary transliteration reads 君の名は as
    "kun no mei ha" instead of "kimi no na wa": choosing between a word's
    readings needs the sentence, which a lookup table does not have.
    pykakasi is the offline safety net.
    """
    if not lines:
        return [], "none"
    model = model or pick_model()
    if model:
        try:
            produced = _run_lines(model, _ROMAJI_SYSTEM, lines, progress)
            if sum(1 for a, b in zip(lines, produced) if a != b) >= len(lines) // 2:
                return produced, f"ollama:{model}"
        except Exception:  # noqa: BLE001
            pass
    fallback = romanize_offline(lines)
    if fallback is not None:
        return fallback, "pykakasi"
    return list(lines), "none"


def romanize_offline(lines: list[str]) -> list[str] | None:
    try:
        import pykakasi
    except Exception:  # noqa: BLE001
        return None
    kks = pykakasi.kakasi()
    out: list[str] = []
    for line in lines:
        pieces = [str(item.get("hepburn", "")) for item in kks.convert(line)]
        text = " ".join(piece for piece in pieces if piece)
        out.append(re.sub(r"\s+([,.;:!?…])", r"\1", text).strip() or line)
    return out


def translate(lines: list[str], *, source: str | None, target: str,
              model: str | None = None, progress=None,
              cache_dir: Path | None = None) -> dict:
    """Translate lyric lines, pivoting through English when the script demands.

    Returns the translated lines plus the metadata the UI and the Lyrixa
    envelope need to explain what happened.
    """
    route = plan_route(source, target)
    if not lines or not route:
        return {"lines": list(lines), "route": route, "routeLabel": route_label(route),
                "backend": "none", "model": None}

    model = model or pick_model()
    if not model:
        raise TranslationUnavailable(
            "No hay traductor local disponible. Instala Ollama y baja un "
            "modelo multilingüe (por ejemplo: ollama pull qwen3.5)."
        )

    current = list(lines)
    hops = len(route)
    for index, (hop_source, hop_target) in enumerate(route):
        system = _system_prompt(_language_name(hop_source),
                                _language_name(hop_target))

        def hop_progress(done: int, total: int, _i=index) -> None:
            if progress:
                progress((_i + done / max(total, 1)) / hops)

        current = _run_cached_lines(
            model, system, current, source=hop_source, target=hop_target,
            cache_dir=cache_dir, progress=hop_progress)

    return {
        "lines": current,
        "route": route,
        "routeLabel": route_label(route),
        "backend": "ollama",
        "model": model,
    }


# Dónde queda la traducción de cada línea para no volver a pedirla.  Lo
# declara este módulo y no cada llamante: el pipeline y el endpoint tienen que
# compartir caché, y que lo decidiera cada uno por su cuenta es justo cómo se
# acaba traduciendo dos veces lo mismo.
DEFAULT_CACHE_DIR = paths.TRANSLATION_CACHE_DIR


# --------------------------------------------------------------------------- #
# Mezclas multilingües
#
# Un mix de una hora no tiene *un* idioma: tiene tramos en japonés, en inglés
# y en español, y cada tramo necesita su propia ruta y su propio romaji.  Esto
# vive aquí y no en `server` porque es política de traducción — qué agrupar,
# qué romanizar, qué ruta tomar — y el pipeline no es su único cliente.
# --------------------------------------------------------------------------- #
def group_by_language(languages: list[str | None],
                      fallback: str | None) -> dict[str | None, list[int]]:
    """Line indices grouped by the language they were sung in."""
    groups: dict[str | None, list[int]] = {}
    for index, code in enumerate(languages):
        key = normalize_code(code) or normalize_code(fallback)
        groups.setdefault(key, []).append(index)
    return groups


def translate_clip_lines(lines: list[str], languages: list[str | None], *,
                         source: str | None, target: str,
                         romaji: bool, emit) -> dict:
    """Romaniza y traduce las líneas ya confirmadas por el consenso.

    Se hace al final a propósito: traducir un texto que todavía puede cambiar
    es trabajo tirado, y una traducción hecha sobre texto inventado se ve
    perfectamente correcta, que es la peor forma de estar mal.

    Las líneas se agrupan por el idioma en el que se cantaron, no por el del
    archivo: en un mix de una hora conviven canciones en japonés y en inglés, y
    cada grupo necesita su propia ruta (ja→en→es, en→es) y su propio romaji.
    """
    out: dict = {"lines": None, "romaji": None, "backend": None,
                 "routeLabel": None, "targetLanguage": None, "error": None,
                 "routes": []}
    if not lines:
        return out

    groups = group_by_language(languages, source)
    errors: list[str] = []
    routes: list[str] = []

    # --- Romaji: sólo los tramos que realmente se cantaron en japonés ---
    # Ojo con el nombre: el parámetro NO puede llamarse `romanize`, porque
    # taparía a la función `romanize()` de este mismo módulo y la llamada de
    # abajo reventaría con "'bool' object is not callable" — silenciada por el
    # `except` y dejando el romaji vacío sin que nadie se entere.
    romaji_lines: list[str] | None = None
    if romaji:
        for code, indices in groups.items():
            if code not in ROMANIZABLE:
                continue
            emit("status", stage="romanizing",
                 message=(f"Pasando a romaji {len(indices)} líneas en "
                          f"{language_label(code)}…"))
            try:
                converted, backend = romanize(
                    [lines[i] for i in indices],
                    progress=lambda done, total: emit(
                        "progress", stage="romanizing",
                        progress=done / max(total, 1)),
                )
            except Exception as error:  # noqa: BLE001
                errors.append(f"No se pudo romanizar: {error}")
                continue
            if romaji_lines is None:
                romaji_lines = list(lines)
            for position, index in enumerate(indices):
                romaji_lines[index] = converted[position]
            out["romajiBackend"] = backend
        out["romaji"] = romaji_lines

    if not target:
        out["error"] = "; ".join(errors) or None
        return out

    # --- Traducción: una ruta por idioma de origen ---
    translated = list(lines)
    did_translate = False
    for code, indices in groups.items():
        route = plan_route(code, target)
        if not route:
            continue                    # ya está en el idioma de destino
        label = route_label(route)
        emit("status", stage="translating",
             message=(f"Traduciendo {len(indices)} líneas ya confirmadas · "
                      f"{label}…"))
        try:
            result = translate(
                [lines[i] for i in indices], source=code, target=target,
                cache_dir=DEFAULT_CACHE_DIR,
                progress=lambda p: emit("progress", stage="translating",
                                        progress=p),
            )
        except TranslationUnavailable as error:
            out["error"] = str(error)
            return out
        except Exception as error:  # noqa: BLE001
            errors.append(f"{label}: {error}")
            continue
        for position, index in enumerate(indices):
            translated[index] = result["lines"][position]
        routes.append(f"{label} ({len(indices)})")
        out["backend"] = result["backend"]
        did_translate = True

    out["error"] = "; ".join(errors) or None
    if not did_translate:
        return out
    out["lines"] = translated
    out["routes"] = routes
    out["routeLabel"] = " · ".join(routes)
    out["targetLanguage"] = normalize_code(target)
    return out
