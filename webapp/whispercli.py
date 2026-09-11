"""Whisper por CLI: ejecutarlo y escribir sus salidas
(txt/srt/vtt/tsv/json). El modo ultra no pasa por aquí;
reusa `write_whisper_outputs`.
"""

import os
import subprocess
from pathlib import Path
import paths
from serveutil import _TS_RE, device_name, parse_ts, safe_stem


def run_whisper(audio: Path, model: str, language: str, outdir: Path,
                duration: float, emit, out_stem: str | None = None) -> dict:
    """Corre Whisper por CLI en modo verbose y transmite cada segmento.

    `out_stem` es el nombre con el que queremos que salgan los archivos. Whisper
    los nombra según el audio que le damos, que puede ser el id de YouTube o el
    `vocals.wav` de Demucs; aquí los renombramos al título de verdad.
    """
    cmd = ["whisper", str(audio), "--model", model,
           "--output_format", "all", "--output_dir", str(outdir),
           "--verbose", "True", "--model_dir", str(paths.CACHE_DIR),
           # Tiempos por palabra: bordes de segmento más precisos (al ms) y
           # habilita words[] en el JSON para el puente con Lyrixa.
           "--word_timestamps", "True",
           # Reduce bucles de texto sobre silencios/residuos de Demucs. No
           # elimina coros repetidos por postproceso: la decisión sigue siendo
           # de Whisper y los resultados raw permanecen auditables.
           "--condition_on_previous_text", "False",
           "--hallucination_silence_threshold", "2.0"]
    if language:
        cmd += ["--language", language]

    emit("status", stage="transcribing",
         message=f"Transcribiendo con '{model}' en {device_name()}…",
         model=model, device=device_name(), duration=duration)

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )
    for line in proc.stdout:
        line = line.rstrip("\n")
        m = _TS_RE.search(line)
        if m:
            start = parse_ts(m.group(1))
            end = parse_ts(m.group(2))
            text = m.group(3).strip()
            progress = min(end / duration, 1.0) if duration else None
            emit("segment", start=start, end=end, text=text, progress=progress)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("Whisper terminó con error.")

    stem = audio.stem
    target = safe_stem(out_stem or stem, fallback=stem)
    files = {}
    for ext in ("txt", "srt", "vtt", "tsv", "json"):
        f = outdir / f"{stem}.{ext}"
        if not f.exists():
            continue
        if target != stem:
            renamed = outdir / f"{target}.{ext}"
            f.replace(renamed)      # replace: sobrescribe si ya existía
            f = renamed
        files[ext] = f.name
    return files
def write_whisper_outputs(result: dict, outdir: Path, stem: str) -> dict:
    """Escribe txt/srt/vtt/tsv/json usando los writers del propio Whisper.

    El modo ultra no pasa por la CLI de Whisper, pero sus segmentos tienen
    exactamente la misma forma; reutilizar los writers garantiza que un SRT del
    modo ultra y uno del modo normal sean indistinguibles para cualquier
    reproductor.
    """
    from whisper.utils import get_writer

    files: dict[str, str] = {}
    options = {"max_line_width": None, "max_line_count": None,
               "highlight_words": False, "max_words_per_line": None}
    for extension in ("txt", "srt", "vtt", "tsv", "json"):
        try:
            writer = get_writer(extension, str(outdir))
            writer(result, f"{stem}.wav", options)
        except Exception:  # noqa: BLE001
            continue
        target = outdir / f"{stem}.{extension}"
        if target.is_file():
            files[extension] = target.name
    return files
