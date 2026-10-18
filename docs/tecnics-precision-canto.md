# Precisión en canto: qué funciona y qué no

Investigación del 16 oct 2026. Rescatada del subagente de omp, que sintetizó los
hallazgos pero murió al intentar escribir el archivo (se cayó la inferencia de
skynet a mitad de la sesión). Las fuentes son papers de ISMIR/arXiv; las medidas
propias vienen de experimentos sobre `Música Anime Sin Copyright 2022 - Parte 1`.

## El problema

Whisper sobre el stem de voz detecta **inglés** en canto japonés, y la votación
por consenso descarta ~40 % de las palabras reales. Resultado: 120 huecos que
suman 41 minutos. El hueco de 14:16 tiene energía vocal medible (RMS 0.041) y aun
así no devuelve ni una palabra.

## Causa raíz: sangrado del stem

No es el modelo ni la velocidad. Es que el stem de voz arrastra instrumental, y
esa mezcla empuja la detección de idioma hacia el inglés.

## Lo que se probó y NO funciona

| Idea | Resultado |
| --- | --- |
| Ralentizar el audio (0.75×) | Peor: alucina texto de más |
| Ralentizar más (0.5×) | Se mete en bucles |
| Cambiar large-v3 por turbo | Igual: los dos fallan con el sangrado |

El paper de LyricWhiz añade un dato incómodo: **separar la voz degrada el
resultado de Whisper**, salvo con large-v2 en inglés. O sea, la separación no es
gratis.

## Lo que sí recomienda la literatura

1. **Mejor separación de stems.** BS-RoFormer es lo mejor disponible según el
   paper, y ya lo usamos — pero la corrida de referencia fue con aislamiento
   `precise`, no `ultra`. Eso es lo primero que hay que repetir.
2. **Detectar el idioma solo sobre el stem de voz**, nunca sobre la mezcla
   completa. Es la medida crítica contra el falso positivo de inglés.
3. **Prompt inicial para Whisper**: usar `"lyrics:"` como prefijo.
4. **Postproceso con un LLM** para corregir y reducir la varianza entre pasadas.
   Es lo que hace LyricWhiz (Whisper + LLM), que es el estado del arte en pop y
   rock en inglés: varias pasadas más un modelo de lenguaje encima.
5. **Alineación contra la letra de referencia** (lyrics-to-audio alignment).

## Alineación: el camino práctico

De las dos ideas que se plantearon, esta es la buena. La técnica clásica es
**DTW** (dynamic time warping) para encajar una letra conocida contra el audio
real. Referencias útiles:

- **ACE-Step** usa un Diffusion Transformer con atención cruzada y DTW.
- **lyric-align**, proyecto japonés, hace exactamente alineación CJK.
- LRCLIB tiene letras japonesas sincronizadas para anime.

**Ya está medio implementado**: `track_guides.py` hace esto
(`build_guide`, `fetch_reference`, `_align_keys`, `_line_score`, `_resync_words`,
`language_priors`, `prompt_spans`, `_apply_priors`). El problema es que la corrida
de referencia se hizo con `hasTracklist: false` y `fetchLyrics: false`, así que
nunca se activó.

## Siguientes pasos concretos

1. Repetir la corrida con aislamiento `ultra` en vez de `precise`.
2. Activar `hasTracklist` y `fetchLyrics` para que `track_guides.py` entre en juego.
3. Mover la detección de idioma al stem de voz exclusivamente.
4. Añadir `"lyrics:"` como prompt inicial de Whisper.
5. Solo después de eso, evaluar postproceso con LLM.

## Fuentes

- LyricWhiz: Robust Multilingual Zero-shot Lyrics Transcription by Whispering to
  ChatGPT — arXiv 2306.17103 (ISMIR 2023)
- Exploiting Music Source Separation for Automatic Lyrics Transcription with
  Whisper — arXiv 2506.15514 (ICMEW 2025)
- JAM-ALT: A Formatting-Aware Lyrics Transcription Benchmark — arXiv 2311.13987
- Multilingual Lyrics-to-Audio Alignment — ISMIR 2020
- User-Centered Evaluation of Lyrics-to-Audio Alignment — ISMIR 2021
- Lyrics Transcription in Western Classical Music with Whisper (Winterreise) —
  NLP4MusA 2024
