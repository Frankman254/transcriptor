# Relevo: precisión en canto

Para la siguiente sesión. Escrito el 16 oct 2026 al cerrar la anterior, que corría
sobre inferencia local y se quedó sin modelo a mitad.

## El encargo

"Fix Japanese misdetection and use heaviest model": el transcriptor detecta
**inglés** en canto japonés y pierde palabras. Hay que arreglarlo.

## Qué pasó en la sesión anterior

Un subagente investigó la literatura (ISMIR, arXiv) durante ~46 min. Llegó a
conclusiones buenas y **murió justo al escribir el documento**: primero se cayó
la red hacia el Spark de Rolando (`ENOTFOUND skynet.tailbdb430.ts.net`), omp bajó
a un Qwen3.5 9B local, y ese modelo con 165K de contexto encima no consiguió
emitir la llamada a `write` — dos `error; retried` y sesión atascada a 3.6 tok/s.

Las conclusiones se rescataron a mano y están en
**[tecnics-precision-canto.md](tecnics-precision-canto.md)**. Léelo primero: es
el resultado de la investigación y no hace falta repetirla.

## Estado medido de la corrida de referencia

Job `1b0734de70f7` — `webapp/output/1b0734de70f7/`, 62 min de audio,
`Música Anime Sin Copyright 2022 - Parte 1`. Terminó bien (`status: completed`),
84 min de proceso.

Lo que dice su propio `job.json`:

```
settings.separationQuality = "precise"      ← NO es ultra
settings.hasTracklist      = false          ← guías apagadas
settings.fetchLyrics       = false          ← guías apagadas
settings.model             = "large-v3"
settings.precisionLevel    = "balanced"

stem.bleedRatio  = 0.7912
stem.verdict     = "bleed"
stem.notes       = "Queda bastante instrumental filtrado en la voz.
                    El perfil Ultra hace una segunda pasada que suele limpiarlo."

precision.language            = "en"   (probabilidad 0.618)
precision.languages.seconds   = en 1736.5 | ja 1061.1 | uk 8.6 | hi 3.8
precision.rounds[0].words     = 3039
precision.rounds[0].dropped   = 1627     ← retención 0.7922
precision.rounds[0].looped    = 308
precision.converged           = false
precision.reference           = null     ← nunca se usó letra de referencia
```

**El propio sistema ya diagnosticó el problema y sugirió la solución.** El stem
tiene 79 % de sangrado, el veredicto literal es `bleed`, y la nota dice que Ultra
hace una segunda pasada que lo limpia — pero la corrida se hizo en `precise`.

Y aunque hay 1061 segundos de japonés detectado, el dominante salió `en` con solo
0.618 de probabilidad. Clasifica todo como inglés y arrastra el error.

## Lo que hay que hacer (en este orden)

1. **Repetir la corrida con `separationQuality: "ultra"`.** Es el cambio de una
   línea con más probabilidad de arreglar el problema de raíz, y lo recomienda el
   propio diagnóstico del job. Mide `bleedRatio` después: si no baja de 0.79, el
   resto de pasos cambian de prioridad.
2. **Mover la detección de idioma al stem de voz exclusivamente**, nunca sobre la
   mezcla completa. Es la medida crítica contra el falso "inglés" según la
   literatura. Verificar dónde se hace hoy.
3. **Activar `hasTracklist: true` y `fetchLyrics: true`.** `webapp/track_guides.py`
   ya implementa alineación contra letra de referencia (`build_guide`,
   `fetch_reference`, `_align_keys`, `_line_score`, `_resync_words`,
   `language_priors`, `prompt_spans`, `_apply_priors`) y nunca se activó —
   `precision.reference` salió `null`. LRCLIB tiene letras japonesas
   sincronizadas para anime.
4. **Añadir `"lyrics:"` como prompt inicial de Whisper.** Lo recomiendan los
   papers; es barato de probar.
5. Solo después de todo lo anterior, evaluar postproceso con un LLM
   (el enfoque LyricWhiz).

**No vuelvas a probar ralentizar el audio.** Ya se midió: 0.75× alucina texto de
más y 0.5× se mete en bucles. Está descartado experimentalmente.

## Dónde está cada cosa

| Qué | Dónde |
| --- | --- |
| Alineación con letra de referencia | `webapp/track_guides.py` |
| Votación por consenso | `webapp/consensus.py` (`VOTE_WEIGHT`, `CONFIDENCE_WEIGHT`, `NULL_CONFIDENCE = 0.55`, `REPEAT_KEEP = 4`) |
| Salidas de la corrida | `webapp/output/1b0734de70f7/` |
| Investigación ya hecha | `docs/tecnics-precision-canto.md` |
| Contrato con Lyrixa | `docs/contrato-lyrixa.md` |

## Avisos

- **Escribe los hallazgos en disco según los vayas teniendo, no al final.** Así
  murió la sesión anterior: tenía todo pensado y perdió el informe entero porque
  solo intentó guardarlo al terminar.
- El `.mp3` y el `.wav` de la corrida pesan ~117 MB cada uno. No los metas en git.
- Si usas inferencia local: el 9B no aguanta este trabajo con contexto grande.
  Usa el 27B.

---

## Sesión del 16 oct (continuación) — hallazgos según aparecen

**H1. El paso 2 ya está hecho.** La detección de idioma corre sobre el stem de
voz, no sobre la mezcla: `pipelines.py` reasigna `audio = vocals` tras separar,
lo pasa por `condition_for_asr` (→ `asr-input.wav`) y eso es lo que recibe
`precision.run`, que es quien llama a `_detect_language`. No hay nada que mover.

**H2. `languageProbability = 0.618` no es una probabilidad de Whisper.** Es
`LanguageMap.share`: la fracción de segundos cantados cuyo tramo ganó `en`. Las
detecciones por ventana con probabilidad < `MIN_LANGUAGE_PROBABILITY` ya se
descartan antes. Y la corrida **sí** se trató como multilingüe
(`multilingual: true`): los 1061 s de `ja` se transcribieron en japonés. El
problema real son los tramos `en` — falta comprobar cuántos son japonés mal
clasificado y cuántos son inglés de verdad (es una recopilación "sin copyright":
puede llevar canciones en inglés).

**H3. Los tramos `en` son mayoritariamente alucinación, no inglés.** Muestras:
"He's up now.", "That's all.", "I love you." ×4 seguidos, "Thank you.",
"transcript Emily Beynon" (el clásico de Whisper sobre no-voz), y japonés dentro
de tramos `en` ("ありがとうございました We are the Dingoes!"). Densidad: 0.65
palabras/s en `en` frente a 1.66 en `ja`. Hay inglés real suelto (las letras de
anime llevan frases en inglés), pero no 29 minutos.

**H4. CAUSA RAÍZ: el motor MLX no detecta idioma, finge que sí.**
`_MlxWhisperEngine.detect_language` (transcribe_engines.py) hace una
transcripción completa con `language=None` y devuelve `probability = 1.0`
siempre. Consecuencias:
- `MIN_LANGUAGE_PROBABILITY = 0.60` no descarta nada con MLX: una ventana de
  sangrado instrumental vota `en` con "certeza" total. Con faster-whisper y
  openai-whisper ese filtro sí funciona (devuelven la probabilidad real).
- Es además carísimo: decodifica la ventana entera para leer un solo token.
- La corrida de referencia usó `engine: mlx-whisper`. Es justo el caso roto.
Esto encaja mejor con los datos que el sangrado por sí solo: el sangrado hace
que Whisper dude, pero es el `1.0` falso lo que convierte la duda en un voto.

**H5. Medido sobre las 208 ventanas de la corrida, con la detección real de MLX**
(`model.detect_language(mel)` de mlx-whisper 0.4.3; 70 s en total, frente a
decodificar cada ventana entera):

| Escenario | en | ja | sin voto seguro |
|---|---:|---:|---:|
| Corrida real (prob. falsa 1.0) | 1736 s | 1061 s | — |
| Prob. real, umbral 0.0 | 1737 s | 1061 s | — ← reproduce la corrida: simulación fiel |
| **Prob. real, umbral 0.60 (el que ya existe)** | **830 s** | **1046 s** | 934 s → van al dominante (`ja`) |
| Prob. real, umbral 0.80 | 509 s | 1025 s | 1277 s |

- 84 de las 123 ventanas que ganó `en` lo hicieron con probabilidad < 0.6.
- Cuando gana `en`, el segundo es `ja` en 45 ventanas, `ru` en 32, `es` en 13:
  ruido de un clasificador que duda, no inglés.
- El argmax es el mismo con los dos métodos; lo único que faltaba era la
  probabilidad para poder dudar.
Script: scratchpad de la sesión (`langid_experiment.py`), resultados en
`langid.jsonl`. Siguiente: transcribir unas cuantas regiones que cambian de
`en` a `ja` y ver si sale japonés con sentido o más basura.

**H6. Corrige H5: el idioma no es el problema principal; la separación lo es.**
Transcribí las regiones que pasarían de `en` a `ja`, en los dos idiomas, sobre
el stem: salen las alucinaciones canónicas de cada idioma ("Thank you.",
"I'm sorry." / "ご視聴ありがとうございました", "お疲れ様でした"). Cambiar la
etiqueta no recupera palabras. Y las regiones `en` con prob. ≥ 0.99 son inglés
real ("Ever on and on I continue / Circling with nothing but my hate / In a
carousel of agony"): la recopilación lleva canciones en inglés.

El hueco de 14:41–18:59 (4 min, "el de 14:16") decidió la cuestión:
- Sobre el **stem**: nada en ningún idioma, aunque el RMS es 0.075.
- Sobre la **mezcla original**: Whisper detecta `ja` y transcribe canto
  ("あなたが何人なんて何度わかり…", "わかりません? なんて? まだまだまだ").
- Ratio stem/instrumental en el hueco (0.080/0.183) parecido al de los tramos
  que sí funcionan: no es que ahí haya más sangrado; es que la separación se
  comió esa voz (canto rápido, estilo denpa).
Es el hallazgo de LyricWhiz: separar puede degradar a Whisper. Queda por saber
por qué `rescueOriginal: true` no rescató estos 4 minutos.

**H7. Por qué `rescueOriginal` no rescató el hueco: por construcción no puede.**
`_rescue_intervals` (precision.py) solo propone segmentos de baja confianza y
huecos de **1 a 8 s** entre segmentos (pensado para un count-in borrado). Un
hueco de 4 min nunca es candidato. Y `rescue_from_original` fuerza el idioma
del segmento vecino o, si no hay, `result.language` (= `en`), así que aunque lo
mirase transcribiría canto japonés forzando inglés y las puertas acústicas lo
tirarían con razón. La mezcla, sin forzar idioma, detecta `ja` sola.
La firma del fallo es medible: **región que el VAD marca como voz y sin
palabras en la transcripción final.**

**H8. Arreglo hecho: el rescate mira los tramos borrados.** (precision.py,
consensus.py; 231 pruebas en verde)
- `PrecisionResult.speech_regions` guarda dónde oyó voz el VAD del stem.
- `_erased_intervals`: regiones de voz ≥ 3 s con < 0.4 palabras/s en la letra
  final, en trozos de ≤ 29 s, hasta 160 repartidos por la mezcla.
- En esos tramos `rescue_from_original` **no fuerza idioma** (decide la
  mezcla y el segmento hereda el idioma que detectó) y el tope de palabras es
  6/s de tramo en lugar de 12 fijas. Los remiendos de siempre no cambian.
- El rescate descarta una hipótesis que sea entera un reflejo de Whisper.
- `is_filler_line` ignora espacios ("ご 視 聴 ありがとう…" tal como llega) y
  reconoce créditos (作詞/作曲/編曲…); la lista suma Outro, Outro Music,
  Music, BGM, お疲れ様でした, 音楽 y la despedida coreana, todos vistos aquí.

Validado ejecutando el `rescue_from_original` real sobre el resultado guardado
de 1b0734de70f7 y su mezcla: **2894 → 3066 palabras, 191 rescatadas en 30
líneas, ninguna alucinación**; 80 zonas revisadas en 269 s. Ejemplos:
20:02–20:51 "雨の間外を駆け出した / わかって聞いてよ何も言えなくても",
37:33 "落ちた鳥は風を待ちわびる", 45:42 "I'm gonna find the truth, only one",
53:20 "太陽系を抜け出して平行線で交じる". La primera versión dejaba pasar
"ご視/聴/ありがとうございました" y "作/詞/作曲編曲初音ミク": eso es lo que
arreglan los dos últimos puntos.

## Decisiones abiertas (para el usuario)

**D1. El umbral de puntuación del rescate (0.58) tira letra real.** *Decidido (18 oct): bajado a 0.40 (`RESCUE_MIN_SCORE`).* Cantar
sobre música baja la probabilidad por palabra. Con los datos de esta corrida,
bajarlo añadiría: 06:40 "I have to let you go without a fight" (0.42), 41:11
"世界で一番の君でさ何が本当に怖いの?" (0.50), 49:27 "この夏の一日が始まる"
(0.55), 50:19 (0.53), 52:22 (0.44) — y también 18:31, el denpa en bucle
(0.57). Sin letra de referencia no se puede medir la precisión que se pierde.
No lo he tocado.

**D2. La detección de idioma de MLX (H4) sigue sin arreglar, a propósito.**
Devolver la probabilidad real es fácil (`model.detect_language(mel)`, 70 s
para 208 ventanas frente a decodificarlas enteras), pero con el umbral de
0.60 actual **empeoraría** esta corrida: 45:27 (en 0.39) y 46:15 (en 0.585)
son inglés cantado real y pasarían a `ja` por el dominante. Lo correcto es
probabilidad real + voto suave por región (sumar distribuciones, no argmax
filtrado), validado en varias corridas, no en una. La ganancia inmediata sería
de tiempo, no de letra.

**D3. El hueco 14:41–18:59 no tiene arreglo con lo que hay.** Es un denpa que
también entra en bucle sobre la mezcla (compression ratio 3.7–14): el rescate
lo rechaza bien. Solo lo arreglaría una letra de referencia (paso 3).

## Estado de los pasos del encargo

| Paso | Estado |
| --- | --- |
| 1. Repetir con `separationQuality: "ultra"` | **Sin hacer.** ~90 min de máquina; lo lanza el usuario. Comparar contra las 3066 palabras de H8 y las regiones borradas. |
| 2. Detectar idioma sobre el stem | **Ya estaba hecho** (H1). El problema era otro (H6–H7). |
| 3. `hasTracklist` + `fetchLyrics` | **Bloqueado:** la guía solo se activa con `tracklist_text` no vacío, y esta subida no trae lista. Hace falta que el usuario la pegue (títulos + tiempos). |
| 4. Prompt `"lyrics:"` | **Choca con una decisión existente:** los motores MLX y openai pasan `initial_prompt=None` adrede porque Whisper repite el prompt (comentario en `_MlxWhisperEngine.transcribe`). Probarlo exige medir ecos de "lyrics:" en la salida, no solo activarlo. |
| 5. Postproceso LLM | Pendiente; tiene sentido después de 1 y 3. |

## Sesión del 18 oct

**H9. Por qué la guía de letras "no se impone" aunque encuentre la canción.**
El usuario lo ha visto: pega la lista, LRCLIB devuelve la letra, y el modelo
sigue escribiendo lo que le parece. Hay dos causas, y en japonés las dos
actúan a la vez:
1. **`_fold` solo conserva `[a-z0-9]`.** `refine_segments` decide con
   `_line_score`, que compara `_fold(oído)` contra `_fold(referencia)`. Una
   línea en kana/kanji se pliega a `""`, así que puntúa 0.0 y **nunca** se
   sustituye. Comprobado: `_fold("雨の間外を駆け出した") == ""`, mientras que una
   línea en inglés con una palabra perdida puntúa 0.96 y sí se corrige. En
   anime, la guía solo aporta las pistas de idioma y la etiqueta `track`.
2. **MLX y openai-whisper no reciben la letra al decodificar**, a propósito
   (`initial_prompt=None`: Whisper repite el prompt y una línea repetida
   pasaría por "corroborada"). Solo faster-whisper la usa, como `hotwords`. En
   un Mac el motor `auto` es MLX, así que la referencia no sesga nada.

Arreglo propuesto para 1 (sin hacer, falta el visto bueno): comparar por
caracteres cuando el texto es CJK, porque el japonés no separa palabras. Lo
delicado es `_resync_words`: una línea japonesa de LRCLIB es un único token
por `split()`, y Lyrixa corta clips a partir de `words[]`. Hay que repartir
los caracteres de la referencia sobre los tiempos de las palabras oídas, no
meter la línea entera como una sola palabra.

**H10. Medido con `RESCUE_MIN_SCORE = 0.40`** (mismo procedimiento que H8,
sobre 1b0734de70f7): 15 zonas aceptadas de 80 (antes 8), **2894 → 3405
palabras**, 533 rescatadas en 64 líneas (antes 191 en 30); 283 s. Entran los
versos que D1 daba por perdidos (06:40, 41:11, 49:32, 50:19, 52:22, con
mezcla ja/en coherente: "I'll never stop", "I'm gonna hold you"). El precio
previsto: **18:43–18:56, siete líneas de kana sin sentido** del denpa
("なばたたんがりびたりひらがり…", score 0.57). Quedan marcadas con
`rescuedFromOriginal` y `rescueScore`, así que se pueden filtrar o revisar en
Lyrixa. "I'm a machine" ×3 (06:30) puede ser coro real o eco: sin verificar.
Lo siguiente es la corrida `ultra` del usuario, que se compara contra esto.

### H11 — La corrida del usuario (18 oct, 15:38) y el bug de la referencia

Archivo: `~/Downloads/… Parte 1.lyrixa (2).json`. Salió con **turbo** (no
large-v3) y con referencia LRCLIB (12 canciones encontradas, 37 líneas
casadas), así que no se compara 1:1 con las 3405 palabras de H10.

| | 16 oct (large-v3) | 18 oct (turbo + ref) |
| --- | --- | --- |
| Líneas | 481 | 535 |
| Palabras latinas / caracteres CJK | 1226 / 2189 | 1846 / 2059 |
| Segundos en ja | 1061 | 1330 |
| Huecos descartados / palabras en bucle | 1627 / 308 | 2358 / 693 |
| Ventanas rescatadas | 2 | 11 |

**Bug grave (corregido):** ~150 líneas en japonés y coreano quedaron
sustituidas por la primera línea de la letra de su canción ("夢の続き
追いかけていたはずなのに" ×45 entre 25:36 y 30:25, "教えて 教えてよ" ×23,
"ララララ" ×27, "어느 날 난…" ×7…). En `refine_segments`, `_fold` vacía las
líneas no latinas y la regla `exact_short` comparaba `"" == ""`. Ahora exige
al menos una palabra; prueba
`test_a_japanese_line_is_never_overwritten_by_the_first_lyric_line`. Hay que
repetir la corrida para medir de verdad; el arreglo de H9 (casar CJK por
caracteres) sigue sin aprobar.
