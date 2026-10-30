# Auditoría de precisión japonesa v1

Estado inicial: 28 de octubre de 2026. Esta auditoría corresponde al encargo
`SCRIBIX_PROMPT_PRECISION_JA_V1.md` y se mantiene separada de los resultados de
audio personales.

## Línea base

- Rama: `feat/scribix-brand`.
- Suite antes de modificar el pipeline japonés: **245 pruebas, 245 correctas**.
- Audio real del tramo 45:03–45:24: disponible en trabajos locales, pero los
  tiempos del encargo son visuales; la validación auditiva sigue pendiente.
- Fixture de regresión: `tests/fixtures/japanese_precision_cases.json`. Contiene
  extractos sintéticos mínimos, no una letra completa protegida.

## Mapa del pipeline real

| Etapa | Entrada → salida | Código responsable |
|---|---|---|
| Separación y acondicionamiento | mezcla → `vocals.wav` → `asr-input.wav` | `media.separate_vocals`, `audio_analysis.condition_for_asr`, `pipelines.pipeline` |
| Detección y consenso | audio vocal → segmentos con `words[]`, idioma, acuerdo y tiempos | `precision.run`, `consensus.py`, `transcribe_engines.py` |
| Rescate | huecos vocales sin palabras → segmentos desde mezcla original | `precision.rescue_from_original`, `audio_analysis.erased_voice_intervals` |
| Referencia | tracklist/LRCLIB + segmentos → ortografía corroborada | `track_guides.build_guide`, `prompt_spans`, `refine_segments` |
| Segmentación visual | segmentos/tiempos por palabra → clips legibles | `lyrixa_bridge.normalize_whisper_segments`, `split_words` |
| Romaji y traducción | líneas finales → listas paralelas | `translation.translate_clip_lines`, `romanize`, `translate` |
| Exportación | clips + listas derivadas → cuatro capas con `sourceId` común | `lyrixa_bridge.build_lyrixa_envelope`, `pipelines.build_lyrixa_bundle` |

El JSON crudo de Whisper se conserva. Romaji y traducción se calculan después
del consenso y antes de construir el proyecto Lyrixa.

## Causalidad por síntoma

### Confirmado

1. **Referencia CJK excluida.** `track_guides._fold()` conserva únicamente
   `[a-z0-9]`. `prompt_spans()` descarta líneas japonesas y `_line_score()`
   devuelve cero. La protección contra `"" == ""` evita una sustitución
   catastrófica, pero no permite una coincidencia japonesa correcta.
2. **Tiempos inconsistentes al corregir CJK.** `_resync_words()` usa
   `reference.split()`. Una línea japonesa sin espacios se convierte en un solo
   token, aunque Whisper hubiera aportado varios intervalos reales.
3. **Lotes que cruzan canciones.** `translate_clip_lines()` agrupa índices por
   idioma en todo el mix. Dos canciones japonesas alejadas comparten el mismo
   lote y contexto implícito.
4. **Caché sin contexto.** La clave `v1` depende de modelo, ruta y texto, pero no
   de la canción ni de las líneas vecinas que condicionan una lectura.
5. **Límite CJK genérico.** `lyrixa_bridge._boundary_bonus()` aplica el mismo
   coste a cualquier par CJK; no distingue un cambio de frase de `少 | し`.

### Hipótesis que requieren audio/modelo

- Los cambios `ちゃ→じゃ`, `見え→慣れ` y las sílabas huérfanas pueden venir del
  ASR, de una referencia de otra versión o del romanizador. Una coincidencia de
  texto no demuestra cuál de las tres fuentes es correcta.
- La ruta directa japonés→español puede superar al pivote japonés→inglés→español
  en el modelo instalado, pero no se cambiará globalmente sin evaluación
  etiquetada.

### Ya resuelto en HEAD

- El rescate de regiones con voz borrada y el rechazo de muletillas de Whisper.
- La detección se ejecuta sobre la voz acondicionada, no sobre la mezcla.
- Cuatro capas fijas, `role: transliteration`, `sourceId` compartido y
  `bridgeSchemaVersion: 4`.
- La letra original nunca se sustituye por romaji.

## Pruebas a construir antes de cada cambio

1. Clave Unicode no vacía para kana, kanji y hangul; símbolos y vacío nunca
   coinciden.
2. Coincidencia positiva de variantes cercanas y negativa de versos distintos.
3. Reparto de referencia CJK sobre varios tiempos sin perder caracteres.
4. Prohibición de corte `少 | し` y preferencia por un cambio CJK/latín.
5. Traducción/romaji agrupados por canción + idioma + continuidad, conservando
   exactamente cantidad, orden e identidad externa.
6. Caché versionada por contexto relevante; coros con contexto equivalente se
   reutilizan y contextos distintos no colisionan.

## Límites de esta ejecución

- No se afirmará precisión acústica sin escuchar y etiquetar el audio.
- No se incorporarán letras completas ni audio protegido al repositorio.
- No se inventarán timestamps: toda corrección conserva o redistribuye dentro
  del intervalo medido existente y queda marcada con procedencia.
- El contrato Lyrixa no cambia; cualquier diagnóstico nuevo irá en los reportes
  internos mientras no exista coordinación con el consumidor.

## Resultado implementado

- La letra dispone de normalización Unicode propia; `_fold()` permanece
  limitado a metadatos para no alterar la selección histórica de canciones.
- `prompt_spans()` acepta y deduplica CJK. `refine_segments()` exige el mismo
  umbral conservador, asigna un `trackId` estable y reparte el texto corregido
  sobre varios intervalos reales en lugar de fabricar un token gigante.
- Los clips normalizados conservan `trackId`. Traducción y romaji se procesan
  por grupos contiguos de track + idioma sin cruzar canciones; la huella de
  caché usa solo una línea vecina a cada lado.
- La caché de traducción sube a v2 e incluye modelo, prompt, ruta, texto y
  huella contextual. Una edición del original invalida solo las entradas cuyo
  texto o vecindad cambió.
- La salida de romaji se valida línea por línea. Japonés residual, vacío o sin
  caracteres latinos activa el respaldo local `pykakasi` únicamente para la
  línea inválida.
- La segmentación mantiene `bridgeSchemaVersion: 4`, las cuatro capas fijas y
  `sourceId` compartido. Las invariantes de tiempos, identidad y capas siguen
  cubiertas por las pruebas contractuales existentes.

No se añadió `needsReview` al `.lyrixa.json`: sería un campo de producto que
debe coordinarse con Lyrixa. La procedencia (`referenceProvider`), puntuación
de referencia y acuerdo continúan disponibles en el resultado de Scribix.

## Validación pendiente

La fixture prueba regresiones de cadenas y tiempos, no calidad acústica. Falta
escuchar el tramo real, etiquetar una referencia autorizada y medir CER/WER,
lectura romaji, traducción y tiempo de inferencia. Tampoco se cambia todavía la
ruta global con pivote por inglés: se requiere comparar ambos recorridos con el
modelo local que realmente se vaya a distribuir.

La unidad de traducción sigue siendo el clip visual final. Separar una frase
semántica larga de sus clips de presentación exigiría una alineación explícita
frase→clips que el contrato actual no contiene; hacerlo por heurística podría
desplazar traducciones. Por eso esta ejecución aísla el contexto por track y
preserva identidades, pero deja esa resegmentación como trabajo coordinado con
Lyrixa. El validador de romaji detecta cobertura estructural, no puede probar
por sí solo una lectura polifónica ni una sílaba plausible pero equivocada.
