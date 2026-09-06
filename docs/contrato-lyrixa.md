# Contrato Transcriptor → Lyrixa

Qué produce el Transcriptor, qué puede pedirle Lyrixa y qué falta implementar
del otro lado. Este archivo es la referencia; si el código y este documento no
coinciden, gana el código y hay que corregir el documento.

Estado: **puente v3**, motor local con API.

---

## 1. Quién hace qué

```
audio → separación → transcripción → normalización → traducción opcional
      → proyecto Lyrixa → revisión humana en Lyrixa → LiveWallpaper
```

Transcriptor es el motor de IA local. **No** edita, no dibuja y no reproduce.
Lyrixa es la superficie de autoría. LiveWallpaper renderiza.

Lo que **no** debe duplicarse en Lyrixa: configuración de Ollama, prompts,
elección de modelo, caché de traducción, detección de idioma, romaji, rutas
pivote. Todo eso vive en `webapp/translation.py` y se consume por `POST
/api/translate`.

## 2. Identidad de línea

Una línea de letra es una unidad lógica que existe en varias capas a la vez.
Su identidad es `sourceId`, y es **la misma en todas las capas**:

| Capa | `layerId` | `id` del clip | `sourceId` |
|------|-----------|---------------|------------|
| Letra | `layer-main` | `clip-main-000042` | `line-000042` |
| Traducción | `layer-backing` | `clip-es-000042` | `line-000042` |

Reglas:

- `sourceId` se acuña **después** de que la segmentación decida cuáles son las
  líneas. Un segmento de Whisper partido en tres son tres líneas con tres
  identidades; numerarlas por su origen les daría la misma.
- `sourceId` es único dentro de una capa. Lyrixa resuelve clips con
  `clip.layerId === layerId && clip.sourceId === line.sourceId`
  (`src/core/timeline/tapSync.ts`) y deriva ids con `tapSyncClipId`, así que un
  valor repetido resuelve al clip equivocado y genera ids duplicados.
- El `sourceId` de un clip **nunca** es el `id` de otro clip. Son cosas
  distintas: el `id` dice qué caja del timeline es esta, el `sourceId` dice qué
  línea de letra representa.
- `sourceIndex` es la posición de la línea (0, 1, 2…). Redundante con el
  `sourceId` a propósito: Lyrixa ya lo usaba.
- `segmentIndex` guarda de qué segmento de Whisper salió la línea. Es
  procedencia, no identidad: varias líneas comparten segmento legítimamente, y
  una línea cosida entre dos segmentos no lleva el campo.

Antes de v3 el campo estaba sobrecargado: la capa principal llevaba
`segment-N` (repetido en el 21% de los clips de una mezcla real de una hora) y
la de traducción llevaba el `id` del clip principal. Las dos capas no
compartían nada.

## 3. Capas

`role` y `language` son campos aditivos sobre `LyricLayer`. Lyrixa ya los
declara y los valida en `normalizeLayers`; un `role` desconocido se descarta en
vez de viajar al bundle.

Vocabulario (el de Lyrixa, `src/core/types/layer.ts`):

```
primary · translation · transliteration · backing · fx · annotation
```

**Es `transliteration`, no `romanization`.** El Transcriptor refleja esa lista
en `lyrixa_bridge.LAYER_ROLES` y no inventa un séptimo valor.

Lo que emite hoy el puente:

| `id` | `layerType` | `role` | `language` | Ancla |
|------|-------------|--------|------------|-------|
| `layer-main` | `lyrics` | `primary` | idioma cantado | `bottom` |
| `layer-backing` | `backing` | `translation` / `backing` | idioma destino | `top` |
| `layer-fx` | `fx` | `fx` | — | `top-right` |

**Las tres salen siempre**, aunque el trabajo no lleve traducción: en ese caso
la capa 1 va vacía y marcada `backing` en vez de `translation` (un canal vacío
no es traducción de nada). El motivo es que el `order` tiene que significar lo
mismo en todas las exportaciones — si la capa 1 desapareciera, la de FX pasaría
de `order: 2` a `order: 1` y una escena de LiveWallpaper montada sobre «la capa
2» se rompería según el trabajo que se abriese.

La capa 2 no lleva letra: lleva comandos para el renderer, y su
`suppressClipText: true` es lo que la convierte en un canal de órdenes. El
Transcriptor la emite vacía y no la interpreta nunca.
Ver [contrato-capa-fx.md](contrato-capa-fx.md).

Dos capas de texto **nunca** comparten ancla: los dos renderizadores colocan
los clips de una capa de forma absoluta según su preset, así que dos capas con
`bottom` se dibujan una encima de la otra.

### Romaji

El romaji sustituye el texto de la capa principal y el original se conserva en
`clip.originalText`. La capa sigue siendo `role: "primary"` —es la voz
principal, y un renderizador que oculte las capas de transliteración no debe
dejar la pantalla sin letra— y declara `romanized: true`.

Si algún día hace falta el japonés y el romaji **a la vez**, el sitio correcto
es una tercera capa con `role: "transliteration"` y su propia ancla. No existe
hoy porque nadie la ha pedido.

## 4. API local

Base: `http://127.0.0.1:8000` por defecto; configurable (§6). Las rutas de
`links` son **relativas**: se resuelven contra la base por la que el cliente
llegó, así que `localhost` y `127.0.0.1` no se convierten en dos orígenes.

| Método | Ruta | Para qué |
|--------|------|----------|
| `GET` | `/api/health` | ¿Está vivo el motor? No toca disco. |
| `GET` | `/api/info` | Modelos, idiomas y el bloque `service` (§6). |
| `GET` | `/api/jobs` | Trabajos guardados. |
| `GET` | `/api/jobs/{id}` | Metadata del trabajo, con `links`. |
| `GET` | `/api/jobs/{id}/lyrixa` | El proyecto `.lyrixa.json`. `?download=1` fuerza descarga. |
| `GET` | `/api/jobs/{id}/audio` | El audio original. Acepta `Range`. |
| `GET` | `/api/jobs/{id}/stems/{vocals\|instrumental}` | Las pistas separadas. |
| `GET` | `/api/jobs/{id}/events` | SSE del progreso, con reconexión por `Last-Event-ID`. |
| `POST` | `/api/translate` | Traducción y romaji como servicio. |
| `POST` | `/api/jobs` · `/api/jobs/{id}/resume` · `/api/jobs/{id}/translate` | Lanzar y retomar trabajos. |
| `DELETE` | `/api/jobs/{id}` | Borrar un trabajo. |

`GET /api/jobs/{id}` incluye:

```json
{
  "links": {
    "self": "/api/jobs/068e7448eb91",
    "events": "/api/jobs/068e7448eb91/events",
    "lyrixa": "/api/jobs/068e7448eb91/lyrixa",
    "audio": "/api/jobs/068e7448eb91/audio",
    "openInLyrixa": "http://127.0.0.1:5173/?job=068e7448eb91&transcriptor=http%3A%2F%2F127.0.0.1%3A8000"
  }
}
```

Una clave ausente significa que ese recurso no existe todavía en el trabajo, no
que la ruta esté rota.

Las rutas son **relativas** a propósito: Lyrixa habla con el backend a través
de un prefijo del mismo origen (`/transcriptor`, que su proxy de Vite
redirige), y una URL absoluta le obligaría a pedir permiso CORS sin
necesidad. La única absoluta es `openInLyrixa`, que por definición apunta a
otra aplicación.

Los nombres de los dos parámetros no son elección del Transcriptor: son los
que lee `resolveBridgeEndpoints` y `readRequestedJobId` en
`src/core/bridge/bridgeConfig.ts`. Viven en `config.LYRIXA_JOB_PARAM` y
`config.LYRIXA_API_PARAM`, y hay una prueba que los fija. Mandar
`transcriptor=` es lo que hace que el enlace funcione con el puerto real: sin
ese parámetro Lyrixa cae en su valor por defecto y el enlace solo sirve
mientras corra el proxy de desarrollo.

### `POST /api/translate`

Una línea suelta o una capa entera. Se responde en la misma forma en que llegó.

```json
{ "text": "I wanna know what love is", "sourceLanguage": "en", "targetLanguage": "es" }
```

```json
{
  "text": "Quiero saber qué es el amor",
  "lines": ["Quiero saber qué es el amor"],
  "sourceLanguage": "en", "targetLanguage": "es",
  "route": [["en", "es"]], "routeLabel": "Inglés → Español",
  "backend": "ollama"
}
```

- `lines: [...]` en lugar de `text` para traducir varias de una vez; la
  respuesta trae `lines` y no `text`. El orden se conserva línea a línea.
- `romaji: true` añade `romaji` (y `romajiText` si se pidió con `text`).
- Sin `sourceLanguage`, se traduce igual: `route` sale como `[["auto","es"]]`.
  Devolver el texto tal cual sería dejar creer que eso es la traducción.
- Origen y destino iguales → el texto vuelve intacto con `route: []`. No es un
  error.
- Escrituras no latinas pivotan por inglés: `ja→en→es`.
- Errores: `400` petición inválida, `413` más de 4000 líneas, `503` no hay
  traductor local, `502` el traductor falló.

## 5. CORS

El servidor escucha en loopback, así que el riesgo no es la red: es una página
remota usando el navegador del usuario como puente. Por eso el permiso por
defecto es cualquier puerto de `localhost` / `127.0.0.1` y nada más, sin
credenciales. `https://ejemplo.com` no recibe cabecera y el navegador le corta
la respuesta.

Se ajusta con `TRANSCRIPTOR_ALLOWED_ORIGIN_REGEX` y
`TRANSCRIPTOR_ALLOWED_ORIGINS`.

## 6. Configuración

Ningún puerto ni URL está escrito dentro de un handler. Todo vive en
`webapp/config.py`:

| Variable | Por defecto | Qué es |
|----------|-------------|--------|
| `TRANSCRIPTOR_HOST` | `127.0.0.1` | Interfaz donde escucha. |
| `TRANSCRIPTOR_PORT` | `8000` | Puerto. |
| `TRANSCRIPTOR_PUBLIC_URL` | derivado | Cómo se ve el backend desde fuera. |
| `TRANSCRIPTOR_LYRIXA_URL` | `http://127.0.0.1:5173/` | Dónde vive Lyrixa. |
| `TRANSCRIPTOR_ALLOWED_ORIGIN_REGEX` | solo localhost | Quién puede llamar. |
| `TRANSCRIPTOR_ALLOWED_ORIGINS` | vacío | Orígenes exactos extra. |

`GET /api/info` devuelve el bloque `service` con estos valores resueltos, más
`bridgeSchemaVersion`, `layerRoles`, `translationLanguages` y `romanizable`,
para que un cliente no tenga que adivinar nada.

## 7. Estado del otro lado

Lyrixa ya tiene su mitad del puente (sin commitear a fecha de hoy):

| Pieza | Dónde | Qué hace |
|---|---|---|
| `TranscriptorClient` | `src/core/bridge/transcriptorClient.ts` | `isAvailable()`, `getJob`, `getProject`, `getAudio`, `postJson`. |
| `resolveBridgeEndpoints` | `src/core/bridge/bridgeConfig.ts` | Resuelve la dirección: URL → localStorage → env → proxy. |
| `useTranscriptorJob` | `src/features/bridge/useTranscriptorJob.ts` | El flujo completo de *Open in Lyrixa*. |
| `reconcileClipLineIdentity` | `src/core/timeline/lineIdentity.ts` | Repara el `sourceId` roto de los proyectos v2. |
| `translationService` | `src/core/translation/translationService.ts` | Pide traducciones aquí en vez de a Ollama. |

Los campos que lee de `GET /api/jobs/{id}` son `id`, `title`, `duration`,
`status`, `files.audio`, `files.lyrixa` y `settings.language`. Están todos, y
hay una prueba de contrato que los cubre.

Un detalle: `settings.language` es el nombre que usa Whisper (`"Japanese"`),
no un código ISO. Sirve para mostrarlo, no para comparar con `layer.language`,
que sí es un código.

Lo que sigue pendiente allí:

1. **Conservar `sourceId` al re-cortar un clip.** Si el usuario parte una línea
   en dos, las dos mitades necesitan identidades nuevas y su traducción tiene
   que seguirlas; heredar el mismo `sourceId` reintroduce el duplicado.
2. **Mostrar `role`/`language` en el inspector de capas**, para que se vea que
   una capa es traducción y no coros.
3. **La capa FX**: autoría y validación del lenguaje de comandos
   ([contrato-capa-fx.md](contrato-capa-fx.md)).

### El puente es de un solo sentido

Lyrixa nunca escribe en `output/`. Todo lo que hace son `GET` más un
`POST /api/translate` que no toca el disco. Por eso el `.lyrixa.json` que
genera el Transcriptor es una **semilla, no un documento compartido**: volver a
traducir un trabajo lo regenera entero, y cualquier ajuste hecho en Lyrixa vive
en el proyecto de Lyrixa, no aquí. No hay pérdida de datos, pero tampoco hay
ida y vuelta.

## 8. Lo que no se debe romper

`.lyrixa.json` v1 y v2 siguen abriéndose: Lyrixa ignora los campos que no
conoce y `sourceId` es opcional. Un proyecto viejo importa igual, solo que sin
el emparejamiento entre capas —que en v1 no existía y en v2 estaba roto.
