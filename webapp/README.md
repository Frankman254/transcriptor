# 🎙️ Transcriptor local

App web para transcribir audio/video **en tu propia máquina** con Whisper.
Muestra en vivo: el **modelo** que se usa, el **dispositivo** (CPU/GPU), la **barra
de progreso**, y **cada segmento con su tiempo de inicio → fin** mientras avanza.

Tiene cuatro modos:
- **🎧 Transcribir** — texto con timestamps de un YouTube o archivo.
- **🎤 Letra de canción** — aísla la voz con BS-RoFormer o Demucs y la
  transcribe. Mucho mejor que transcribir la canción directa.
- **🎯 Letra ultra precisa** — el modo avanzado: separa, detecta dónde hay
  canto, transcribe varias veces con ajustes distintos, vota palabra por
  palabra y repite hasta converger. Después traduce la letra ya confirmada
  para la segunda capa de Lyrixa. Ver [Modo ultra](#modo-ultra).
- **⬇️ Solo audio** — descarga un YouTube en MP3, M4A, Opus, OGG,
  FLAC, ALAC, WAV o en su formato original, sin transcribir.

## Requisitos (ya instalados en esta máquina)
- `python3`
- `yt-dlp` y `ffmpeg` (Homebrew) — para YouTube y para leer audio/video
- `openai-whisper` (pip) — el motor de transcripción
- BS-RoFormer (opcional, recomendado) — se instala con `./setup_separator.sh`
- `demucs` (pip, opcional) — perfiles vocales equilibrado/rápido
- `faster-whisper` (pip) — motor del modo ultra; varias veces más rápido que
  `openai-whisper`, que sigue como respaldo si no está
- `pykakasi` (pip, opcional) — romaji sin conexión cuando no hay Ollama
- [Ollama](https://ollama.com) (opcional) — traducción y romaji en contexto,
  también local: `ollama pull qwen3.5`
- `mutagen` (pip) — permite que `yt-dlp` incruste portadas en todos los
  contenedores compatibles

## Arrancar

```bash
cd webapp
./run.sh
```

Luego abre **http://127.0.0.1:8000** en el navegador.

La primera vez instala `fastapi`, `uvicorn`, `python-multipart` y `mutagen`
automáticamente.

Para habilitar una vez el perfil vocal **Preciso**:

```bash
cd webapp
./setup_separator.sh
```

Se instala en `.separator-venv`, separado de Whisper y Demucs. El modelo de
aproximadamente 610 MB se descarga automáticamente al usarlo por primera vez.

## Cómo funciona
1. Pegas un **link de YouTube** o subes un **archivo** (mp3, m4a, wav, mp4…).
2. Eliges **modelo** (tiny → large) e **idioma** (o auto).
3. El backend:
   - Descarga el audio con `yt-dlp` (si es YouTube), en el formato y bitrate
     elegidos. Incrusta título, artista, álbum, fecha y portada cuando el
     contenedor lo permite (WAV no admite portada).
   - Se asegura de tener el modelo: si no está en `~/.cache/whisper`, lo baja con
     `curl` (evita el error SSL de proxys corporativos).
   - Corre `whisper --verbose True` y **parsea cada segmento** apenas aparece,
     transmitiéndolo al navegador por SSE (Server-Sent Events).
4. Al terminar puedes **descargar** el audio y los archivos `.txt`, `.srt`,
   `.vtt`, `.tsv`, `.json` y `.lyrixa.json`, todos nombrados a partir del audio
   o video original.

El `.json` sigue siendo la salida raw de Whisper. El `.lyrixa.json` es una
derivación independiente: divide los segmentos largos usando exclusivamente los
timestamps reales por palabra y deja clips de letra manejables. Impórtalo en
Lyrixa mediante **More → Import project (.lyrixa.json)**, carga el mismo audio
y revisa la sincronización antes de exportar el bundle para LiveWallpaper.

En el modo **Letra de canción** aparecen además dos reproductores: voz aislada
y pista instrumental. Son los WAV originales creados por el separador elegido
y permiten detectar si una frase incorrecta viene de una mala separación o de
una alucinación de Whisper.

El perfil **Preciso** es el recomendado para mixes: usa BS-RoFormer y procesa
los audios de más de diez minutos por bloques para contener el uso de memoria.
**Equilibrado** ejecuta dos shifts de `htdemucs` con solapamiento `0.5` y
**Rápido** usa una sola pasada con solapamiento `0.25`.

Los resultados quedan en `webapp/output/<job_id>/`.

## Modo ultra

Una sola pasada de Whisper falla de dos maneras: **inventa** texto sobre una
nota sostenida y **se come** palabras al borde de una ventana. Las dos fallas
son inestables — cambia la temperatura o mueve el corte y se mueven con él.
La letra real no se mueve. El modo ultra explota justo esa diferencia.

```
separar voz → medir la separación → VAD → ventanas → N pasadas en paralelo
      → alineación y voto por palabra → ¿cambió algo? → sí: otra pasada
                                                      → no: traducir
```

1. **Separación medida.** El perfil *Ultra* pasa BS-RoFormer dos veces y se
   queda con la que mide mejor: la segunda solo gana si baja la fuga
   instrumental **sin** perder canto. La app muestra nivel, cobertura de voz y
   fuga, así que sabes si una frase mala viene del separador o de Whisper.
2. **VAD.** Sobre la voz aislada, el umbral sale del propio ruido de fondo de
   la pista. Solo se transcribe donde hay canto: eso quita de raíz la
   alucinación sobre silencio y, en una mezcla de tres horas, quita también la
   mitad del cómputo.
3. **Pasadas decorrelacionadas.** Cada pasada usa otro `beam`/temperatura y,
   sobre todo, **corta las ventanas en otro sitio**. Dos pasadas idénticas
   darían el mismo error cinco veces; lo que hace útil el voto es que
   discrepen. Corren en varios procesos, cada uno con su copia del modelo.
4. **Voto (ROVER).** Las hipótesis se alinean con Needleman–Wunsch — guiado por
   los tiempos, para que un estribillo repetido a los tres minutos no se
   alinee con el del minuto uno — y cada posición vota. Una palabra que oyó
   una sola pasada pierde contra el silencio que oyeron las demás; una que
   oyeron todas sobrevive aunque la escriban distinto.
5. **Convergencia.** Tras cada ronda se compara el texto con el de la ronda
   anterior. Cuando una pasada más cambia menos del 1%, para: lo que queda de
   desacuerdo está en el audio, no en el decodificador.
6. **Bucles.** Si *todas* las pasadas se enganchan en la misma repetición, el
   voto no puede verlo: coinciden, pero coinciden en algo falso. Un guardia
   aparte recorta las repeticiones descontroladas y deja las de verdad.
7. **Muletillas de subtítulo.** Whisper aprendió de vídeo subtitulado y suelta
   «ご視聴ありがとうございました» o «Thank you.» sobre la música. Todas las
   pasadas las inventan igual, así que el voto tampoco las ve. Se descartan por
   lista, y **solo cuando ocupan la línea entera**: «Thank you for the love you
   gave me» es letra y se queda. En un mix de una hora esto fueron 75 de 507
   líneas.

Solo entonces se traduce. Traducir texto que todavía puede cambiar es trabajo
tirado, y una traducción hecha sobre texto inventado se ve perfectamente
correcta — que es la peor forma de estar mal.

### Idioma, romaji y traducción

- **Detección automática por voto**: se muestrean hasta ocho puntos repartidos
  por la canción en vez de fiarse de los primeros 30 segundos. Un tema japonés
  que abre con un sample en inglés se detecta bien.
- **Japonés → romaji**: lo hace el modelo local, que sabe la lectura *en
  contexto* (君の名は → `kimi no na wa`). `pykakasi` queda de respaldo sin
  conexión, pero es un diccionario sin análisis morfológico y produce
  `kun no mei ha`.
- **Traducción con pivote**: los idiomas de escritura no latina pasan por
  inglés (ruso → inglés → español). Es la ruta con más masa de entrenamiento
  en cualquier modelo que quepa en esta máquina.
- **Una línea, una traducción**: cada lote se pide numerado y se verifica; si
  vuelve mal, se reintenta y luego se traduce línea por línea. Si aun así falla
  una, se conserva el original. Una línea sin traducir se ve; una capa
  desplazada, no.

### Salidas

Además de `.txt`/`.srt`/`.vtt`/`.tsv`/`.json` (idénticos a los del modo
normal), el modo ultra escribe:

- `<título>.letra.txt` — original, romaji y traducción una debajo de otra,
  para revisarlo sin abrir un JSON de varios megas.
- `<título>.lyrixa.json` — el proyecto con **dos capas de texto**:
  `layer-main` (`role: "primary"`) con la letra, y `layer-backing`
  (`role: "translation"`) con la traducción en los mismos tiempos al
  milisegundo. Cada clip traducido apunta con `sourceId` al clip que traduce.
  Si hubo romaji, el clip principal lleva el romaji en `text` y el original en
  `originalText`.

En la interfaz, las líneas con menos del 80% de acuerdo entre pasadas salen
marcadas en ámbar: son las que conviene escuchar antes de darlas por buenas.

### Cuánto tarda

El modo ultra hace N veces el trabajo de una transcripción normal, pero solo
sobre las partes con canto. En una mezcla donde se canta la mitad del tiempo,
cinco pasadas cuestan aproximadamente 2,5 transcripciones completas. Para
audios de una a tres horas, `turbo` con el nivel *Equilibrado* es el punto
sensato; `large-v3` con *Máximo* es para cuando la letra importa más que la
tarde.

## Modelos
| Modelo | Calidad | Velocidad (CPU) |
|--------|---------|-----------------|
| tiny   | baja    | muy rápida      |
| base   | baja+   | rápida          |
| small  | media   | media           |
| medium | alta    | ~1x duración    |
| large-v3 | máxima | lenta          |
| turbo  | alta    | rápida          |

El primer uso de un modelo nuevo lo descarga una vez (~1–3 GB según el modelo).

## Notas
- Todo corre **local**: el audio no sale de tu máquina.
- Sin GPU, `medium` tarda aproximadamente lo que dura el audio. Para audios
  largos, `turbo` o `small` son bastante más rápidos.
