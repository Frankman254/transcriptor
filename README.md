# 🎙️ Transcriptor local

App web para transcribir audio/video y sacar letras de canciones **100% en tu
propia máquina** con [Whisper](https://github.com/openai/whisper), BS-RoFormer y
[Demucs](https://github.com/facebookresearch/demucs). El audio nunca sale de tu
equipo: no usa ninguna API en la nube.

Progreso en vivo: modelo, dispositivo (CPU/GPU), barra de avance y cada segmento
con su tiempo `inicio → fin` conforme se genera (vía SSE).

## Modos

| Modo | Qué hace |
|------|----------|
| 🎧 **Transcribir** | Texto con timestamps de un link de YouTube o un archivo. |
| 🎤 **Letra de canción** | Aísla la voz con BS-RoFormer o Demucs y luego la transcribe. |
| 🎯 **Letra ultra precisa** | Consenso de varias transcripciones + traducción. Ver abajo. |
| ⬇️ **Solo audio** | Descarga audio de YouTube en MP3, M4A, Opus, OGG, FLAC, ALAC, WAV u original. |

## Modo ultra: por qué deja de inventar letra

Una pasada de Whisper a veces inventa una frase sobre una nota sostenida y a
veces se come una al borde de una ventana. Las dos fallas son **inestables**:
cambia la temperatura de decodificación o mueve el corte de ventana y se mueven
con ellas. La letra real no se mueve.

El modo ultra transcribe el mismo audio varias veces con ajustes deliberadamente
distintos, alinea las hipótesis palabra por palabra y deja que cada posición
vote. Una palabra que oyó una sola pasada pierde contra el silencio que oyeron
las demás; una que oyeron todas sobrevive aunque la escriban distinto. Después
vuelve a votar con una pasada más, y para cuando el texto deja de cambiar —
ahí, el desacuerdo que queda está en el audio, no en el decodificador.

Antes de transcribir mide la separación (nivel, cobertura de canto y fuga
instrumental) y usa detección de actividad vocal para transcribir **solo donde
se canta**: eso elimina de raíz la alucinación sobre silencio y, en una mezcla
de tres horas, ahorra la mitad del cómputo.

El desglose completo — perfil de separación de doble pasada, ventanas
desfasadas, alineación con Needleman–Wunsch, guardia de bucles de repetición y
tiempos esperados — está en [`webapp/README.md`](webapp/README.md#modo-ultra).

## Traducción e idiomas (local)

Cuando la letra ya está confirmada, y no antes, el modo ultra la traduce con un
modelo local vía [Ollama](https://ollama.com):

- **Detección de idioma por voto** en varios puntos de la canción, así que
  funciona con ruso, japonés o coreano aunque el tema abra con un sample en
  otro idioma.
- **Japonés en romaji**, con la lectura correcta en contexto (君の名は →
  `kimi no na wa`, no `kun no mei ha`).
- **Pivote por inglés** para escrituras no latinas: ruso → inglés → español.
- **Una traducción por línea, siempre**: los lotes se piden numerados y se
  verifican; si algo falla se reintenta línea a línea y, en último caso, se
  conserva el original. Una capa desplazada rompería toda la sincronía.

## Puente hacia Lyrixa

Además de conservar las salidas originales de Whisper, cada transcripción
genera un proyecto de autoría `.lyrixa.json`.

Los segmentos largos se dividen con los timestamps reales de `words[]`. El corte
no es glotón: se minimiza un coste sobre **todos** los repartos posibles a la
vez, premiando los cortes en coma o punto y penalizando los párrafos
desequilibrados. Por eso no sale una línea con veinte palabras y la siguiente
con dos. Si los timestamps por palabra faltan o están incompletos, se conserva
el segmento original completo como fallback.

Con traducción activa, el proyecto llega a Lyrixa con **dos capas de texto**:

| Capa | `role` | Contenido |
|------|--------|-----------|
| `layer-main` | `primary` | La letra (romaji si el original era japonés) |
| `layer-backing` | `translation` | La traducción, con los mismos tiempos al ms |

Cada clip traducido apunta con `sourceId` al clip que traduce, así que las dos
capas siguen emparejadas aunque después edites los tiempos.

Las dos capas de texto salen **en anclas distintas** (letra abajo, traducción
arriba). Los dos renderizadores colocan los clips de una capa de forma absoluta
según su ancla, así que dos capas con la misma se dibujarían una encima de la
otra. Si prefieres la traducción justo debajo de la letra, muévela con
*Position Y* en LiveWallpaper (pestaña Lyrics → Capas del Bundle) o cambia el
ancla en el inspector de capas de Lyrixa.

### Cuándo desaparece una línea

El timestamp de la última palabra marca dónde acaba el *alineamiento del
token*, no dónde deja de sonar la vocal cantada. Una línea que se apaga en ese
instante se siente como si el párrafo cambiara antes de que termine la voz. Por
eso cada línea **se queda 0,7 s de más**, pero solo hacia el silencio que la
sigue: si otra línea entra antes, se corta 50 ms antes de ella y le cede el
turno. Nunca hay dos líneas de la misma capa en pantalla a la vez.

En Lyrixa usa **More → Import project (.lyrixa.json)** y después carga el mismo
audio para revisar el timeline. Tras editar, usa **Export lyrics bundle** para
obtener el `.lyrixa-lyrics.json` que consume LiveWallpaperAnimeGlitch, donde la
capa de traducción se puede mostrar u ocultar desde la pestaña de Lyrics.

Al terminar el modo de letras, el resultado también muestra reproductores y
descargas para la voz aislada y el instrumental. Se sirven directamente desde
la salida del separador elegido, sin recomprimir ni duplicar los WAV, para poder
auditar qué material recibió Whisper.

La separación ofrece cuatro perfiles. **Ultra** pasa BS-RoFormer dos veces y se
queda con la que mide mejor — la segunda solo gana si baja la fuga instrumental
sin perder canto, así que nunca sale peor que Preciso. **Preciso**
(predeterminado) es una pasada de BS-RoFormer y recupera bien voces débiles;
**Equilibrado** usa Demucs con dos shifts y 50% de solapamiento; **Rápido** usa
una sola pasada de Demucs.

## Arrancar

```bash
cd webapp
./run.sh
```

Abre **http://127.0.0.1:8000** en el navegador.

## Requisitos

- `python3`
- `ffmpeg` y `yt-dlp` (para leer audio/video y descargar de YouTube)
- `openai-whisper` (motor de transcripción)
- BS-RoFormer *(opcional, recomendado para letras)*: `cd webapp && ./setup_separator.sh`
- `demucs` *(opcional)* — perfiles equilibrado/rápido: `pip3 install demucs`

Para el modo ultra:

- `faster-whisper` — varias veces más rápido, que es lo que hace viable correr
  cinco pasadas sobre una mezcla de tres horas. Sin él, el modo ultra sigue
  funcionando con `openai-whisper`, solo que más lento.
- [Ollama](https://ollama.com) con un modelo multilingüe, para la traducción y
  el romaji: `ollama pull qwen3.5`. Sin él la transcripción funciona igual;
  solo se pierde la segunda capa.
- `pykakasi` *(opcional)* — romaji de respaldo sin Ollama.

Las dependencias del backend (`fastapi`, `uvicorn`, `python-multipart`,
`mutagen`, `numpy`, `faster-whisper`, `pykakasi`) las instala `run.sh` la
primera vez.

Las descargas desde YouTube conservan el título original del video. Cuando el
formato lo permite, también incluyen título, artista, álbum, fecha y portada
incrustada. WAV no admite portada.

Más detalles en [`webapp/README.md`](webapp/README.md).

## Tests

```bash
python3 -m unittest discover -s tests -v
```
