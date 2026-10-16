# Trabajar en el Transcriptor

Motor local en Python (FastAPI + Whisper + BS-RoFormer/Demucs + Ollama) que
saca la letra de un audio y la entrega a **Lyrixa** como `.lyrixa.json`. Es
parte de una suite de cuatro proyectos: lee
**[.agents/SUITE.md](.agents/SUITE.md)** antes de nada.

**Qué hacer ahora:** [.agents/TAREAS.md](.agents/TAREAS.md).

## Rutas de los proyectos vecinos (esta máquina)

| Proyecto                         | Ruta                                                                     |
| -------------------------------- | ------------------------------------------------------------------------ |
| Lyrixa (consume `.lyrixa.json`)  | `/Users/frankman254/Desktop/Personal-Projects/Lyrixa`                    |
| Vibrix (renderiza el vídeo)      | `/Users/frankman254/Desktop/Personal-Projects/LiveWallpaperAnimeGlitch`  |
| Web de venta                     | `/Users/frankman254/Desktop/Personal-Projects/vibrix-web`                |

## Leer antes de escribir código

1. [README.md](README.md) y [webapp/README.md](webapp/README.md) — modos, API, modelos.
2. [docs/arquitectura.md](docs/arquitectura.md) — capas y las cuatro reglas.
3. [docs/contrato-lyrixa.md](docs/contrato-lyrixa.md) — lo que Lyrixa espera. El
   lado que lee está en `/Users/frankman254/Desktop/Personal-Projects/Lyrixa/src/core/bridge/transcriptorClient.ts`.
4. [docs/contrato-capa-fx.md](docs/contrato-capa-fx.md) — lenguaje de efectos.
5. Papel del Transcriptor en el lanzamiento:
   `/Users/frankman254/Desktop/Personal-Projects/LiveWallpaperAnimeGlitch/docs/plans/PLAN_MAESTRO_LANZAMIENTO.md` §0.5 punto 3.

## Reglas duras

- **Nada por debajo de `server` importa `server`.** Las funciones del pipeline
  reciben `emit`.
- **`paths` es el único dueño de las rutas:** usar `paths.OUTPUT_DIR`, nunca
  `from paths import OUTPUT_DIR`.
- **`jobstore` no sabe qué está corriendo:** recibe `running=`.
- Un módulo nuevo necesita su fila en `tests/test_module_boundaries.py`
  y su `__all__`: **lo que no está en `__all__` no lo toca otro módulo**
  (sus propias pruebas sí).
- Ningún host ni puerto escrito a mano: todo sale de `webapp/config.py`
  (`TRANSCRIPTOR_*`).
- **YouTube / yt-dlp es solo para uso personal.** Nunca en un build, imagen o
  servicio que se distribuya o cobre (viola los términos de YouTube).
- El `.lyrixa.json` es un contrato: añadir campos opcionales está bien;
  renombrar o quitar exige versionar y avisar en `docs/contrato-lyrixa.md`
  y en Lyrixa el mismo día.
- Los audios y salidas personales de la raíz están en `.gitignore`: no los
  borres ni los muevas.
- Docs en español; commits solo si el usuario lo pide.

## Antes de terminar

```bash
python3 -m unittest discover -s tests -v
```
