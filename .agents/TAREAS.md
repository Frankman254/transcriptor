# Transcriptor — qué hacer ahora

> Fuente: plan maestro de la suite
> (`/Users/frankman254/Desktop/Personal-Projects/LiveWallpaperAnimeGlitch/docs/plans/PLAN_MAESTRO_LANZAMIENTO.md`,
> §0.5 punto 3 y tarea 4.9). Actualiza este archivo al cerrar cada tarea.

**Papel en el lanzamiento:** herramienta **interna** hasta después de
Vibrix `1.0.0`. No se vende ni se expone públicamente todavía, porque necesita
Python, modelos de varios GB y GPU (no corre en el navegador del cliente) y
porque la precisión aún no es fiable en mezclas densas. Sirve para producir
rápido las letras de las plantillas y de los vídeos de marketing. Más adelante
(Fase 6) será "Auto-letras", una función Pro que siempre abre el resultado en
Lyrixa para revisión humana.

## T1 · Aislar YouTube detrás de un flag — prioridad alta

- Nueva variable `TRANSCRIPTOR_ENABLE_YOUTUBE` en `webapp/config.py`,
  **por defecto `0`** (desactivado). Tu `run.sh` local puede exportarla a `1`.
- Con el flag apagado: la UI oculta la entrada por link y el modo "Solo audio",
  y las rutas que descargan responden 403 con un mensaje claro.
- Test en `tests/test_api_contract.py` que falle si con el flag apagado se
  puede lanzar una descarga.
- Documentar en `README.md` que es solo para uso personal.

## T2 · Inventario de licencias de modelos — prioridad alta

Crear `docs/licencias-modelos.md` con, por cada modelo y binario: nombre
exacto del artefacto, licencia del **código** y de los **pesos**, URL de la
fuente y fecha de consulta. Mínimo: Whisper (openai / mlx-community /
Systran faster-whisper), BS-RoFormer (**verificar la licencia de los pesos**),
Demucs, modelos de Ollama usados (`qwen3.5` u otro), pykakasi, ffmpeg y yt-dlp.
Marcar en rojo lo que impediría uso comercial.

## T3 · Calidad en mezclas densas — continuo, sin fecha

Es trabajo del autor y no bloquea nada. Si tocas precisión:

- Añade el caso a `tests/fixtures` antes de cambiar el pipeline.
- Mide antes y después (WER sobre letra de referencia) y anótalo en `webapp/README.md` §Modo ultra.

## T4 · Producir letras para las 12 plantillas de Vibrix — cuando Vibrix llegue a Fase 2

Las plantillas están listadas en el plan maestro, tarea 2.1. El audio debe
tener licencia propia o libre (nada de YouTube). El flujo es Transcriptor →
revisión en Lyrixa → bundle a Vibrix.

## No hacer (congelado hasta Vibrix `1.0.0`)

- Servicio público o en la nube, cuentas, cobros.
- Empaquetar el Transcriptor dentro de Vibrix o Lyrixa.
- Cambiar el contrato `.lyrixa.json` sin coordinarlo con Lyrixa.
