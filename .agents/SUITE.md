# La suite — mapa para agentes

> Archivo idéntico en los cuatro proyectos. Si lo cambias, cópialo a los demás
> (`.agents/SUITE.md` de cada ruta de abajo).

Cuatro proyectos del mismo autor forman **una sola cadena de producción de
vídeo musical**. Todos viven en esta máquina, uno al lado del otro:

```
/Users/frankman254/Desktop/Personal-Projects/
├── transcriptor/              IA: saca la letra del audio      (Python)
├── Lyrixa/                    Edita y sincroniza la letra      (React)
├── LiveWallpaperAnimeGlitch/  Vibrix: la convierte en vídeo    (React)
└── vibrix-web/                Web que vende la suite           (Next.js, por crear)
```

```
 audio                       letra revisada                  vídeo final
   │                               │                              │
┌──▼───────────┐ .lyrixa.json ┌────▼──────┐ .lyrixa-lyrics.json ┌─▼──────┐
│ Transcriptor │ ───────────► │  Lyrixa   │ ──────────────────► │ Vibrix │
└──────────────┘  HTTP local  └───────────┘   archivo o HTTP    └────────┘
```

## Rutas

| Proyecto                | Ruta absoluta                                                            | Stack                                    | Paquetes | Tests / checks                                                                                 |
| ----------------------- | ------------------------------------------------------------------------ | ---------------------------------------- | -------- | ---------------------------------------------------------------------------------------------- |
| **Vibrix** (ancla)      | `/Users/frankman254/Desktop/Personal-Projects/LiveWallpaperAnimeGlitch`  | Vite + React 19 + TS + Zustand + R3F     | `pnpm`   | `pnpm format && pnpm lint && pnpm architecture:check && pnpm test:run && pnpm build` (ver AGENTS.md) |
| **Lyrixa**              | `/Users/frankman254/Desktop/Personal-Projects/Lyrixa`                    | Vite + React 19 + TS                     | `npm`    | `npm test && npm run lint && npm run build`                                                    |
| **Transcriptor**        | `/Users/frankman254/Desktop/Personal-Projects/transcriptor`              | Python 3 + FastAPI + Whisper             | `pip`    | `python3 -m unittest discover -s tests -v`                                                     |
| **Web de venta**        | `/Users/frankman254/Desktop/Personal-Projects/vibrix-web`                | Next.js + Tailwind CSS (por crear)       | `pnpm`   | `pnpm lint && pnpm build`                                                                      |
| Landing personal (ref.) | `/Users/frankman254/Desktop/Personal-Projects/Landing-page-frankmandev` | Next.js 16 + Tailwind 3 (solo referencia) | `npm`    | —                                                                                              |

Repos en GitHub: `Frankman254/Lyrixa`, `Frankman254/transcriptor`; Vibrix y la
web tienen el suyo propio.

## Documentos que mandan

| Qué                                        | Dónde                                                                                         |
| ------------------------------------------ | --------------------------------------------------------------------------------------------- |
| **Plan maestro de lanzamiento** (la suite) | `LiveWallpaperAnimeGlitch/docs/plans/PLAN_MAESTRO_LANZAMIENTO.md`                             |
| **Plan de la web de venta**                | `LiveWallpaperAnimeGlitch/docs/plans/PLAN_WEB_VENTA.md`                                       |
| Contrato Lyrixa → Vibrix                   | `LiveWallpaperAnimeGlitch/docs/features/LYRIXA_CONTRACT.md`                                   |
| Contrato Transcriptor → Lyrixa             | `transcriptor/docs/contrato-lyrixa.md`                                                        |
| Lenguaje de efectos (capa 3)               | `transcriptor/docs/contrato-capa-fx.md`                                                       |
| Arquitectura de cada app                   | Vibrix `docs/architecture/ARCHITECTURE.md` · Lyrixa `docs/01-architecture.md` · Transcriptor `docs/arquitectura.md` |

Todos los paths de esta tabla son relativos a
`/Users/frankman254/Desktop/Personal-Projects/`.

## Reglas que valen para los cuatro

1. **Los contratos entre apps no se rompen.** Un cambio de formato sube la
   versión del sobre (`version` del bundle/proyecto) y el lado que lee acepta
   también la versión anterior. Si tocas un contrato, actualiza el doc de los
   **dos** lados en el mismo día.
2. **Un agente trabaja en un solo repo por tarea.** Puede _leer_ los vecinos
   para entender un contrato, pero no los modifica sin que el usuario lo pida.
3. **Vibrix se lanza primero.** Lyrixa sale con Vibrix (mismo dominio);
   el Transcriptor queda como herramienta interna hasta después de `1.0.0`.
   El porqué está en el plan maestro §0.5.
4. **Nada de YouTube/yt-dlp en nada que se distribuya o se venda.**
5. **Arte y audio de demo con licencia propia o libre.** Nada de personajes de
   anime con copyright en plantillas, web o vídeos de marketing.
6. **Docs en español.** Código, identificadores y commits en inglés.
7. Commits solo cuando el usuario lo pida; si estás en `main`, crea rama antes.

## Estado del plan (actualizar al cerrar cada fase)

| Fase | Qué                              | Estado                          |
| ---- | -------------------------------- | ------------------------------- |
| 0    | Higiene (Vibrix `0.4.1-alpha`)   | Hecha 2026-09-13                |
| 1    | Export de vídeo offline (Vibrix) | **En curso**                    |
| 2    | Plantillas + letras LRC/SRT      | Pendiente (Lyrixa puede adelantar sus exports) |
| 3    | Rendimiento                      | Pendiente                       |
| 4    | Pagos, legal, web de venta       | Pendiente (la web puede empezar su maqueta ya) |
| 5    | Beta → `1.0.0`                   | Pendiente                       |

## Skills disponibles

Las skills de Matt Pocock están instaladas **globalmente** como plugin de
Claude Code (`mattpocock-skills`, scope usuario). No se copian a cada repo
(instalar las dos variantes duplica todo). Las más útiles aquí:

- `/grill-with-docs` o `/grill-me` — antes de una tarea ambigua: te hace
  preguntas hasta alinear qué se construye.
- `/to-spec` → `/to-tickets` → `/implement` — de idea a tickets pequeños.
- `/tdd`, `/diagnosing-bugs`, `/code-review`, `/improve-codebase-architecture`.
- `/handoff` — dejar el estado listo para el siguiente agente.

Una vez por repo, el usuario puede correr `/setup-matt-pocock-skills`
(es interactivo: elige tracker de issues y dónde viven `CONTEXT.md`/ADRs).
Para Codex/Cursor: `npx skills@latest add mattpocock/skills`.
