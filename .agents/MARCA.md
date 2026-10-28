# Línea visual de la suite — guía para agentes

> Archivo idéntico en los cuatro proyectos (`.agents/MARCA.md`), igual que
> `SUITE.md`. Si lo cambias, cópialo a los demás. Creado 2026-10-18.

Tres productos, una familia: cada uno tiene su propio acento y su propio
motivo, pero los tres iconos se dibujan con la misma gramática. Quien vea los
tres favicons juntos tiene que reconocer que vienen de la misma casa.

| Producto    | Repo                       | Papel                        | Monograma | Motivo de fondo                     | Acento            |
| ----------- | -------------------------- | ---------------------------- | --------- | ----------------------------------- | ----------------- |
| **Vibrix**  | `LiveWallpaperAnimeGlitch` | Convierte la letra en vídeo  | `V`       | Anillos concéntricos (sonido)       | Cian `#21dde8`    |
| **Lyrixa**  | `Lyrixa`                   | Edita y sincroniza la letra  | `L`       | Línea de tiempo: tiempos \| bloques | Ámbar `#f5b83d`   |
| **Scribix** | `transcriptor`             | Saca la letra del audio (IA) | `S`       | Renglones de texto                  | Violeta `#a47cff` |

**Scribix** es el nombre visible del Transcriptor. El repo, los módulos, las
variables `TRANSCRIPTOR_*` y los contratos (`.lyrixa.json`) siguen llamándose
"transcriptor": el nombre solo cambia lo que ve una persona.

## Nombres

Patrón de la suite: raíz que dice el oficio + terminación `-ix` / `-ixa`
(_vibe_ → Vibrix, _lyric_ → Lyrixa, _scribe_ → Scribix). Un producto nuevo
sigue el mismo patrón. En la interfaz el nombre va en **mayúsculas con
espaciado ancho** (`letter-spacing: .18–.2em`, peso 600), al lado del
monograma; un subtítulo opcional va en minúsculas y en gris.

## Gramática del icono

Lienzo `viewBox="0 0 64 64"`. Las referencias exactas son
`LiveWallpaperAnimeGlitch/public/favicon.svg`, `Lyrixa/public/favicon.svg` y
`transcriptor/webapp/static/brand/favicon.svg`: cópialas en vez de redibujar.

1. **Monograma**: un solo trazo (`stroke-width="5.5"`, extremos y uniones
   redondos, sin relleno), dibujado tres veces:
    - magenta `#f0409b` con `translate(-2.6 0)` (Vibrix usa `-3`);
    - cian `#21dde8` con `translate(2.6 0)`;
    - neutro `#F2F3F5` encima, sin desplazar.

    Esa aberración cromática magenta/cian es **la firma de la familia**: va en
    todos los iconos, sea cual sea el acento del producto.

2. **Motivo de fondo**: 2–3 elementos que cuentan qué hace el producto. El
   primero (arriba o por fuera) en neutro; los demás en el acento del
   producto.
3. **Perforación**: una `<mask>` con el mismo trazo del monograma en negro y
   `stroke-width` ≈ 9.5–10 recorta el fondo alrededor de la letra, para que
   la letra lo "atraviese" con aire.
4. **Variantes**, por producto:

    | Archivo                       | Fondo        | Uso                                                                                                                                                   |
    | ----------------------------- | ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
    | `favicon.svg`                 | Transparente | Pestaña del navegador. El neutro pasa a `#14171d` con `@media (prefers-color-scheme: light)`.                                                         |
    | `<producto>-mark.svg`         | Transparente | Dentro de la app, sobre la barra oscura. **Sin** media query: en un `<img>` manda el tema del sistema, no el de la página, y el neutro desaparecería. |
    | `<producto>-logo.svg`         | Placa        | Fuente de los PNG, manifest y README.                                                                                                                 |
    | `<producto>-logo-192/512.png` | Placa        | `apple-touch-icon`, manifest (`any maskable`), Safari (ignora media queries en SVG).                                                                  |

    **Placa** (en 512): `rect` de 480 con `rx="112"`, degradado vertical del
    fondo de la app, un brillo radial del acento al 13–16 % y un borde de 2 px
    blanco al 8 %. El monograma va con `translate(64 64) scale(6)`.

Los PNG salen del SVG con Chrome sin ventana, sin dependencias nuevas:

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless --disable-gpu --hide-scrollbars --default-background-color=00000000 --window-size=192,192 --screenshot=logo-192.png file:///ruta/a/una/pagina-que-solo-tiene-el-img.html
```

## `<head>` mínimo de cada app

`description`, `application-name`, `apple-mobile-web-app-title`,
`theme-color` (el fondo de la app), `icon` SVG, `icon` PNG 192,
`apple-touch-icon` y `manifest` (`theme_color` = acento,
`background_color` = fondo). Vibrix (`LiveWallpaperAnimeGlitch/index.html`)
es el modelo.

## Colores y tipografía

- **Tema oscuro** en las tres apps. Cada una tiene su rampa de grises (Vibrix
  y Scribix fríos, Lyrixa cálido a propósito); no se unifican.
- **Texto sobre un botón de acento: oscuro**, nunca blanco (con blanco el
  cian, el ámbar y el violeta se quedan en ~3:1 de contraste). Cada app
  define su `--accent-fg`.
- Magenta y cian de la suite son **de marca**: logo, degradados de progreso,
  selección de texto. No se usan como estado (ok / aviso / error).
- **Monoespaciada común: JetBrains Mono.** La sans de interfaz es hoy Inter
  en Vibrix, Geist en Lyrixa y Scribix, y Space Grotesk + Inter en la web de
  venta. Unificarla es una decisión del usuario, no de un agente.

## Estado y pendientes por repo

Marca lo que cierres y pon la fecha.

- **Vibrix** — referencia. Hecho.
- **Scribix** (`transcriptor`) — hecho 2026-10-18: nombre, iconos y manifest
  en `webapp/static/brand/`, acento violeta, Geist + JetBrains Mono.
    - [ ] Opcional: las pestañas y paneles usan emojis como iconos (🔗 📁 🎧 🎚️…);
          cambiarlos por iconos de línea de un solo color.
- **Lyrixa** — 2026-10-18: `public/favicon.svg` (antes era el rayo por
  defecto de Vite), `lyrixa-logo.svg`, PNG 192/512, `site.webmanifest` y
  `<head>` completo.
    - [ ] Cambiar el cuadrado con la letra "L" en degradado de
          `src/features/editor/EditorTopBar.tsx` (`.brand-mark` en
          `LyrixaEditorShell.css`) por `<img src="/lyrixa-mark.svg">` (crear la
          variante sin media query a partir de `public/favicon.svg`).
    - [ ] Borrar `src/assets/vite.svg`, `src/assets/react.svg` y
          `public/icons.svg` si nada los usa (`grep` antes).
- **Web de venta** (`vibrix-web`) — usa el icono de Vibrix. Hecho.
    - [ ] Cuando la web enseñe Lyrixa o Scribix (flujo letra → sincronía →
          visual, lista de espera), copiar sus `-logo.svg` a `public/brand/` en
          vez de dibujar otros.
