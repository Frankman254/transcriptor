# Contrato de la capa 3 — el lenguaje de efectos

> Estado: **propuesta cerrada, sin implementar**. Transcriptor ya emite el
> canal; Lyrixa y LiveWallpaper todavía no lo interpretan. Este documento es
> lo que tienen que implementar para entenderse.

## 1. La idea

Las cuatro capas del proyecto son un canal cada una:

| # | id | role | Qué lleva |
|---|----|------|-----------|
| 0 | `layer-main` | `primary` | La letra cantada, en su escritura original. |
| 1 | `layer-backing` | `translation` / `backing` | Traducción o voz secundaria. |
| 2 | `layer-romaji` | `transliteration` | Cómo suena la letra en alfabeto latino. |
| 3 | `layer-fx` | `fx` | **Órdenes para el renderer.** |

Las tres primeras son lecturas del mismo verso y comparten su `sourceId`. La
cuarta es de otra naturaleza: no lleva letra. Lleva comandos: un clip que dice `glitch 0.8` no es
texto que se dibuje, es una orden que enciende el glitch mientras ese clip
dura.

Esto no necesita un formato nuevo. Un clip de Lyrixa ya es *texto con un
inicio y un final*, y la capa FX ya sale marcada con
`renderSettings.suppressClipText: true`, que significa «no dibujes el texto de
estos clips». Ese campo es justo lo que convierte la capa en un canal de
comandos: **el texto del clip es el comando, y el clip es su duración.**

La consecuencia práctica es que el editor de efectos ya está construido.
Arrastrar un clip lo alarga, moverlo lo retrasa, recortarlo lo acorta. No hace
falta una interfaz aparte.

## 2. Sintaxis

Un clip contiene una o varias órdenes separadas por `;`.

```
orden   := ["!"] efecto argumento*
efecto  := [a-z][a-z0-9-]*          (sin distinguir mayúsculas)
argumento :=
      0.0 … 1.0        intensidad          glitch 0.8
    | #rrggbb          color               lights #ff2a6d
    | clave=valor      parámetro con nombre particles density=high
    | palabra          preajuste o modo    particles burst
```

Ejemplos:

```
glitch
glitch 0.8
scanlines 0.3 ; rgbshift 0.5
lights #ff2a6d
particles burst density=high
!glitch
clear
```

## 3. Semántica

Tres reglas, y de la tercera dependen las otras dos.

**1. Un clip es un intervalo, no un instante.** `glitch` entre 12.0 y 14.5
significa glitch encendido en 12.0 y apagado en 14.5. No hay que escribir la
orden de apagado: la escribe el borde del clip.

**2. `!efecto` es un evento puntual en `startTime`.** Apaga el efecto y no
ocupa duración. Sirve para cortar antes de tiempo algo que sigue abierto.
`clear` apaga todo.

**3. El estado en el instante T es función únicamente de los clips y de T.**
Nada depende de por dónde pasó la reproducción antes. Esta es la regla que
manda, y es la razón de que el lenguaje no tenga interruptores: con un
`glitch` que alterna, saltar el cabezal al minuto 3 deja la pantalla en el
estado equivocado, porque el estado correcto dependería de cuántos clips se
cruzaron por el camino. Con intervalos, el renderer se salta el problema
entero — mira qué clips cubren T y ya está.

**Solapes.** Si varios clips del mismo efecto están abiertos en T, gana el que
haya empezado más tarde. Cuando ese termina, el efecto sigue encendido con los
parámetros del anterior si todavía está abierto. Es el comportamiento que
espera cualquiera que haya trabajado con automatización en un DAW, y se
resuelve ordenando por `startTime` sin necesidad de mantener una pila.

**Efectos desconocidos.** Se ignoran, y se registran una vez. Un Lyrixa más
nuevo que el renderer tiene que degradar, no reventar. Lo mismo con un
argumento que no se entienda: se descarta el argumento, no la orden.

**Identidad.** Un clip de FX puede llevar `sourceId` si su autor lo ancló a
una línea de la letra, y entonces re-cronometrar la línea puede arrastrar el
efecto. Si no lo lleva, `resolveClipLineKey` cae en el id del clip y el efecto
es independiente. Las dos formas son válidas; es decisión de quien lo autora.

## 4. Vocabulario inicial

Lo define **LiveWallpaper**, porque es quien implementa los efectos. Lyrixa lo
usa para autocompletar y avisar de un nombre que no existe.

| Efecto | Argumento principal | Qué hace |
|---|---|---|
| `glitch` | intensidad | Corte y desplazamiento de bloques. |
| `rgbshift` | intensidad | Separación de canales de color. |
| `scanlines` | intensidad | Líneas de barrido. |
| `particles` | preajuste (`burst`, `drift`, `rain`) | Sistema de partículas. |
| `lights` | color | Luces de escena. |
| `strobe` | intensidad | Parpadeo. |
| `shake` | intensidad | Sacudida de cámara. |
| `bloom` | intensidad | Halo sobre las zonas claras. |
| `zoom` | `to=` factor | Empuje de cámara. |
| `invert` | — | Inversión de color. |
| `vignette` | intensidad | Oscurecido de bordes. |
| `trails` | intensidad | Estela de fotogramas. |
| `pixelate` | intensidad | Reducción de resolución. |

Y dos que no son efectos sino estructura:

| Orden | Qué hace |
|---|---|
| `scene <nombre>` | Cambia de escena. |
| `clear` | Apaga todos los efectos activos. |

## 5. Quién implementa qué

**Lyrixa** — autora. Escribe los clips, autocompleta el vocabulario, avisa de
un efecto desconocido, y conserva la capa al exportar el bundle de letras.

**LiveWallpaper** — lee. Parsea el texto de cada clip de la capa `fx`,
resuelve el estado en cada fotograma según §3, y despacha a sus efectos.

**Transcriptor** — no participa. Emite el canal vacío para que exista desde el
primer import, y nada más: no autora efectos, no parsea comandos y no conoce
el vocabulario. Un motor de transcripción que decidiera cuándo brillan las
luces sería LiveWallpaper con otro nombre.

## 6. Por qué el canal sale siempre

`build_layers` emite las cuatro capas incluso cuando el trabajo no tiene
traducción ni japonés: la capa 1 va vacía y marcada `backing` en vez de
`translation` (vacía no es traducción de nada), y la capa 2 va vacía y
`visible: false`.

El motivo es el `order`. Si un canal desapareciera cuando está vacío, la capa
FX cambiaría de número según el trabajo, y una escena de Vibrix montada sobre
«la capa 3» se rompería al abrir la siguiente. Cuatro canales fijos cuestan
dos objetos vacíos y hacen que el número de capa signifique siempre lo mismo.

## 7. Lo que falta

- **Lyrixa**: parser y validación al escribir, autocompletado, y decidir si un
  clip de FX se ancla a una línea (`sourceId`) o vive suelto.
- **LiveWallpaper**: el registro de efectos, el resolutor de estado de §3, y
  el descarte silencioso de lo desconocido.
- **Ambos**: de dónde sale el vocabulario en tiempo de ejecución. Mientras sea
  esta tabla copiada en los dos lados, se desincronizará; lo natural es que
  LiveWallpaper lo publique y Lyrixa lo lea, igual que Transcriptor publica
  hoy `layerRoles` en `/api/info`.
