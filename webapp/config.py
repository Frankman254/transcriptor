"""Dónde vive este servicio y quién tiene permiso para hablarle.

Transcriptor deja de ser "la web que abro en el 8000" para convertirse en un
motor local que otros procesos consultan.  Una futura shell de escritorio
arrancará el backend, Lyrixa y el renderer en puertos que elige ella, así que
ningún puerto ni URL puede quedar escrito en medio de un handler.

Todo tiene un valor por defecto que funciona con `./run.sh` a secas y una
variable de entorno con la que la shell lo sobreescribe.
"""

from __future__ import annotations

import os
import re
from urllib.parse import urlencode, urlsplit, urlunsplit


def _text(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def _port(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw.isdigit():
        return default
    value = int(raw)
    return value if 1 <= value <= 65535 else default


HOST = _text("TRANSCRIPTOR_HOST", "127.0.0.1")
PORT = _port("TRANSCRIPTOR_PORT", 8000)

# Cómo se ve este backend *desde fuera*.  No siempre coincide con HOST: la
# shell puede exponerlo detrás de otro nombre, y un cliente que reciba
# "0.0.0.0" no podría conectarse.
PUBLIC_URL = _text("TRANSCRIPTOR_PUBLIC_URL",
                   f"http://{'127.0.0.1' if HOST in {'', '0.0.0.0', '::'} else HOST}:{PORT}")

# Dónde vive Lyrixa.  Solo se usa para *ofrecer* un enlace "Abrir en Lyrixa";
# el backend nunca depende de que responda.
LYRIXA_URL = _text("TRANSCRIPTOR_LYRIXA_URL", "http://127.0.0.1:5173/")

# Quién puede llamar a la API desde un navegador.  El servidor escucha en
# loopback, así que el riesgo real no es la red: es una página remota que use
# el navegador del usuario como puente.  Por eso el permiso por defecto es
# cualquier puerto de localhost y ninguna otra cosa — https://ejemplo.com no
# entra ni con credenciales.
DEFAULT_ORIGIN_REGEX = r"^http://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$"
ALLOWED_ORIGIN_REGEX = _text("TRANSCRIPTOR_ALLOWED_ORIGIN_REGEX",
                             DEFAULT_ORIGIN_REGEX)
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("TRANSCRIPTOR_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]


def origin_allowed(origin: str) -> bool:
    """El mismo criterio que aplica el middleware, sin montar la app."""
    if origin in ALLOWED_ORIGINS:
        return True
    return bool(re.match(ALLOWED_ORIGIN_REGEX, origin))


def api_url(path: str) -> str:
    """URL absoluta de un recurso de esta API."""
    return f"{PUBLIC_URL.rstrip('/')}/{path.lstrip('/')}"


# Nombres de los parámetros del handshake, tal y como los lee Lyrixa hoy
# (`readRequestedJobId` y `resolveBridgeEndpoints` en src/core/bridge/
# bridgeConfig.ts).  Están aquí y no incrustados en la f-string porque el
# contrato es de Lyrixa, no nuestro: si allí se renombran, esto es lo único
# que cambia.
LYRIXA_JOB_PARAM = "job"
LYRIXA_API_PARAM = "transcriptor"


def lyrixa_open_url(job_id: str) -> str:
    """El enlace «Abrir en Lyrixa» de un trabajo.

    Lyrixa resuelve la dirección del backend por prioridad y la URL gana a
    todo lo demás, así que mandarla aquí es lo que hace que el enlace funcione
    aunque la shell haya movido el puerto o aunque Lyrixa esté empaquetado sin
    el proxy de Vite.  Sin este parámetro Lyrixa cae en su valor por defecto
    (`/transcriptor`, mismo origen) y el enlace solo funciona en desarrollo.
    """
    parts = urlsplit(LYRIXA_URL)
    query = urlencode({LYRIXA_JOB_PARAM: job_id,
                       LYRIXA_API_PARAM: PUBLIC_URL.rstrip("/")})
    separator = "&" if parts.query else ""
    return urlunsplit((parts.scheme, parts.netloc, parts.path or "/",
                       f"{parts.query}{separator}{query}", parts.fragment))


def as_dict() -> dict:
    """Lo que un cliente necesita saber para integrarse."""
    return {
        "publicUrl": PUBLIC_URL,
        "lyrixaUrl": LYRIXA_URL,
        "allowedOriginRegex": ALLOWED_ORIGIN_REGEX,
        "allowedOrigins": ALLOWED_ORIGINS,
    }
