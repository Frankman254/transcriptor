"""Trabajos en curso: colas de eventos SSE con reconexión sin pérdidas.

Un solo lugar para lo que antes estaba repartido por `server.py`: el diccionario
de trabajos vivos, el `Condition` que despierta a los lectores, y el buffer de
eventos numerados que permite al navegador retomar tras un corte (los `id:` que
`EventSource` reenvía como `Last-Event-ID`).

El buffer es finito (`MAX_EVENTS`): un trabajo de horas ya no puede crecer sin
tope en RAM. Al desbordar, los eventos más antiguos *se retiran* hacia un
`snapshot` acumulado (las letras dichas hasta ahí, los rounds de consenso, el
último estado de cada panel). Cuando un lector pide un punto que ya fue retirado,
el stream manda primero un evento `snapshot` y luego la cola conservada: la
página recargada a mitad de un trabajo largo recupera toda la letra vista, no
solo los últimos minutos.
"""

from __future__ import annotations

import json
import threading

# Máximo de eventos por trabajo en el buffer de reconexión. Por encima, los más
# antiguos se retiren al snapshot (seguían llegando al cliente, pero ya no se
# reenvían uno a uno).
MAX_EVENTS = 4000

# Máximo de elementos que el snapshot conserva por lista. Acota la RAM de un
# trabajo de muchas horas; lo más antiguo más allá de este tope se pierde.
MAX_SNAPSHOT = 2000

# Eventos que el snapshot acumula como lista (el panel se reconstruye fila a
# fila). Solo se guardan cuando el evento *sale* del buffer, de modo que snapshot
# + cola conservada cubren la historia exacta, sin duplicar.
_LIST_EVENTS = {"segment": "segments", "stem": "stems", "consensus": "rounds"}

# Eventos cuyo último valor retirado basta para reconstruir el panel. Son
# idempotentes al reaplicarlos, así que no importa si también siguen en cola.
_LATEST_EVENTS = {"status", "meta", "progress", "precision", "translation",
                  "reference", "rescue"}


class JobHub:
    """Registro de trabajos vivos con su cola de eventos."""

    def __init__(self) -> None:
        self._jobs: dict[str, dict] = {}

    def require(self, job_id: str) -> dict:
        """Como `get` pero explota si el trabajo no fue registrado: los
        pipelines solo corren después de `create`, así que un dict ausente aquí
        sería un bug de programación, no una condición de carrera."""
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(f"job {job_id} no está registrado")
        return job

    # ---------------------------------------------------------------- ciclo
    def create(self, job_id: str) -> dict:
        """Registra un trabajo nuevo y devuelve su dict (los pipelines guardan
        aquí sus claves privadas: `_timing`, `stem_paths`, …)."""
        job = {
            "events": [],       # cola de eventos aún reenviables
            "first_id": 0,      # id del más antiguo en `events`
            "next_id": 0,       # id del próximo evento
            "cond": threading.Condition(),
            "done": False,
            "snapshot": {"segments": [], "stems": [], "rounds": [],
                         "latest": {}},
        }
        self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> dict | None:
        return self._jobs.get(job_id)

    def pop(self, job_id: str) -> None:
        self._jobs.pop(job_id, None)

    def live(self, job_id: str) -> bool:
        """Hay un pipeline vivo para este trabajo en *este* proceso."""
        job = self._jobs.get(job_id)
        return bool(job and not job["done"])

    # -------------------------------------------------------------- eventos
    def publish(self, job: dict, event: str, data: dict) -> None:
        """Publica un evento: lo numera, lo encola y despierta a los lectores.

        Si la cola rebosa, los eventos retirados pasan al snapshot antes de
        descartarse, para que una reconexión antigua no pierda letra.
        """
        with job["cond"]:
            job["events"].append(
                {"seq": job["next_id"], "event": event, "data": data})
            job["next_id"] += 1
            overflow = len(job["events"]) - MAX_EVENTS
            for item in job["events"][:max(overflow, 0)]:
                self._retire(job, item)
            if overflow > 0:
                del job["events"][:overflow]
                job["first_id"] += overflow
            job["cond"].notify_all()

    @staticmethod
    def _retire(job: dict, item: dict) -> None:
        """Mueve un evento desalojado del buffer al snapshot acumulado."""
        event, data = item["event"], item["data"]
        bucket = _LIST_EVENTS.get(event)
        if bucket is not None:
            snap = job["snapshot"][bucket]
            snap.append(data)
            if len(snap) > MAX_SNAPSHOT:
                del snap[:len(snap) - MAX_SNAPSHOT]
        elif event in _LATEST_EVENTS:
            job["snapshot"]["latest"][event] = data

    def finish(self, job: dict) -> None:
        """Marca el fin del stream para todos los lectores pendientes."""
        with job["cond"]:
            job["done"] = True
            job["cond"].notify_all()

    # ---------------------------------------------------------------- stream
    def stream(self, job: dict, start: int):
        """Cuerpo SSE (text/event-stream), retomando desde el id `start`.

        Si `start` quedó por detrás del buffer, se manda antes un `snapshot`
        —sin `id:`, para no mover el Last-Event-ID del navegador— y la cola
        sigue desde lo más antiguo conservado. Así una recarga a mitad de un
        trabajo largo reconstruye los paneles completa y luego continúa normal.
        """
        cond: threading.Condition = job["cond"]
        with cond:
            first = job["first_id"]
        if start < first:
            with cond:
                payload = json.dumps(job["snapshot"])
            yield f"event: snapshot\ndata: {payload}\n\n"
            start = first

        i = start
        while True:
            with cond:
                if i >= job["next_id"] and not job["done"]:
                    cond.wait(timeout=15)  # espera evento nuevo o manda heartbeat
                first = job["first_id"]
                if i < first:      # lector más lento que el desalojo: sin lag
                    i = first
                batch = job["events"][i - first:]
                done = job["done"]
                nxt = job["next_id"]
            stale = False
            for item in batch:
                # Un vaciado lento puede quedar desalojado a mitad: ese evento
                # ya vive en el snapshot de futuras reconexiones; reemitirlo lo
                # duplicaría. Cortamos el lote y nos realineamos arriba.
                with cond:
                    if item["seq"] < job["first_id"]:
                        stale = True
                if stale:
                    break
                yield (f"id: {item['seq']}\n"
                       f"event: {item['event']}\n"
                       f"data: {json.dumps(item['data'])}\n\n")
                i += 1
            if not batch and not stale:
                yield ": keep-alive\n\n"  # mantiene viva la conexión TCP
            if done and i >= nxt:
                break


# El registro vivo es uno solo por proceso: rutas y pipelines lo comparten
# importándolo, sin que nadie tenga que crearlo ni pasárselo.
hub = JobHub()
