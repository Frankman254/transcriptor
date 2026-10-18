"""Cancelación cooperativa de un trabajo largo y sus procesos hijos.

El token no sabe nada de FastAPI ni del almacén de trabajos. El pipeline lo
comparte con los módulos que lanzan binarios y el servidor se limita a pedir
la cancelación. Cada token pertenece a un solo trabajo: nunca se mata Uvicorn
ni un proceso registrado por otro trabajo.
"""

from __future__ import annotations

__all__ = ["CancellationToken", "JobCancelled"]

import os
import signal
import threading
from collections.abc import Callable
from typing import Any


class JobCancelled(Exception):
    """Salida deliberada solicitada por el usuario, no un fallo del motor."""


class CancellationToken:
    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._processes: set[Any] = set()
        self._stoppers: set[Callable[[], None]] = set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def check(self) -> None:
        if self.cancelled:
            raise JobCancelled("Trabajo cancelado por el usuario.")

    def register_process(self, process: Any) -> None:
        """Registra un hijo creado con ``start_new_session=True``."""
        with self._lock:
            self._processes.add(process)
            cancelled = self.cancelled
        if cancelled:
            self._terminate(process)

    def unregister_process(self, process: Any) -> None:
        with self._lock:
            self._processes.discard(process)

    def register_stopper(self, stopper: Callable[[], None]) -> None:
        """Registra cómo interrumpir trabajo Python que no es un Popen."""
        with self._lock:
            self._stoppers.add(stopper)
            cancelled = self.cancelled
        if cancelled:
            self._call_stopper(stopper)

    def unregister_stopper(self, stopper: Callable[[], None]) -> None:
        with self._lock:
            self._stoppers.discard(stopper)

    def cancel(self) -> bool:
        """Solicita la cancelación una vez y despierta todo trabajo registrado."""
        with self._lock:
            first = not self.cancelled
            self._event.set()
            processes = list(self._processes)
            stoppers = list(self._stoppers)
        for process in processes:
            self._terminate(process)
        for stopper in stoppers:
            self._call_stopper(stopper)
        return first

    @staticmethod
    def _call_stopper(stopper: Callable[[], None]) -> None:
        try:
            stopper()
        except Exception:
            pass

    @staticmethod
    def _terminate(process: Any) -> None:
        try:
            if process.poll() is not None:
                return
        except (AttributeError, OSError):
            pass
        try:
            # Todos los Popen registrados nacen en sesión propia. Terminar el
            # grupo incluye ffmpeg u otros nietos que el CLI haya lanzado.
            os.killpg(process.pid, signal.SIGTERM)
        except (AttributeError, OSError, ProcessLookupError):
            try:
                process.terminate()
            except Exception:
                return

        def force_after_grace() -> None:
            try:
                process.wait(timeout=3)
                return
            except Exception:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (AttributeError, OSError, ProcessLookupError):
                try:
                    process.kill()
                except Exception:
                    pass

        threading.Thread(target=force_after_grace, daemon=True).start()
