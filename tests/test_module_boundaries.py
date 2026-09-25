"""La arquitectura del backend, comprobada en vez de documentada.

Este proyecto llegó a tener un `server.py` de 2500 líneas porque nada impedía
añadirle una más.  Estas pruebas son ese impedimento: describen en qué
dirección pueden apuntar las dependencias y qué módulos son hojas, de forma
que romper la separación falle aquí y no seis meses después.

La comprobación de nombres indefinidos existe por un caso real: al mover el
descargador a `media.py` se quedó atrás su `from collections import deque` y
el módulo importaba perfectamente — el fallo solo aparecía al descargar algo.
Un import limpio no demuestra que un módulo funcione.
"""

import ast
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEBAPP = ROOT / "webapp"
MODULES = sorted(p.stem for p in WEBAPP.glob("*.py"))

# Quién puede importar a quién.  Cada módulo lista los módulos *locales* que
# tiene permitido usar; lo que no está listado, no entra.
#
# La regla de fondo es que las flechas van hacia abajo: el servidor conoce el
# motor, el motor no conoce el servidor.  Es lo que permite llamar al pipeline
# desde una CLI, una prueba o una shell de escritorio sin levantar FastAPI.
ALLOWED = {
    # Hojas: no saben que existe el resto del backend.
    "paths": set(),
    "catalogs": set(),
    "config": set(),
    "jobhub": set(),
    "textjoin": set(),
    "audio_analysis": set(),
    "consensus": {"textjoin"},
    "track_guides": {"textjoin"},
    "lyrixa_bridge": {"textjoin", "consensus"},
    "translation": {"paths"},
    # Binarios externos: yt-dlp, ffprobe, los separadores.
    "media": {"paths", "catalogs", "audio_analysis"},
    # Un trabajo en disco.
    "jobstore": {"paths", "config", "media", "audio_analysis", "track_guides",
                 "lyrixa_bridge", "precision"},
    "transcribe_engines": {"paths", "consensus", "textjoin"},
    "precision": {"paths", "consensus", "textjoin", "transcribe_engines",
                  "audio_analysis", "translation"},
    # El lado servidor del motor: utilidades de ruta/trabajo y el CLI de
    # whisper. Nada del pipeline web entra aquí.
    "serveutil": {"paths"},
    "whispercli": {"paths", "serveutil"},
    # El trabajo largo: invoca el motor y publica eventos. No conoce FastAPI.
    "pipelines": {"audio_analysis", "catalogs", "jobhub", "jobstore",
                  "lyrixa_bridge", "media", "paths", "precision", "serveutil",
                  "track_guides", "translation", "whispercli"},
    # La fachada HTTP, partida por familia de rutas; cada una solo mira hacia
    # abajo. `routes_jobs` es la única que arranca pipelines.
    "routes_system": {"catalogs", "config", "lyrixa_bridge", "media", "paths",
                      "precision", "serveutil", "translation"},
    "routes_jobs": {"catalogs", "jobhub", "jobstore", "paths", "pipelines",
                    "track_guides", "translation"},
    "routes_serve": {"jobhub", "jobstore", "paths", "serveutil"},
    # El cableado: solo monta routers; el trabajo real ya no vive aquí.
    # `paths` para montar /static: la interfaz ya no es un único HTML.
    "server": {"config", "paths", "routes_system", "routes_jobs",
               "routes_serve"},
}


def local_imports(module: str) -> set[str]:
    tree = ast.parse((WEBAPP / f"{module}.py").read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found & set(MODULES)


class LayeringTests(unittest.TestCase):
    def test_every_module_is_declared(self):
        # Un módulo nuevo entra en la tabla de arriba deliberadamente, con una
        # decisión sobre a qué capa pertenece, y no por olvido.
        self.assertEqual(set(MODULES), set(ALLOWED))

    def test_nobody_imports_more_than_its_layer_allows(self):
        for module in MODULES:
            with self.subTest(module=module):
                extra = local_imports(module) - ALLOWED[module]
                self.assertEqual(extra, set(),
                                 f"{module} importa {sorted(extra)}, que está "
                                 f"por encima de su capa")

    def test_nothing_below_the_server_imports_the_server(self):
        # La regla que de verdad importa: si el motor importa al servidor, ya
        # no se puede usar sin él.
        for module in MODULES:
            if module == "server":
                continue
            with self.subTest(module=module):
                self.assertNotIn("server", local_imports(module))

    def test_the_dependency_graph_has_no_cycles(self):
        state: dict[str, int] = {}

        def visit(module: str, chain: list[str]) -> None:
            if state.get(module) == 2:
                return
            self.assertNotEqual(state.get(module), 1,
                                f"ciclo de imports: {' -> '.join(chain + [module])}")
            state[module] = 1
            for dependency in sorted(local_imports(module)):
                visit(dependency, chain + [module])
            state[module] = 2

        for module in MODULES:
            visit(module, [])


class UndefinedNameTests(unittest.TestCase):
    def test_no_module_uses_a_name_it_never_imported(self):
        try:
            import pyflakes  # noqa: F401
        except ImportError:
            self.skipTest("pyflakes no está instalado")
        result = subprocess.run(
            [sys.executable, "-m", "pyflakes", *[str(WEBAPP / f"{m}.py") for m in MODULES]],
            capture_output=True, text=True,
        )
        undefined = [line for line in result.stdout.splitlines()
                     if "undefined name" in line]
        self.assertEqual(undefined, [], "\n".join(undefined))


if __name__ == "__main__":
    unittest.main()
