"""La arquitectura del backend, comprobada en vez de documentada.

Este proyecto llegó a tener un `server.py` de 2500 líneas porque nada impedía
añadirle una más.  Estas pruebas son ese impedimento: describen en qué
dirección pueden apuntar las dependencias y qué módulos son hojas, de forma
que romper la separación falle aquí y no seis meses después.

La tabla de capas dice quién puede hablar con quién; `__all__` dice *qué* se
le puede decir.  Hicieron falta las dos: hubo un `routes_system` que llegaba a
`transcribe_engines` escribiendo `precision.te.…`, sin un solo import que
delatara la dependencia, y la regla de capas lo daba por bueno.

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
    "cancellation": set(),
    "paths": set(),
    "catalogs": set(),
    "config": set(),
    "jobhub": {"cancellation"},
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


def declared_interface(module: str) -> set[str] | None:
    """El `__all__` de un módulo, o None si no declara ninguno."""
    tree = ast.parse((WEBAPP / f"{module}.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "__all__"
                        for t in node.targets)):
            return {element.value for element in node.value.elts}
    return None


def crossings(module: str) -> dict[str, set[str]]:
    """Qué nombre de qué otro módulo local toca este módulo.

    Mira las dos formas de tocarlo, no solo la primera: `from N import x` y
    `N.x` sobre el nombre con el que se importó `N` (resolviendo `import N as
    n`).  Leer solo los `import` fue el agujero real de esta prueba: un
    `routes_system` que llamaba a `precision.te.mlx_whisper_available()`
    dependía de `transcribe_engines` sin que apareciera ni un import, y la
    regla de capas daba verde.
    """
    tree = ast.parse((WEBAPP / f"{module}.py").read_text(encoding="utf-8"))
    touched: dict[str, set[str]] = {}
    alias: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for element in node.names:
                if element.name in MODULES and element.name != module:
                    alias[element.asname or element.name] = element.name
        elif (isinstance(node, ast.ImportFrom) and node.level == 0
                and node.module in MODULES and node.module != module):
            touched.setdefault(node.module, set()).update(
                element.name for element in node.names)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in alias):
            touched.setdefault(alias[node.value.id], set()).add(node.attr)
    return touched


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


class InterfaceTests(unittest.TestCase):
    """Lo que un módulo promete y lo que de verdad le piden.

    Un módulo con veinte nombres públicos de los que fuera solo se llaman seis
    no tiene una interfaz de veinte: tiene una de seis y catorce detalles que
    cualquiera puede empezar a usar sin darse cuenta.  `__all__` es la que
    promete; estas pruebas comprueban que nadie pida nada más.
    """

    def test_every_module_declares_its_interface(self):
        # Igual que la tabla de capas: un módulo nuevo decide qué expone, y lo
        # decide alguien, no el azar de qué quedó sin guión bajo delante.
        for module in MODULES:
            with self.subTest(module=module):
                self.assertIsNotNone(
                    declared_interface(module),
                    f"{module} no declara `__all__`: no se sabe qué parte de "
                    f"él es interfaz y qué parte es implementación")

    def test_nobody_reaches_into_another_module_private(self):
        for module in MODULES:
            for target, names in sorted(crossings(module).items()):
                private = {name for name in names if name.startswith("_")}
                with self.subTest(module=module, target=target):
                    self.assertEqual(
                        private, set(),
                        f"{module} usa {sorted(private)} de {target}: es la "
                        f"implementación de {target}, no su interfaz")

    def test_nobody_reaches_past_a_declared_interface(self):
        for module in MODULES:
            for target, names in sorted(crossings(module).items()):
                interface = declared_interface(target)
                if interface is None:
                    continue
                with self.subTest(module=module, target=target):
                    extra = names - interface
                    self.assertEqual(
                        extra, set(),
                        f"{module} usa {sorted(extra)} de {target}, que no "
                        f"está en su `__all__`: o es interfaz y se declara, o "
                        f"no se usa desde fuera")


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
