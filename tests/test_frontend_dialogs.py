import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "webapp" / "static"


class FrontendDialogTests(unittest.TestCase):
    def test_native_browser_dialogs_are_not_used(self):
        javascript = (STATIC / "app.js").read_text(encoding="utf-8")

        self.assertIsNone(
            re.search(r"\b(?:alert|confirm|prompt)\s*\(", javascript),
            "Usa el componente showAppDialog en vez de un diálogo nativo.",
        )

    def test_reusable_dialog_is_present_once(self):
        html = (STATIC / "index.html").read_text(encoding="utf-8")

        self.assertEqual(html.count('id="app-dialog"'), 1)
        self.assertIn('aria-labelledby="app-dialog-title"', html)
        self.assertIn('aria-describedby="app-dialog-message"', html)

    def test_frontend_assets_are_cache_busted(self):
        html = (STATIC / "index.html").read_text(encoding="utf-8")

        self.assertIn("app.css?v=__SCRIBIX_ASSET_VERSION__", html)
        self.assertIn("app.js?v=__SCRIBIX_ASSET_VERSION__", html)

    def test_dialog_has_all_visual_variants(self):
        css = (STATIC / "app.css").read_text(encoding="utf-8")

        self.assertIn(".app-dialog.is-warning", css)
        self.assertIn(".app-dialog.is-danger", css)


if __name__ == "__main__":
    unittest.main()
