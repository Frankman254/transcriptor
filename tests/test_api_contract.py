"""The local-service contract: what Lyrixa and a desktop shell can rely on.

These exercise routing, validation and wiring — not Whisper, not Ollama, not
the separator.  Anything that would need a model is replaced, so the suite
stays runnable on a laptop with nothing installed.
"""

import json
import shutil
import sys
import tempfile
import unittest
from urllib.parse import parse_qs, urlsplit
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webapp"))

import config  # noqa: E402
import jobstore  # noqa: E402
import media  # noqa: E402
import paths  # noqa: E402
import routes_serve  # noqa: E402
import routes_system  # noqa: E402
import server  # noqa: E402
import translation  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


JOB_ID = "0123456789ab"


class ApiTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="transcriptor-test-"))
        self.outdir = self.tmp / JOB_ID
        self.outdir.mkdir()
        self._patches = [
            mock.patch.object(paths, "OUTPUT_DIR", self.tmp),
            # ffprobe is a subprocess and the fixtures are not real audio.
            mock.patch.object(media, "ffprobe_duration", lambda path: 180.0),
        ]
        for patch in self._patches:
            patch.start()
        self.addCleanup(self.stop)
        self.client = TestClient(server.app)

    def stop(self):
        for patch in reversed(self._patches):
            patch.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write_project(self, **extra):
        payload = {"schemaVersion": 1, "app": "Lyrixa",
                   "project": {"clips": []}, **extra}
        path = self.outdir / "cancion.lyrixa.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def write_audio(self, name="cancion.mp3", data=b"\xff\xfb" + b"0" * 4094):
        path = self.outdir / name
        path.write_bytes(data)
        return path


class HealthTests(ApiTestCase):
    def test_health_answers_without_touching_disk_or_models(self):
        body = self.client.get("/api/health").json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["service"], "transcriptor")
        self.assertGreaterEqual(body["bridgeSchemaVersion"], 3)
        self.assertTrue(body["publicUrl"].startswith("http://"))


class UiShellTests(ApiTestCase):
    def test_index_always_points_at_the_current_frontend_assets(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertNotIn("__SCRIBIX_ASSET_VERSION__", response.text)
        self.assertRegex(response.text, r"app\.js\?v=[0-9a-f]+")
        self.assertRegex(response.text, r"app\.css\?v=[0-9a-f]+")


class CorsTests(ApiTestCase):
    def test_lyrixa_on_its_own_port_is_allowed_to_call_the_api(self):
        response = self.client.get(
            "/api/health", headers={"Origin": "http://localhost:5173"})
        self.assertEqual(
            response.headers.get("access-control-allow-origin"),
            "http://localhost:5173",
        )

    def test_a_remote_page_is_not(self):
        response = self.client.get(
            "/api/health", headers={"Origin": "https://ejemplo.com"})
        self.assertEqual(response.status_code, 200)   # la respuesta existe…
        # …pero el navegador no se la entrega a esa página.
        self.assertIsNone(response.headers.get("access-control-allow-origin"))

    def test_the_preflight_of_a_post_succeeds_from_localhost(self):
        response = self.client.options(
            "/api/translate",
            headers={"Origin": "http://127.0.0.1:4173",
                     "Access-Control-Request-Method": "POST",
                     "Access-Control-Request-Headers": "content-type"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("POST",
                      response.headers.get("access-control-allow-methods", ""))

    def test_no_credentials_are_ever_allowed(self):
        # Sin cookies no hay nada que una página ajena pueda reutilizar.
        response = self.client.get(
            "/api/health", headers={"Origin": "http://localhost:5173"})
        self.assertIsNone(
            response.headers.get("access-control-allow-credentials"))


class LyrixaProjectTests(ApiTestCase):
    def test_the_project_is_served_by_job_id_not_by_file_name(self):
        self.write_project(marker="hola")
        response = self.client.get(f"/api/jobs/{JOB_ID}/lyrixa")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["marker"], "hola")
        self.assertTrue(
            response.headers["content-type"].startswith("application/json"))

    def test_a_job_without_a_project_says_so_instead_of_500(self):
        self.assertEqual(
            self.client.get(f"/api/jobs/{JOB_ID}/lyrixa").status_code, 404)

    def test_an_unknown_job_is_a_404(self):
        self.assertEqual(
            self.client.get("/api/jobs/ffffffffffff/lyrixa").status_code, 404)

    def test_a_malformed_job_id_cannot_escape_the_output_directory(self):
        (self.tmp / "secreto.lyrixa.json").write_text("{}", encoding="utf-8")
        for bad in ("..", "../..", "not-a-job", "0123456789AB"):
            with self.subTest(bad):
                response = self.client.get(f"/api/jobs/{bad}/lyrixa")
                self.assertIn(response.status_code, (404, 400))

    def test_download_asks_the_browser_to_save_it(self):
        self.write_project()
        response = self.client.get(f"/api/jobs/{JOB_ID}/lyrixa?download=1")
        self.assertIn("attachment", response.headers["content-disposition"])

    def test_the_newest_project_wins_after_a_re_run(self):
        import os
        import time
        old = self.outdir / "vieja.lyrixa.json"
        old.write_text(json.dumps({"marker": "vieja"}), encoding="utf-8")
        os.utime(old, (time.time() - 600, time.time() - 600))
        self.write_project(marker="nueva")
        body = self.client.get(f"/api/jobs/{JOB_ID}/lyrixa").json()
        self.assertEqual(body["marker"], "nueva")


class AudioTests(ApiTestCase):
    def test_lyrixa_can_fetch_the_source_audio_by_job_id(self):
        self.write_audio()
        response = self.client.get(f"/api/jobs/{JOB_ID}/audio")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/mpeg")

    def test_a_three_hour_mix_can_be_seeked_instead_of_downloaded_whole(self):
        self.write_audio(data=bytes(range(256)) * 16)
        response = self.client.get(f"/api/jobs/{JOB_ID}/audio",
                                   headers={"Range": "bytes=100-199"})
        self.assertEqual(response.status_code, 206)
        self.assertEqual(len(response.content), 100)

    def test_the_conditioned_asr_wav_is_not_mistaken_for_the_source(self):
        (self.outdir / "asr-input.wav").write_bytes(b"RIFF" + b"0" * 100)
        self.assertEqual(
            self.client.get(f"/api/jobs/{JOB_ID}/audio").status_code, 404)

    def test_a_job_with_no_audio_is_a_404(self):
        self.assertEqual(
            self.client.get(f"/api/jobs/{JOB_ID}/audio").status_code, 404)

    def test_each_container_declares_its_own_media_type(self):
        for name, expected in [("a.opus", "audio/opus"), ("a.m4a", "audio/mp4"),
                               ("a.flac", "audio/flac"), ("a.wav", "audio/wav")]:
            self.assertEqual(routes_serve.audio_media_type(Path(name)),
                             expected)


class JobLinkTests(ApiTestCase):
    def test_a_job_publishes_its_own_routes(self):
        self.write_project()
        self.write_audio()
        links = self.client.get(f"/api/jobs/{JOB_ID}").json()["links"]
        self.assertEqual(links["lyrixa"], f"/api/jobs/{JOB_ID}/lyrixa")
        self.assertEqual(links["audio"], f"/api/jobs/{JOB_ID}/audio")
        self.assertEqual(links["events"], f"/api/jobs/{JOB_ID}/events")

    def test_the_links_it_publishes_actually_resolve(self):
        self.write_project()
        self.write_audio()
        links = self.client.get(f"/api/jobs/{JOB_ID}").json()["links"]
        for key in ("self", "lyrixa", "audio"):
            with self.subTest(key):
                self.assertEqual(self.client.get(links[key]).status_code, 200)

    def test_a_job_without_a_project_does_not_advertise_one(self):
        self.write_audio()
        links = self.client.get(f"/api/jobs/{JOB_ID}").json()["links"]
        self.assertNotIn("lyrixa", links)
        self.assertIn("audio", links)

    def test_open_in_lyrixa_carries_the_job_and_the_api_base(self):
        self.write_project()
        self.write_audio()
        url = self.client.get(f"/api/jobs/{JOB_ID}").json()["links"]["openInLyrixa"]
        query = parse_qs(urlsplit(url).query)
        self.assertEqual(query["job"], [JOB_ID])
        self.assertEqual(query["transcriptor"], [config.PUBLIC_URL.rstrip("/")])
        self.assertTrue(url.startswith(config.LYRIXA_URL.split("?")[0]))

    def test_the_handshake_uses_the_parameter_names_lyrixa_reads(self):
        # `bridgeConfig.ts` looks up `job` and `transcriptor`; anything else
        # makes Lyrixa fall back to its dev proxy and the link only works
        # while Vite is running.  Pinned so a rename here cannot pass tests.
        self.assertEqual(config.LYRIXA_JOB_PARAM, "job")
        self.assertEqual(config.LYRIXA_API_PARAM, "transcriptor")


class TranslateEndpointTests(ApiTestCase):
    def fake_translate(self, lines, *, source, target, **kwargs):
        self.seen = {"lines": list(lines), "source": source, "target": target}
        return {"lines": [f"[{target}] {line}" for line in lines],
                "backend": "modelo-de-prueba"}

    def post(self, payload):
        with mock.patch.object(translation, "translate", self.fake_translate):
            return self.client.post("/api/translate", json=payload)

    def test_one_line_in_one_line_out(self):
        body = self.post({"text": "I wanna know what love is",
                          "sourceLanguage": "en",
                          "targetLanguage": "es"}).json()
        self.assertEqual(body["text"], "[es] I wanna know what love is")
        self.assertEqual(body["sourceLanguage"], "en")
        self.assertEqual(body["targetLanguage"], "es")
        self.assertEqual(body["backend"], "modelo-de-prueba")

    def test_a_whole_layer_comes_back_in_order(self):
        body = self.post({"lines": ["uno", "dos", "tres"],
                          "sourceLanguage": "es",
                          "targetLanguage": "en"}).json()
        self.assertEqual(body["lines"], ["[en] uno", "[en] dos", "[en] tres"])
        self.assertNotIn("text", body)

    def test_a_non_latin_source_is_routed_through_english(self):
        body = self.post({"text": "君の名は", "sourceLanguage": "ja",
                          "targetLanguage": "es"}).json()
        self.assertEqual(body["route"], [["ja", "en"], ["en", "es"]])
        self.assertIn("Inglés", body["routeLabel"])

    def test_text_already_in_the_target_language_comes_back_untouched(self):
        body = self.post({"text": "hola", "sourceLanguage": "es",
                          "targetLanguage": "es"}).json()
        self.assertEqual(body["text"], "hola")
        self.assertEqual(body["route"], [])

    def test_romaji_can_be_requested_on_its_own(self):
        with mock.patch.object(translation, "romanize",
                               lambda lines, **kw: ([f"romaji:{l}" for l in lines],
                                                    "pykakasi")):
            body = self.client.post("/api/translate",
                                    json={"text": "君の名は", "romaji": True}).json()
        self.assertEqual(body["romajiText"], "romaji:君の名は")
        self.assertEqual(body["romajiBackend"], "pykakasi")

    def test_an_empty_request_is_rejected_before_reaching_the_model(self):
        for payload in [{}, {"lines": []}, {"text": "hola"}]:
            with self.subTest(payload):
                self.assertEqual(
                    self.client.post("/api/translate", json=payload).status_code,
                    400)

    def test_an_unknown_target_language_is_rejected(self):
        response = self.client.post(
            "/api/translate", json={"text": "hola", "targetLanguage": "klingon"})
        self.assertEqual(response.status_code, 400)

    def test_a_runaway_request_is_capped(self):
        response = self.client.post("/api/translate", json={
            "lines": ["x"] * (routes_system.MAX_TRANSLATE_LINES + 1),
            "targetLanguage": "es"})
        self.assertEqual(response.status_code, 413)

    def test_no_local_translator_is_a_503_not_a_crash(self):
        def unavailable(*args, **kwargs):
            raise translation.TranslationUnavailable("Ollama no responde")

        with mock.patch.object(translation, "translate", unavailable):
            response = self.client.post(
                "/api/translate", json={"text": "hola", "targetLanguage": "en"})
        self.assertEqual(response.status_code, 503)
        self.assertIn("Ollama", response.json()["error"])


class ServiceDescriptionTests(ApiTestCase):
    def test_info_tells_a_client_how_to_integrate(self):
        with mock.patch.object(translation, "available_models", lambda: []):
            body = self.client.get("/api/info").json()
        service = body["service"]
        self.assertTrue(service["publicUrl"].startswith("http://"))
        self.assertIn("translation", service["layerRoles"])
        self.assertIn("transliteration", service["layerRoles"])
        self.assertIn("ja", service["romanizable"])
        self.assertGreaterEqual(service["bridgeSchemaVersion"], 3)


if __name__ == "__main__":
    unittest.main()
