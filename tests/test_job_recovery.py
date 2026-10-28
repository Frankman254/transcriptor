import asyncio
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webapp"))

import jobstore  # noqa: E402
import media  # noqa: E402
import paths  # noqa: E402
import routes_jobs  # noqa: E402


class JobRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.previous_jobs = {j: routes_jobs.hub.get(j)
                              for j in routes_jobs.hub._jobs}
        routes_jobs.hub._jobs.clear()
        self.temp = tempfile.TemporaryDirectory()
        self.output = Path(self.temp.name)
        self.output_patch = mock.patch.object(paths, "OUTPUT_DIR", self.output)
        self.output_patch.start()

    def tearDown(self):
        self.output_patch.stop()
        self.temp.cleanup()
        routes_jobs.hub._jobs.clear()
        routes_jobs.hub._jobs.update(
            {k: v for k, v in self.previous_jobs.items() if v is not None})

    def make_interrupted_job(self, job_id="a1b2c3d4e5f6"):
        outdir = self.output / job_id
        (outdir / "roformer").mkdir(parents=True)
        (outdir / "Mix recuperable.mp3").touch()
        (outdir / "roformer" / "vocals.wav").touch()
        (outdir / "roformer" / "instrumental.wav").touch()
        (outdir / "asr-input.wav").touch()
        return job_id, outdir

    def test_invalid_job_id_cannot_escape_the_output_directory(self):
        self.assertIsNone(jobstore.recovery_assets("../../etc/passwd"))

    def test_asr_checkpoint_is_preferred_and_metadata_is_recovered(self):
        job_id, outdir = self.make_interrupted_job()
        with mock.patch.object(media, "ffprobe_duration", return_value=3588.1):
            assets = jobstore.recovery_assets(job_id)
        self.assertEqual(assets["checkpoint"], (outdir / "asr-input.wav").resolve())
        self.assertEqual(assets["title"], "Mix recuperable")
        self.assertEqual(assets["duration"], 3588.1)

    def test_state_marks_a_disk_checkpoint_as_interrupted_and_recoverable(self):
        job_id, _ = self.make_interrupted_job()
        with mock.patch.object(media, "ffprobe_duration", return_value=10.0):
            response = routes_jobs.job_state(job_id)
        self.assertEqual(response["status"], "interrupted")
        self.assertTrue(response["recoverable"])
        self.assertEqual(response["checkpoint"], "asr")
        self.assertIn("asr-input.wav", response["saved"])

    def test_resume_registers_the_same_job_without_starting_separation(self):
        job_id, _ = self.make_interrupted_job()
        with mock.patch.object(routes_jobs.threading, "Thread") as thread, \
             mock.patch.object(jobstore, "reusable_checkpoint",
                               return_value=(Path("asr-input.wav"), "asr")):
            response = asyncio.run(routes_jobs.resume_job(
                job_id, model="small", language="", precision_level="fast",
                translate_to="", romaji="1", tracklist_text="",
                fetch_lyrics="1",
            ))
        self.assertEqual(response, {"job_id": job_id, "resumed": True})
        self.assertTrue(routes_jobs.hub.live(job_id))
        self.assertIsInstance(routes_jobs.hub.get(job_id)["cond"],
                              threading.Condition)
        thread.return_value.start.assert_called_once()

    def test_checkpoint_prefers_complete_asr_and_falls_back_to_vocals(self):
        _, outdir = self.make_interrupted_job()
        asr = outdir / "asr-input.wav"
        vocals = outdir / "roformer" / "vocals.wav"
        asr.write_bytes(b"0" * 100)
        vocals.write_bytes(b"0" * 100)
        assets = {"duration": 100.0, "asr": asr, "vocals": vocals}
        with mock.patch.object(media, "ffprobe_duration",
                               side_effect=lambda path: 80.0 if path == asr else 100.0):
            checkpoint, kind = jobstore.reusable_checkpoint(assets)
        self.assertEqual(checkpoint, vocals)
        self.assertEqual(kind, "vocals")

    def test_invalid_checkpoint_is_rejected_before_archiving_outputs(self):
        job_id, outdir = self.make_interrupted_job()
        transcript = outdir / "Mix recuperable.json"
        transcript.write_text("{}", encoding="utf-8")
        with mock.patch.object(jobstore, "reusable_checkpoint",
                               return_value=(None, None)):
            response = asyncio.run(routes_jobs.resume_job(job_id))
        self.assertEqual(response.status_code, 409)
        self.assertTrue(transcript.is_file())
        self.assertFalse((outdir / "iterations").exists())

    def test_delete_removes_only_the_requested_inactive_job(self):
        job_id, outdir = self.make_interrupted_job()
        other_id, other = self.make_interrupted_job("b1c2d3e4f5a6")
        response = routes_jobs.delete_saved_job(job_id)
        self.assertEqual(response, {"deleted": True, "job_id": job_id})
        self.assertFalse(outdir.exists())
        self.assertTrue(other.exists())

    def test_delete_refuses_a_running_job(self):
        job_id, outdir = self.make_interrupted_job()
        routes_jobs.hub.create(job_id)
        response = routes_jobs.delete_saved_job(job_id)
        self.assertEqual(response.status_code, 409)
        self.assertTrue(outdir.exists())

    def test_cancel_signals_only_the_requested_running_job(self):
        first_id, _ = self.make_interrupted_job()
        second_id, _ = self.make_interrupted_job("b1c2d3e4f5a6")
        first = routes_jobs.hub.create(first_id)
        second = routes_jobs.hub.create(second_id)
        response = routes_jobs.cancel_job(first_id)
        self.assertEqual(response, {"job_id": first_id, "cancelling": True})
        self.assertTrue(first["cancel"].cancelled)
        self.assertFalse(second["cancel"].cancelled)
        self.assertEqual(first["events"][-1]["event"], "status")

    def test_cancelled_status_wins_over_an_older_result_file(self):
        job_id, outdir = self.make_interrupted_job()
        (outdir / "resultado.json").write_text("{}", encoding="utf-8")
        jobstore.update_manifest(job_id, status="cancelled")
        with mock.patch.object(media, "ffprobe_duration", return_value=42.0):
            record = jobstore.saved_job(job_id)
        self.assertEqual(record["status"], "cancelled")

    def test_stale_running_manifest_is_listed_as_interrupted(self):
        job_id, outdir = self.make_interrupted_job()
        (outdir / "asr-input.wav").unlink()
        (outdir / "roformer" / "vocals.wav").unlink()
        jobstore.update_manifest(job_id, status="running",
                                 title="Proceso que ya no existe")

        with mock.patch.object(media, "ffprobe_duration", return_value=42.0):
            stale = jobstore.saved_job(job_id, running=False)
            live = jobstore.saved_job(job_id, running=True)

        self.assertEqual(stale["status"], "interrupted")
        self.assertFalse(stale["recoverable"])
        self.assertEqual(live["status"], "running")

    def test_missing_job_state_is_a_404(self):
        response = routes_jobs.job_state("000000000000")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(json.loads(response.body)["error"], "job no existe")

    def test_saved_job_exposes_results_and_reusable_stems(self):
        job_id, outdir = self.make_interrupted_job()
        (outdir / "Mix recuperable.json").write_text("{}", encoding="utf-8")
        with mock.patch.object(media, "ffprobe_duration", return_value=42.0):
            record = jobstore.saved_job(job_id)
        self.assertEqual(record["status"], "completed")
        self.assertTrue(record["recoverable"])
        self.assertEqual(record["files"]["json"], "Mix recuperable.json")
        self.assertEqual(record["files"]["vocals"]["url"],
                         f"/api/jobs/{job_id}/stems/vocals")

    def test_iterating_archives_small_outputs_but_keeps_large_audio(self):
        job_id, outdir = self.make_interrupted_job()
        jobstore.update_manifest(job_id, status="completed",
                               settings={"model": "turbo",
                                         "asrEngine": "mlx-whisper"})
        transcript = outdir / "Mix recuperable.json"
        transcript.write_text("{}", encoding="utf-8")
        iteration = jobstore.archive_outputs(outdir)
        self.assertEqual(iteration, 1)
        self.assertFalse(transcript.exists())
        self.assertTrue((outdir / "iterations" / "iteration-01" /
                         transcript.name).is_file())
        frozen = json.loads((outdir / "iterations" / "iteration-01" /
                             "run.json").read_text(encoding="utf-8"))
        self.assertEqual(frozen["settings"]["model"], "turbo")
        self.assertTrue((outdir / "asr-input.wav").is_file())
        self.assertTrue((outdir / "Mix recuperable.mp3").is_file())


if __name__ == "__main__":
    unittest.main()
