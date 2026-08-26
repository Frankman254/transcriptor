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

import server  # noqa: E402


class JobRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.previous_jobs = server.JOBS.copy()
        server.JOBS.clear()
        self.temp = tempfile.TemporaryDirectory()
        self.output = Path(self.temp.name)
        self.output_patch = mock.patch.object(server, "OUTPUT_DIR", self.output)
        self.output_patch.start()

    def tearDown(self):
        self.output_patch.stop()
        self.temp.cleanup()
        server.JOBS.clear()
        server.JOBS.update(self.previous_jobs)

    def make_interrupted_job(self, job_id="a1b2c3d4e5f6"):
        outdir = self.output / job_id
        (outdir / "roformer").mkdir(parents=True)
        (outdir / "Mix recuperable.mp3").touch()
        (outdir / "roformer" / "vocals.wav").touch()
        (outdir / "roformer" / "instrumental.wav").touch()
        (outdir / "asr-input.wav").touch()
        return job_id, outdir

    def test_invalid_job_id_cannot_escape_the_output_directory(self):
        self.assertIsNone(server.recovery_assets("../../etc/passwd"))

    def test_asr_checkpoint_is_preferred_and_metadata_is_recovered(self):
        job_id, outdir = self.make_interrupted_job()
        with mock.patch.object(server, "ffprobe_duration", return_value=3588.1):
            assets = server.recovery_assets(job_id)
        self.assertEqual(assets["checkpoint"], (outdir / "asr-input.wav").resolve())
        self.assertEqual(assets["title"], "Mix recuperable")
        self.assertEqual(assets["duration"], 3588.1)

    def test_state_marks_a_disk_checkpoint_as_interrupted_and_recoverable(self):
        job_id, _ = self.make_interrupted_job()
        with mock.patch.object(server, "ffprobe_duration", return_value=10.0):
            response = server.job_state(job_id)
        self.assertEqual(response["status"], "interrupted")
        self.assertTrue(response["recoverable"])
        self.assertEqual(response["checkpoint"], "asr")
        self.assertIn("asr-input.wav", response["saved"])

    def test_resume_registers_the_same_job_without_starting_separation(self):
        job_id, _ = self.make_interrupted_job()
        with mock.patch.object(server.threading, "Thread") as thread:
            response = asyncio.run(server.resume_job(
                job_id, model="small", language="", precision_level="fast",
                translate_to="", romaji="1", tracklist_text="",
                fetch_lyrics="1",
            ))
        self.assertEqual(response, {"job_id": job_id, "resumed": True})
        self.assertIn(job_id, server.JOBS)
        self.assertFalse(server.JOBS[job_id]["done"])
        self.assertIsInstance(server.JOBS[job_id]["cond"], threading.Condition)
        thread.return_value.start.assert_called_once()

    def test_missing_job_state_is_a_404(self):
        response = server.job_state("000000000000")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(json.loads(response.body)["error"], "job no existe")

    def test_saved_job_exposes_results_and_reusable_stems(self):
        job_id, outdir = self.make_interrupted_job()
        (outdir / "Mix recuperable.json").write_text("{}", encoding="utf-8")
        with mock.patch.object(server, "ffprobe_duration", return_value=42.0):
            record = server.saved_job(job_id)
        self.assertEqual(record["status"], "completed")
        self.assertTrue(record["recoverable"])
        self.assertEqual(record["files"]["json"], "Mix recuperable.json")
        self.assertEqual(record["files"]["vocals"]["url"],
                         f"/api/jobs/{job_id}/stems/vocals")

    def test_iterating_archives_small_outputs_but_keeps_large_audio(self):
        job_id, outdir = self.make_interrupted_job()
        transcript = outdir / "Mix recuperable.json"
        transcript.write_text("{}", encoding="utf-8")
        iteration = server._archive_outputs(outdir)
        self.assertEqual(iteration, 1)
        self.assertFalse(transcript.exists())
        self.assertTrue((outdir / "iterations" / "iteration-01" /
                         transcript.name).is_file())
        self.assertTrue((outdir / "asr-input.wav").is_file())
        self.assertTrue((outdir / "Mix recuperable.mp3").is_file())


if __name__ == "__main__":
    unittest.main()
