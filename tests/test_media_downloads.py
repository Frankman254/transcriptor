"""Video-download selection without touching YouTube or the network."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webapp"))

import media  # noqa: E402


class VideoFormatTests(unittest.TestCase):
    def test_mp4_1080_limits_height_and_prefers_mp4_streams(self):
        args, spec = media.video_download_args("mp4", "1080")
        selector = args[args.index("-f") + 1]
        self.assertEqual(spec["value"], "mp4")
        self.assertIn("[ext=mp4][height<=1080]", selector)
        self.assertIn("ba[ext=m4a]", selector)
        self.assertEqual(args[args.index("--remux-video") + 1], "mp4")

    def test_maximum_quality_has_no_artificial_height_limit(self):
        args, _spec = media.video_download_args("mkv", "best")
        selector = args[args.index("-f") + 1]
        self.assertNotIn("height<=", selector)
        self.assertEqual(args[args.index("--merge-output-format") + 1], "mkv")

    def test_unknown_container_falls_back_to_mp4(self):
        _args, spec = media.video_download_args("avi", "720")
        self.assertEqual(spec["value"], "mp4")


class VideoDownloadTests(unittest.TestCase):
    def test_download_reports_the_final_video_and_progress(self):
        class FakeProcess:
            stdout = iter([
                "@PROG@50|100|NA|NA\n",
                "@PP@Merger\n",
                "@DONE@abc\tMi video\t42.5\t/tmp/Mi video.mp4\n",
            ])
            returncode = 0

            def wait(self):
                return 0

        events = []
        with tempfile.TemporaryDirectory() as folder, \
             mock.patch.object(media.subprocess, "Popen",
                               return_value=FakeProcess()) as popen:
            path, title, duration = media.download_video(
                "https://example.invalid/watch?v=abc", Path(folder),
                lambda kind, **data: events.append((kind, data)),
                "mp4", "1080",
            )
        command = popen.call_args.args[0]
        self.assertEqual(path, Path("/tmp/Mi video.mp4"))
        self.assertEqual((title, duration), ("Mi video", 42.5))
        self.assertIn("--no-playlist", command)
        self.assertIn(("progress", {"stage": "downloading", "progress": 1.0}),
                      events)
        self.assertTrue(any(data.get("message") == "Uniendo video y audio…"
                            for _kind, data in events))


if __name__ == "__main__":
    unittest.main()
