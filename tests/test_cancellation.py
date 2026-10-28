import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webapp"))

from cancellation import CancellationToken, JobCancelled  # noqa: E402


class CancellationTokenTests(unittest.TestCase):
    def test_cancel_is_idempotent_and_check_raises_the_expected_signal(self):
        token = CancellationToken()
        self.assertTrue(token.cancel())
        self.assertFalse(token.cancel())
        with self.assertRaises(JobCancelled):
            token.check()

    def test_only_registered_processes_are_terminated(self):
        token = CancellationToken()
        process = mock.MagicMock(pid=4321)
        process.poll.return_value = None
        process.wait.return_value = 0
        with mock.patch("cancellation.os.killpg") as kill_group:
            token.register_process(process)
            token.cancel()
        kill_group.assert_called_once()

    def test_a_registered_python_stopper_runs_on_cancel(self):
        token = CancellationToken()
        stopper = mock.Mock()
        token.register_stopper(stopper)
        token.cancel()
        stopper.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
