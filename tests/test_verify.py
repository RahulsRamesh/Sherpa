import unittest
from types import SimpleNamespace
from unittest import mock

from PIL import Image

from sherpa import verify

STEP = SimpleNamespace(say="Type a name.", done_when="the name field has text")
FOUND = SimpleNamespace(x=100, y=100)


def _img(color):
    return Image.new("RGB", (400, 300), color)


def run(images, ask, still_wanted=lambda: True, timeout=0.5):
    it = iter(images)
    last = [images[-1]]

    def grab(x, y):
        try:
            last[0] = next(it)
        except StopIteration:
            pass
        return last[0]

    with mock.patch.object(verify, "_grab", grab), mock.patch.object(verify, "_ask", ask):
        return verify.wait_until_done(STEP, FOUND, "task", still_wanted, timeout_s=timeout, interval_s=0.05)


class VerifyTests(unittest.TestCase):
    def test_unchanged_region_asks_once_then_times_out(self):
        ask = mock.Mock(return_value=False)
        self.assertEqual(run([_img("white")], ask), "timeout")
        self.assertEqual(ask.call_count, 1)  # only the baseline look

    def test_changed_region_done(self):
        ask = mock.Mock(side_effect=[False, True])
        self.assertEqual(run([_img("white"), _img("black")], ask), "done")
        self.assertEqual(ask.call_count, 2)

    def test_model_error_keeps_polling(self):
        ask = mock.Mock(side_effect=[RuntimeError("boom"), True])
        # error on the first poll; second poll sees a changed region vs. the stored thumb
        self.assertEqual(run([_img("white"), _img("black")], ask), "done")

    def test_not_wanted_interrupts(self):
        ask = mock.Mock()
        self.assertEqual(run([_img("white")], ask, still_wanted=lambda: False), "interrupted")
        ask.assert_not_called()


if __name__ == "__main__":
    unittest.main()
