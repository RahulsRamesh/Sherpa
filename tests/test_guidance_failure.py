import unittest
from unittest import mock

from sherpa import guidance
from sherpa.guidance import Step


class FakeClicks:
    def __init__(self, clicks=()):
        self.queue = list(clicks)

    def start(self): pass
    def stop(self): pass
    def clear(self): pass

    def wait(self, timeout):
        return self.queue.pop(0) if self.queue else None


class Loc:
    def __init__(self, x, y):
        self.x, self.y, self.radius, self.label, self.how = x, y, 10, "", "fake"


class FailureTests(unittest.TestCase):
    def run_guide(self, steps, finds, clicks):
        """finds: target -> list of results consumed in order (last one repeats)."""
        spoken, moves = [], []

        def fake_find(step, task):
            seq = finds[step.target]
            return seq.pop(0) if len(seq) > 1 else seq[0]

        fake_guide = mock.Mock()
        fake_guide.move_to.side_effect = lambda x, y, label="": moves.append((x, y))
        with mock.patch.object(guidance, "_find", fake_find), \
             mock.patch.object(guidance, "guide", fake_guide), \
             mock.patch.object(guidance, "get_cursor_position", lambda: (0, 0)), \
             mock.patch.object(guidance, "_Prefetch", mock.Mock()), \
             mock.patch.object(guidance.time, "sleep", lambda s: None):
            guidance.run(steps, "task", lambda: True, spoken.append, clicks=FakeClicks(clicks))
        return spoken, moves

    def steps(self, n):
        return [Step(f"the thing {k}", f"say {k}", f"t{k}", "after_speech") for k in range(n)]

    def test_failure_at_step0(self):
        spoken, moves = self.run_guide(self.steps(2), {"the thing 0": [None], "the thing 1": [Loc(5, 5)]}, [])
        self.assertEqual(moves, [])
        self.assertEqual(spoken, ["I couldn't find the thing 0, but if you can find it for me, give me a shout and I'll show you from there."])

    def test_failure_at_step2_recovers(self):
        finds = {
            "the thing 0": [Loc(1, 1)],
            "the thing 1": [Loc(2, 2), Loc(3, 3)],  # second lookup is the re-point
            "the thing 2": [None, Loc(9, 9)],
        }
        spoken, moves = self.run_guide(self.steps(3), finds, [(0, 0)])
        self.assertEqual(moves, [(1, 1), (2, 2), (3, 3), (9, 9)])
        self.assertIn("Hmm, I don't see the thing 2. Can you go back to the thing 1 and we'll take another look?", spoken)
        self.assertEqual(spoken[-1], "say 2")

    def test_double_failure_ends(self):
        finds = {"the thing 0": [Loc(1, 1)], "the thing 1": [Loc(2, 2)], "the thing 2": [None]}
        spoken, moves = self.run_guide(self.steps(3), finds, [(0, 0)])
        self.assertTrue(spoken[-1].startswith("I couldn't find the thing 2, but"))
        self.assertNotIn("say 2", spoken)

    def test_natural(self):
        self.assertEqual(guidance._natural("The File menu."), "the File menu")


if __name__ == "__main__":
    unittest.main()
