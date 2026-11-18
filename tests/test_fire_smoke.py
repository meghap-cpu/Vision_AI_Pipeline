"""Fire and smoke with a fake VLM answering from a script. Run: python3 -m tests.test_fire_smoke"""
from concurrent.futures import Future

import numpy as np

from vp.plugin import Frame
from vp.plugins.fire_smoke import FireSmokePlugin

NO, YES = {"fire": False, "smoke": False}, {"fire": True, "smoke": False}
SCRIPT = ([NO] * 4                     # quiet
          + [YES] * 3 + [None] + [YES] * 2  # 5 yes in a row (one unreadable reply ignored) -> start
          + [NO] * 3 + [YES]           # 3 no then a yes: not enough to end
          + ["error"]                  # VLM unreachable -> warning
          + [NO] * 10)                 # 10 no in a row -> end


class FakeVLM:
    def __init__(self, answers):
        self.answers, self.unreadable, self.asked = list(answers), 0, []

    def note_unreadable(self):
        self.unreadable += 1

    def ask(self, images, prompt):
        self.asked.append([int(im[0, 0, 0]) for im in images])
        f, a = Future(), self.answers.pop(0)
        if a == "error":
            f.set_exception(ConnectionError("VLM unreachable"))
        else:
            f.set_result({"answer": a, "raw": str(a), "seconds": 0.5})
        return f


def run():
    plugin = FireSmokePlugin("test-cam", {})
    plugin.client = FakeVLM(SCRIPT)
    img, got = np.zeros((288, 512, 3), np.uint8), []
    for n in range(4 * len(SCRIPT) + 2):  # one frame per second, 4 frames per batch
        for ev in plugin.on_frame(Frame("test-cam", n, float(n), [], pixels=img)):
            got.append(ev["class"])
            print(f"t={n:4d}s  {ev['class']}  {ev['extra']}")
    assert got == ["fire_smoke_start", "vlm_error", "fire_smoke_end"], f"unexpected events: {got}"
    assert plugin.client.unreadable == 1, "the unreadable reply is counted"
    print("ok  alert after 5 yes in a row, unreadable reply ignored and counted, VLM failure reported, end after 10 no")
    test_text_true_false()
    test_unclear_never_ends_a_fire()
    test_gap_starts_a_new_batch()
    print("PASS: streaks, strict true/false, unclear stays unknown, no batch across a gap")


def feed(plugin, seconds, start=0):
    """One picture per second; the picture's value is its second, so a test can see which ones were sent."""
    got = []
    for n in range(start, start + seconds):
        got += [e["class"] for e in plugin.on_frame(Frame("c", n, float(n), [], pixels=np.full((8, 8, 3), n % 256, np.uint8)))]
    return got


def test_text_true_false():
    p = FireSmokePlugin("c", {})
    p.client = FakeVLM([{"fire": "false", "smoke": "false"}] * 8)
    assert feed(p, 33) == [], "\"false\" as text is not fire (bool('false') would be True)"
    p = FireSmokePlugin("c", {})
    p.client = FakeVLM([{"fire": "true", "smoke": "no"}] * 6)
    assert feed(p, 25) == ["fire_smoke_start"], "\"true\" as text is fire"
    print("ok  text answers: \"false\" is no fire, \"true\" is fire")


def test_unclear_never_ends_a_fire():
    p = FireSmokePlugin("c", {})
    p.client = FakeVLM([YES] * 5 + [{"answer": "unclear"}] * 12)
    assert feed(p, 4 * 17 + 2) == ["fire_smoke_start"], "12 unclear replies must not end the fire"
    assert p.client.unreadable == 12
    print("ok  unclear: a reply without fire/smoke counts neither way and is counted; the fire stays open")


def test_gap_starts_a_new_batch():
    p = FireSmokePlugin("c", {})
    p.client = FakeVLM([NO] * 3)
    feed(p, 2)            # 2 pictures, then the camera drops out for a minute
    feed(p, 4, start=62)
    assert p.client.asked == [[62, 63, 64, 65]], f"only pictures from after the gap: {p.client.asked}"
    print("ok  gap: pictures from before a gap are not mixed into the next batch")


if __name__ == "__main__":
    run()
