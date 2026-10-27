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
        self.answers = list(answers)

    def ask(self, images, prompt):
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
    print("PASS: alert after 5 yes in a row, unreadable reply ignored, VLM failure reported, end after 10 no")


if __name__ == "__main__":
    run()
