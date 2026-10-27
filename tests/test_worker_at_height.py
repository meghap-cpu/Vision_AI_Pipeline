"""Worker at height with a fake VLM. Run: python3 -m tests.test_worker_at_height"""
from concurrent.futures import Future

import numpy as np

from vp.plugin import Frame, Obj
from vp.plugins.worker_at_height import WorkerAtHeightPlugin

OK = {"harness": True, "helmet": True}
NO_HARNESS = {"harness": False, "helmet": True}
NOTHING = {"harness": False, "helmet": False}
FEET = {1: (0.80, 0.5), 2: (0.70, 0.5), 3: (0.52, 0.5), 4: (0.90, 0.3), 5: (0.20, 0.8)}  # 5 is on the ground


class FakeVLM:
    """Answers in the order the plugin asks: people 1-4 on the first frame. 'slow' never answers."""
    def __init__(self, answers):
        self.answers = list(answers)

    def ask(self, images, prompt, max_tokens=60):
        f, a = Future(), self.answers.pop(0)
        if a != "slow":
            f.set_result({"answer": a, "raw": str(a), "seconds": 0.5})
        return f


def person(tid):
    x, y = FEET[tid]
    return Obj(tid, 0, 0.9, (x - 0.02, y - 0.2, x + 0.02, y), source=1)


def run():
    params = {"zones": [[[0.5, 0.0], [1.0, 0.0], [1.0, 0.6], [0.5, 0.6]]],  # height zone: upper right
              "edges": [[[0.5, 0.0], [0.5, 0.6]]]}                         # its open edge
    plugin = WorkerAtHeightPlugin("test-cam", params)
    plugin.client = FakeVLM([OK, NO_HARNESS, NOTHING, "slow"])
    img, got = np.zeros((720, 1280, 3), np.uint8), []
    for n in range(85):  # 5 fps: everyone present for 12 s, then nobody for 5 s
        t = n / 5
        people = [person(tid) for tid in FEET] if t < 12 else []
        for ev in plugin.on_frame(Frame("test-cam", n, t, people, pixels=img)):
            got.append((ev["class"], ev["track_id"]))
            x = ev["extra"]
            print(f"t={t:5.1f}s  {ev['class']:22s} track {ev['track_id']}  near_edge={x.get('near_edge')}  "
                  f"harness={x.get('harness')}  helmet={x.get('helmet')}")
    expected = [("at_height_critical", 3), ("at_height_end", 2), ("at_height_end", 3), ("at_height_end", 4),
                ("at_height_unverified", 4), ("at_height_violation", 2)]
    assert sorted(got) == expected, f"unexpected events: {sorted(got)}"
    print("PASS: violation, critical near the edge, unverified when the VLM is slow, ends on leaving; "
          "compliant and on-the-ground workers ignored")


if __name__ == "__main__":
    run()
