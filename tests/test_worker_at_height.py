"""Worker at height with a fake VLM. Run: python3 -m tests.test_worker_at_height"""
from concurrent.futures import Future

import numpy as np

from vp.plugin import Frame, Obj
from vp.plugins.worker_at_height import WorkerAtHeightPlugin, parse_ppe

OK = {"harness": True, "helmet": True, "vest": True}
NO_VEST = {"harness": True, "helmet": True, "vest": False}
NO_HARNESS = {"harness": False, "helmet": True, "vest": True}
NOTHING = {"harness": False, "helmet": False, "vest": False}
# 1 ok, 2 no vest, 3 nothing near the edge, 4 slow VLM, 5 on the ground, 6 no harness, 7 ok but only walks through
FEET = {1: (0.80, 0.5), 2: (0.70, 0.5), 3: (0.52, 0.5), 4: (0.90, 0.3), 5: (0.20, 0.8), 6: (0.85, 0.4), 7: (0.75, 0.2)}


class FakeVLM:
    """Answers in the order the plugin asks: people 1, 2, 3, 4, 6, 7 on the first frame. 'slow' never answers."""
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
    assert parse_ppe({"harness": True, "helmet": True})["violation"], "vest not answered must count as missing"
    assert parse_ppe({"notes": "unclear"}) is None, "no item answered must stay unverified"
    params = {"zones": [[[0.5, 0.0], [1.0, 0.0], [1.0, 0.6], [0.5, 0.6]]],  # height zone: upper right
              "edges": [[[0.5, 0.0], [0.5, 0.6]]]}                         # its open edge
    plugin = WorkerAtHeightPlugin("test-cam", params)
    plugin.client = FakeVLM([OK, NO_VEST, NOTHING, "slow", NO_HARNESS, OK])
    img, got = np.zeros((720, 1280, 3), np.uint8), []
    for n in range(85):  # 5 fps: everyone present for 12 s, then nobody for 5 s
        t = n / 5
        people = [person(tid) for tid in FEET if t < (3 if tid == 7 else 12)]
        for ev in plugin.on_frame(Frame("test-cam", n, t, people, pixels=img)):
            got.append((ev["class"], ev["track_id"], ev["severity"]))
            x = ev["extra"]
            print(f"t={t:5.1f}s  {ev['class']:22s} {ev['severity']:8s} track {ev['track_id']}  "
                  f"near_edge={x.get('near_edge')}  harness={x.get('harness')}  helmet={x.get('helmet')}  vest={x.get('vest')}")
    expected = sorted([
        ("at_height_entered", 1, "warning"), ("at_height_end", 1, "info"),     # compliant: low after 5 s, closes
        ("at_height_entered", 4, "warning"),                                   # low after 5 s, before it escalates
        ("at_height_violation", 2, "critical"), ("at_height_left", 2, "info"),   # no vest: high, stays open
        ("at_height_critical", 3, "critical"), ("at_height_left", 3, "info"),    # nothing, near the edge
        ("at_height_unverified", 4, "critical"), ("at_height_left", 4, "info"),  # VLM never answered
        ("at_height_violation", 6, "critical"), ("at_height_left", 6, "info")])  # no harness
    assert sorted(got) == expected, f"unexpected events: {sorted(got)}"
    print("PASS: low after 5 s in the zone; high for any missing item (vest too), near the edge, or no verdict; "
          "high incidents stay open, compliant ones close; walk-throughs and workers on the ground raise nothing")


if __name__ == "__main__":
    run()
