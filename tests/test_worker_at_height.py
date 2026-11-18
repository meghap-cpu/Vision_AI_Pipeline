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
    """Answers in the order the plugin asks: most urgent first, so 3 (near the edge: 3 s) and then 1, 2, 4, 6, 7
    (10 s, in the order seen) on the first frame. 'slow' never answers."""
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
    plugin.client = FakeVLM([NOTHING, OK, NO_VEST, "slow", NO_HARNESS, OK])
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


class CountingVLM:
    """Records each question's crop size and when it was asked; answers at once with `answer`."""
    def __init__(self, answer):
        self.answer, self.asked, self.unreadable = answer, [], 0

    def ask(self, images, prompt, max_tokens=60):
        self.asked.append(images[0].shape[:2])
        f = Future()
        f.set_result({"answer": self.answer, "raw": str(self.answer), "seconds": 0.5})
        return f

    def note_unreadable(self):
        self.unreadable += 1


ZONE = {"zones": [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]]}
IMG = np.zeros((720, 1280, 3), np.uint8)


def test_asks_only_with_its_own_picture():
    p = WorkerAtHeightPlugin("c", ZONE)
    p.client = CountingVLM(OK)
    p.on_frame(Frame("c", 0, 0.0, [], pixels=IMG))  # a picture with nobody in it
    p.on_frame(Frame("c", 1, 0.2, [Obj(1, 0, 0.9, (0.4, 0.3, 0.5, 0.8))]))  # person, but no picture of this moment
    assert p.client.asked == [], "never crops a person out of an older picture"
    p.on_frame(Frame("c", 2, 0.4, [Obj(1, 0, 0.9, (0.4, 0.3, 0.5, 0.8))], pixels=IMG))
    assert len(p.client.asked) == 1, "asks on the next frame that has its own picture"
    print("ok  same moment: the crop is cut only from the picture of the frame the box came from")


def test_too_small_stays_unverified():
    p = WorkerAtHeightPlugin("c", ZONE)
    p.client, got = CountingVLM(NOTHING), []
    tiny = Obj(1, 0, 0.9, (0.50, 0.50, 0.51, 0.55))  # 36 px tall in a 720 px picture (limit 48)
    for n in range(60):
        got += p.on_frame(Frame("c", n, n / 5, [tiny], pixels=IMG))
    assert p.client.asked == [], "too small to judge: not asked, so no false 'no harness'"
    unv = [e for e in got if e["class"] == "at_height_unverified"]
    assert len(unv) == 1 and unv[0]["extra"]["why"] == "too_small", got
    print("ok  too small: never asked, one at_height_unverified with why=too_small (no false violation)")


def test_unreadable_answer_waits_before_asking_again():
    p = WorkerAtHeightPlugin("c", ZONE)
    p.client = CountingVLM({"notes": "cannot tell"})  # answered, but no harness/helmet/vest: unreadable
    for n in range(10):  # 2 s at 5 fps, a picture every frame
        p.on_frame(Frame("c", n, n / 5, [Obj(1, 0, 0.9, (0.4, 0.3, 0.5, 0.8))], pixels=IMG))
    assert len(p.client.asked) == 2, f"asked at 0 s and 1 s, not on every frame: {len(p.client.asked)}"
    assert p.client.unreadable == 2, "each unusable answer is counted"
    print(f"ok  unreadable: asked {len(p.client.asked)} times in 2 s (not 10), each counted as unreadable")


def test_no_zone_is_a_settings_error():
    import queue
    from vp import worker
    try:
        WorkerAtHeightPlugin("c", {"zones": []})
        raise AssertionError("no zone must be refused")
    except ValueError:
        pass
    events = queue.Queue()
    frames = queue.Queue()
    frames.put(None)
    worker.run("worker_at_height", {"cam9": {"rule": "Harness", "zones": []}}, frames, events)
    ev = events.get_nowait()
    assert ev["class"] == "plugin_error" and ev["camera_id"] == "cam9" and ev["extra"]["reason"] == "settings", ev
    print("ok  no zone: reported once as plugin_error (settings) instead of a camera that silently never alerts")


if __name__ == "__main__":
    run()
    test_asks_only_with_its_own_picture()
    test_too_small_stays_unverified()
    test_unreadable_answer_waits_before_asking_again()
    test_no_zone_is_a_settings_error()
    print("PASS: same-moment crops, too-small people stay unverified, unreadable answers wait, no zone is reported")
