"""PPE on made-up detections, 3 checks per second for 22 s. Run: python3 -m tests.test_ppe
Track 1: hardhat on, one wrong frame                     -> nothing
Track 2: no hardhat all along                            -> alert at ~11 s (1 s to establish + 10 s), again at ~19 s
Track 3: head never visible                              -> nothing
Track 4: no hardhat but outside the zone                 -> nothing
Track 5: "no hardhat" only at confidence 0.3 (< 0.5)     -> nothing"""
from vp.plugin import Frame, Obj
from vp.plugins.ppe import PPEPlugin

HARDHAT, NO_HARDHAT, VEST = 0, 1, 4
PEOPLE = {1: (0.10, 0.20, 0.20, 0.80), 2: (0.40, 0.20, 0.50, 0.80), 3: (0.70, 0.20, 0.80, 0.80),
          4: (0.88, 0.20, 0.96, 0.80), 5: (0.25, 0.20, 0.33, 0.80)}


def person(tid):
    return Obj(tid, 0, 0.9, PEOPLE[tid], source=1)


def head(tid, cls, conf=0.8):
    x1, y1, x2, y2 = PEOPLE[tid]
    return Obj(0, cls, conf, (x1 + 0.01, y1, x2 - 0.01, y1 + 0.13 * (y2 - y1)), source=3)


def run():
    params = {"duration_s": 10, "cooldown_s": 8, "min_confidence": 0.5,
              "zones": [[[0.0, 0.0], [0.85, 0.0], [0.85, 1.0], [0.0, 1.0]]]}  # track 4 stands outside it
    plugin = PPEPlugin("test-cam", params)
    got = []
    for n in range(66):
        t = n / 3
        objs = [person(i) for i in PEOPLE]
        objs += [head(1, NO_HARDHAT if n == 5 else HARDHAT), head(2, NO_HARDHAT, 0.85),
                 head(4, NO_HARDHAT), head(5, NO_HARDHAT, 0.3)]
        for ev in plugin.on_frame(Frame("test-cam", n, t, objs)):
            got.append((ev["class"], ev["track_id"], round(t, 1)))
            print(f"t={t:4.1f}s  {ev['class']}  track {ev['track_id']}  conf {ev['confidence']}  {ev['extra']}")
    assert [(c, tr) for c, tr, _ in got] == [("no_hardhat", 2), ("no_hardhat", 2)], f"unexpected: {got}"
    assert 10.5 <= got[0][2] <= 11.7 and 18.5 <= got[1][2] <= 19.9, f"timing: {got}"
    print("PASS: alert after 10 s without a hardhat and again after the cooldown; one bad frame, unknown heads, "
          "people outside the zone and low-confidence detections ignored")


if __name__ == "__main__":
    run()
