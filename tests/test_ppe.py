"""PPE decisions on made-up detections, 3 checks per second for 8 seconds. Run: python3 -m tests.test_ppe
Track 1: hardhat + vest, one wrong No_Hardhat frame        -> nothing
Track 2: no hardhat every check, vest on                    -> one no_hardhat alert
Track 3: head never visible (back turned), vest on          -> nothing (unknown is not a violation)
A loose vest box on nobody                                  -> ignored"""
from vp.plugin import Frame, Obj
from vp.plugins.ppe import PPEPlugin

HARDHAT, NO_HARDHAT, VEST = 0, 1, 4
PEOPLE = {1: (0.10, 0.20, 0.20, 0.80), 2: (0.40, 0.20, 0.50, 0.80), 3: (0.70, 0.20, 0.80, 0.80)}


def person(tid):
    return Obj(tid, 0, 0.9, PEOPLE[tid], source=1)


def head(tid, cls):  # a box on the person's head (top 15% of their box)
    x1, y1, x2, y2 = PEOPLE[tid]
    return Obj(0, cls, 0.8, (x1 + 0.02, y1, x2 - 0.02, y1 + 0.09 * (y2 - y1) * 1.5), source=3)


def vest(tid):  # a box on the person's torso
    x1, y1, x2, y2 = PEOPLE[tid]
    return Obj(0, VEST, 0.8, (x1, y1 + 0.25 * (y2 - y1), x2, y1 + 0.6 * (y2 - y1)), source=3)


def run():
    plugin = PPEPlugin("test-cam", {})
    got = []
    for n in range(24):  # 3 checks per second, 8 seconds
        t = n / 3
        objs = [person(1), person(2), person(3)]
        objs += [head(1, NO_HARDHAT if n == 5 else HARDHAT), vest(1)]   # one wrong frame for track 1
        objs += [head(2, NO_HARDHAT), vest(2)]                          # track 2 never wears a hardhat
        objs += [vest(3)]                                               # track 3: no head box ever
        objs += [Obj(0, VEST, 0.8, (0.90, 0.30, 0.99, 0.50), source=3)]  # a vest on nobody
        for ev in plugin.on_frame(Frame("test-cam", n, t, objs)):
            got.append((ev["class"], ev["track_id"]))
            print(f"t={t:4.1f}s  {ev['class']}  track {ev['track_id']}  {ev['extra']}")
    assert got == [("no_hardhat", 2)], f"unexpected events: {got}"
    print("PASS: one alert for the person without a hardhat; one bad frame, unknown heads and loose vests ignored")


if __name__ == "__main__":
    run()
