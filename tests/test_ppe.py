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
    print("ok  alert after 10 s without a hardhat and again after the cooldown; one bad frame, unknown heads, "
          "people outside the zone and low-confidence detections ignored")
    test_incident_ends()
    test_flicker_keeps_the_timer()
    print("PASS: alerts, incidents that end (wearing / left) with names unique across restarts, flicker tolerated")


def play(plugin, seconds, frame_objs, start=0.0):
    """3 checks per second; frame_objs(n) gives the objects of check n. Returns the events."""
    out = []
    for n in range(int(seconds * 3)):
        out += plugin.on_frame(Frame("c", n, start + n / 3, frame_objs(n)))
    return out


def test_incident_ends():
    p = PPEPlugin("c", {"duration_s": 10, "stability_frames": [8, 10]})
    got = play(p, 14, lambda n: [person(2), head(2, NO_HARDHAT)])  # 3 s to fill 10 checks + 10 s
    got += play(p, 6, lambda n: [person(2), head(2, HARDHAT)], start=14)  # puts the hardhat on
    assert [e["class"] for e in got] == ["no_hardhat", "no_hardhat_end"], [e["class"] for e in got]
    start, end = got
    assert start["incident"] == end["incident"] and "@" in start["incident"], "one incident, named with its start time"
    assert end["extra"]["why"] == "wearing"
    got = play(p, 14, lambda n: [person(2), head(2, NO_HARDHAT)], start=100)  # same track number, later
    assert got[0]["incident"] != start["incident"], "a later problem with the same track number is a new incident"
    got = play(p, 12, lambda n: [], start=114)  # walks away
    assert [(e["class"], e["extra"]["why"]) for e in got] == [("no_hardhat_end", "left")], got
    print("ok  ends: closed when the hardhat goes on (wearing) or the person leaves (left); names never reused")


def test_flicker_keeps_the_timer():
    p = PPEPlugin("c", {"duration_s": 10, "stability_frames": [8, 10]})
    got = play(p, 15, lambda n: [] if n % 3 == 2 else [person(2), head(2, NO_HARDHAT)])  # missed every 3rd frame
    assert [e["class"] for e in got] == ["no_hardhat"], f"flicker must not restart the 10 s timer: {got}"
    print("ok  flicker: a person missed every third frame still raises the alert after 10 s")


if __name__ == "__main__":
    run()
