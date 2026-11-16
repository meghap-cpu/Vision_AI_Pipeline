"""Spill with fake detections, the spill model answering every 5th frame. Run: python3 -m tests.test_spill"""
from vp.plugin import Frame, Obj
from vp.plugins.spill import SpillPlugin

PERSON = Obj(7, 0, 0.9, (0.58, 0.30, 0.68, 0.75), source=1)
SPILLS = {  # name: (box, confidence, seen from, seen until)
    "A real spill":            ((0.30, 0.60, 0.40, 0.70), 0.60, 1.0, 20.0),
    "B under a person":        ((0.60, 0.60, 0.65, 0.70), 0.70, 1.0, 20.0),
    "C too small (92 px)":     ((0.10, 0.80, 0.11, 0.81), 0.70, 1.0, 20.0),
    "D outside the zone":      ((0.50, 0.10, 0.60, 0.20), 0.70, 1.0, 20.0),
    "E one-frame blip":        ((0.80, 0.80, 0.90, 0.90), 0.70, 10.0, 10.0),
    "F low confidence":        ((0.15, 0.55, 0.25, 0.65), 0.30, 1.0, 20.0),
}


def run():
    plugin = SpillPlugin("test-cam", {"zones": [[[0, 0.5], [1, 0.5], [1, 1], [0, 1]]], "min_confidence": 0.45})
    got = []
    for n in range(0, 300):  # 60 s at 5 fps
        t = n / 5
        objs = [PERSON]
        if n % 5 == 0:  # the spill model runs every 5th frame
            objs += [Obj(-1, 1, conf, box, source=5) for box, conf, a, b in SPILLS.values() if a <= t <= b]
        for ev in plugin.on_frame(Frame("test-cam", n, t, objs)):
            got.append((ev["class"], ev["track_id"], round(t, 1), ev["severity"]))
            print(f"t={t:5.1f}s  {ev['class']:15s} track {ev['track_id']}  {ev['severity']:8s} {ev['extra']}")
    assert got == [("spill_detected", 1, 3.0, "warning"), ("spill_end", 1, 50.0, "info")], f"unexpected: {got}"
    print("PASS: only the real spill alerts, after 3 checks over 2 s; person-covered, small, out-of-zone,")
    print("      blip and low-confidence spills never alert; spill_end 30 s after it was last seen")


if __name__ == "__main__":
    run()
