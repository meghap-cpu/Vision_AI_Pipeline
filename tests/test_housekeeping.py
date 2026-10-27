"""Unattended object on made-up frames. Run: python3 -m tests.test_housekeeping

Story (5 fps): a striped car is in view during learning and drives off at 6 s (a ghost trap);
a still person stands far away all along; a bag appears at 5 s with its owner and a stopped car;
the owner leaves at 15 s; the bag is picked up at 40 s.
Expected: unattended_object (~35 s), object_removed (~45 s), and nothing for the ghost,
the still person or the stopped car."""
import numpy as np

from vp.plugin import Frame, Obj
from vp.plugins.housekeeping import HousekeepingPlugin

FPS, H, W = 5, 160, 320
STILL_PERSON = Obj(1, 0, 0.9, (0.11, 0.36, 0.20, 0.82))  # stands still all along, far from the bag
OWNER = Obj(2, 0, 0.9, (0.58, 0.60, 0.61, 0.75))          # stands right next to the bag
CAR = Obj(3, 2, 0.9, (0.76, 0.11, 0.98, 0.33))            # stops with the bag and stays (COCO class 2)


def image(rng, t, bag):
    img = np.clip(100 + rng.normal(0, 2, (H, W)), 0, 255).astype(np.uint8)
    img[60:130, 40:60] = 180                               # the still person's pixels
    if t < 6.0:                                            # striped car in view during learning, then gone
        img[130:158, 200:296] = np.where((np.arange(96) // 16) % 2, 40, 170)  # wide stripes survive the blur
    if bag:
        img[90:120, 150:180] = 200                         # the bag
        img[20:50, 250:310] = 60                           # the stopped car
    return img


def story():
    for n in range(250):
        t = n / FPS
        bag = 5.0 <= t < 40.0
        people = [STILL_PERSON] + ([OWNER] if 5.0 <= t < 15.0 else []) + ([CAR] if bag else [])
        yield n, t, bag, people


def run(params):
    rng = np.random.default_rng(0)
    plugin = HousekeepingPlugin("test-cam", params)
    got = []
    for n, t, bag, people in story():
        for ev in plugin.on_frame(Frame("test-cam", n, t, people, pixels=image(rng, t, bag))):
            got.append((round(t, 1), ev["class"]))
    return got


if __name__ == "__main__":
    on = run({})
    print("ghost check on: ", on)
    assert [c for _, c in on] == ["unattended_object", "object_removed"], f"unexpected events: {on}"
    off = run({"ghost_check": False})
    print("ghost check off:", off)
    assert len(off) > len(on), "the test should trigger a ghost alert when the check is off"
    print("PASS: bag alerted once; no alert for the ghost, the still person or the stopped car "
          "(and the ghost does fire when the check is off)")
