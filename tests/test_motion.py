"""Motion episodes on made-up frames: still -> moving square -> still 12 s -> moving again.
Expected: motion_start, motion_end, motion_start. Run: python3 -m tests.test_motion"""
import numpy as np

from vp.plugin import Frame
from vp.plugins.motion import MotionPlugin

FPS, H, W = 5, 160, 320
rng = np.random.default_rng(0)


def still():
    return np.clip(100 + rng.normal(0, 2, (H, W)), 0, 255).astype(np.uint8)


def moving(k):
    img = still()
    x = 20 + (k * 12) % (W - 60)
    img[60:100, x:x + 40] = 220
    return img


def run():
    plugin = MotionPlugin("test-cam", {"quiet_s": 10.0})
    story = ([still() for _ in range(20)]               # 4 s still: background learned
             + [moving(k) for k in range(15)]           # 3 s moving       -> motion_start
             + [still() for _ in range(60)]             # 12 s still       -> motion_end after 10 s
             + [moving(k) for k in range(10)])          # 2 s moving again -> motion_start
    got = []
    for n, img in enumerate(story):
        for ev in plugin.on_frame(Frame("test-cam", n, n / FPS, [], pixels=img)):
            got.append(ev["class"])
            print(f"t={n / FPS:5.1f}s  {ev['class']}  {ev['extra'].get('duration_s', '')}")
    assert got == ["motion_start", "motion_end", "motion_start"], f"unexpected events: {got}"
    print("PASS: one alert per episode, and the episode end is recorded")


if __name__ == "__main__":
    run()
