"""Crowd episodes on made-up people. Run: python3 -m tests.test_crowd"""
from vp.plugin import Frame, Obj
from vp.plugins.crowd import CrowdPlugin

FPS = 5


def person(track_id, x, y):  # a 0.03 x 0.15 box with its feet at (x, y)
    return Obj(track_id, 0, 0.9, (x - 0.015, y - 0.15, x + 0.015, y))


def group(ids, x0=0.45):
    return [person(t, x0 + 0.02 * n, 0.6 + 0.01 * (n % 2)) for n, t in enumerate(ids)]


def run():
    plugin = CrowdPlugin("test-cam", {})
    story = ([group([1, 2, 3, 4, 5]) + [person(6, 0.05, 0.6), person(7, 0.95, 0.6)] for _ in range(15)]  # 3 s
             + [group([1, 2, 4, 5, 10]) for _ in range(10)]          # 2 s: 3 leaves, 10 joins: same crowd
             + [[person(t, 0.05 + 0.3 * n, 0.6) for n, t in enumerate([1, 2, 4, 5])] for _ in range(60)]  # 12 s apart
             + [group([20, 21, 22]) for _ in range(15)])             # 3 s: only 3 close, below minimum
    got = []
    for n, people in enumerate(story):
        for ev in plugin.on_frame(Frame("test-cam", n, n / FPS, people)):
            got.append(ev["class"])
            x = ev["extra"]
            print(f"t={n / FPS:5.1f}s  {ev['class']}  people={x.get('people', x.get('peak_people'))}  {x.get('duration_s', '')}")
    assert got == ["crowd_start", "crowd_end"], f"unexpected events: {got}"
    print("PASS: one alert per crowd, member changes keep the same crowd, 3 people do not count")


if __name__ == "__main__":
    run()
