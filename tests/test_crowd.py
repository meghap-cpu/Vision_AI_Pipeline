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
    print("ok  one alert per crowd, member changes keep the same crowd, 3 people do not count")
    test_id_swap()
    test_must_be_continuous()
    test_split()
    test_same_frame_names()
    print("PASS: one alert per crowd even when the tracker swaps every ID; confirmation must be continuous; "
          "a split leaves two crowds; names never clash")


def play(plugin, seconds, people, start=0.0):
    """FPS frames per second; people(n) gives the people on frame n. Returns the events."""
    out = []
    for n in range(int(seconds * FPS)):
        out += plugin.on_frame(Frame("c", n, start + n / FPS, people(n)))
    return out


def test_id_swap():
    p = CrowdPlugin("c", {})
    got = play(p, 5, lambda n: group([1, 2, 3, 4, 5]))
    got += play(p, 20, lambda n: group([31, 32, 33, 34, 35]), start=5)  # same people, every track ID swapped
    got += play(p, 12, lambda n: [], start=25)
    assert [e["class"] for e in got] == ["crowd_start", "crowd_end"], [e["class"] for e in got]
    assert got[1]["extra"]["duration_s"] >= 24, "one crowd from start to end"
    print("ok  every track ID swapped, same place: still one crowd")


def test_must_be_continuous():
    got = play(CrowdPlugin("c", {}), 20, lambda n: group([1, 2, 3, 4]) if n % 8 == 0 else [])  # 1 frame every 1.6 s
    assert got == [], f"a group that flickers in and out is not a crowd that held for 2 s: {got}"
    print("ok  a group seen for one frame every 1.6 s never confirms")


def test_split():
    p = CrowdPlugin("c", {})
    got = play(p, 3, lambda n: group([1, 2, 3, 4, 5, 6, 7, 8]))
    got += play(p, 4, lambda n: group([1, 2, 3, 4], 0.10) + group([5, 6, 7, 8], 0.80), start=3)  # splits in two
    starts = [e for e in got if e["class"] == "crowd_start"]
    assert len(starts) == 2 and starts[0]["incident"] != starts[1]["incident"], [e["class"] for e in got]
    print("ok  a crowd that splits in two: the second half becomes its own crowd (guard: one group per crowd per frame)")


def test_same_frame_names():
    got = play(CrowdPlugin("c", {}), 3, lambda n: group([1, 2, 3, 4], 0.10) + group([5, 6, 7, 8], 0.80))
    assert len(got) == 2 and got[0]["incident"] != got[1]["incident"], [(e["class"], e["incident"]) for e in got]
    print("ok  two crowds first seen on the same frame get different names")


if __name__ == "__main__":
    run()
