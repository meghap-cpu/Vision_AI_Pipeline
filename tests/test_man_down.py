"""Man-down on made-up skeletons at 5 fps on a 1280x720 frame. Run: python3 -m tests.test_man_down"""
from vp.plugin import Frame, Obj
from vp.plugins.man_down import ManDownPlugin

W, H, FPS = 1280, 720, 5
# Each joint as (f, u): f = position along the body from head (0) to feet (1), u = sideways offset.
UPRIGHT = [(0.06, 0), (0.05, -.02), (0.05, .02), (0.06, -.05), (0.06, .05), (0.20, -.12), (0.20, .12), (0.35, -.15),
           (0.35, .15), (0.48, -.15), (0.48, .15), (0.52, -.08), (0.52, .08), (0.75, -.08), (0.75, .08), (0.97, -.08), (0.97, .08)]
CROUCH = UPRIGHT[:11] + [(0.55, -.08), (0.55, .08), (0.62, -.18), (0.62, .18), (0.80, -.10), (0.80, .10)]


def person(tid, pose, lying=False, visible=17, size=300, at=(400, 100)):
    pts, score = [], []
    for j, (f, u) in enumerate(pose):
        x, y = (at[0] + f * size, at[1] + 200 + u * size) if lying else (at[0] + u * size, at[1] + f * size)
        pts.append((x, y))
        score.append(0.9 if j < visible else 0.05)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    box = (min(xs) - 15, min(ys) - 15, max(xs) + 15, max(ys) + 15)
    kp = tuple((x / W, y / H, s) for (x, y), s in zip(pts, score))
    return Obj(tid, 0, 0.9, (box[0] / W, box[1] / H, box[2] / W, box[3] / H), source=1, keypoints=kp)


def run():
    plugin = ManDownPlugin("test-cam", {})
    story = ([("stand", 10)] + [("trip", 2), ("stand", 6)]            # a 2 s trip: too short
             + [("fall", 10)] + [("stand", 6)]                       # a real fall, then recovery
             + [("crouch", 10)] + [("partial", 10)]                  # crouching; lying with lower body hidden
             + [("wobbly", 12)])                                     # lying, score dips for 2 frames every 2 s
    got, t = [], 0.0
    for kind, seconds in story:
        for _ in range(seconds * FPS):
            if kind in ("stand",):
                o = person(1, UPRIGHT)
            elif kind in ("trip", "fall"):
                o = person(1, UPRIGHT, lying=True)
            elif kind == "crouch":
                o = person(2, CROUCH)
            elif kind == "partial":
                o = person(3, UPRIGHT, lying=True, visible=11)  # hips and legs not visible
            else:  # wobbly: lying, but two crouch-like frames every 2 s pull the score into the crouching band
                o = person(4, CROUCH) if int(t * FPS) % 10 in (0, 1) else person(4, UPRIGHT, lying=True)
            for ev in plugin.on_frame(Frame("test-cam", int(t * FPS), t, [o])):
                got.append((ev["class"], ev["track_id"]))
                print(f"t={t:5.1f}s  {ev['class']}  track {ev['track_id']}  {ev['extra'].get('down_s', '')}  "
                      f"{ev['extra'].get('signals', '')}")
            t += 1 / FPS
    assert got == [("man_down", 1), ("man_down_end", 1), ("man_down", 4)], f"unexpected events: {got}"
    print("ok  one alert for the real fall, an end record on recovery; trip, crouch and partial body ignored; "
          "a wobbling score still alerts")
    test_small_person_lying()
    test_lost_person()
    print("PASS: falls alert, recoveries end them; small people lying down are judged; lost people end as 'lost'")


def run_plugin(plugin, seconds, objs, start=0.0):
    """FPS frames per second; objs(t) gives the objects at time t. Returns the events (pose_stats left out)."""
    out = []
    for n in range(int(seconds * FPS)):
        t = start + n / FPS
        out += [e for e in plugin.on_frame(Frame("c", n, t, objs(t))) if e["class"] != "pose_stats"]
    return out


def test_small_person_lying():
    # 150 px tall standing, but only ~75 px tall lying down (180 px wide): the guard must look at the longer side
    small = person(1, UPRIGHT, lying=True, size=150)
    assert (small.box[3] - small.box[1]) * H < 80 < (small.box[2] - small.box[0]) * W
    got = run_plugin(ManDownPlugin("c", {}), 10, lambda t: [small])
    assert [e["class"] for e in got] == ["man_down"], got
    print("ok  a small worker lying down (75 px tall, 180 px wide) is judged and alerts")


def test_lost_person():
    p = ManDownPlugin("c", {"lost_s": 60})
    got = run_plugin(p, 10, lambda t: [person(1, UPRIGHT, lying=True)])
    got += run_plugin(p, 30, lambda t: [person(1, UPRIGHT, lying=True, visible=11)], start=10)  # covered: unknown
    assert [e["class"] for e in got] == ["man_down"], f"seen but unscoreable must keep the alarm open: {got}"
    got = run_plugin(p, 59, lambda t: [], start=40)  # carried away
    assert got == [], f"not before lost_s: {got}"
    got = run_plugin(p, 3, lambda t: [], start=99)
    assert [(e["class"], e["extra"]["why"]) for e in got] == [("man_down_end", "lost")], got
    assert 1 not in p.people, "a lost person is forgotten"
    print("ok  alarmed person covered (unscoreable) stays open; gone for 60 s -> man_down_end, why: lost, forgotten")


if __name__ == "__main__":
    run()
