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
    print("PASS: one alert for the real fall, an end record on recovery; trip, crouch and partial body ignored; a wobbling score still alerts")


if __name__ == "__main__":
    run()
