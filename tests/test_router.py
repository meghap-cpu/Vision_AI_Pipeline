"""The probe's decisions (vp/router.py), without a camera or GPU. Run: python3 -m tests.test_router"""
from vp.router import CropStore, Router, SnapshotRequests


def test_pixels_only_when_due():
    # camera 0: motion needs gray pixels every 0.2 s, crowd needs none
    r = Router([["motion", "crowd"]], [{"motion": ([0, 0, 1, 1], "gray", 0.2, None)}])
    due = [(t, [uc for uc, spec in r.plan(0, t) if spec]) for t in (0.0, 0.1, 0.2, 0.25, 0.4)]
    assert due == [(0.0, ["motion"]), (0.1, []), (0.2, ["motion"]), (0.25, []), (0.4, ["motion"])], due
    assert [uc for uc, _ in r.plan(0, 1.0)] == ["motion", "crowd"], "every use case still gets every frame"
    print("ok  pixels are due only at each plugin's rate; frames between need no pixel mapping")


def test_drops_per_use_case():
    r = Router([["ppe", "spill"]], [{}])
    for uc in ["ppe", "ppe", "spill"]:
        r.drop(uc)
    assert r.dropped_by_use_case() == {"ppe": 2, "spill": 1}
    print("ok  dropped frames are counted per use case")


def test_snapshot_requests_expire():
    now = [100.0]
    s = SnapshotRequests(max_age_s=5.0, clock=lambda: now[0])
    s.add("cam01", "e1", {"class": "no_vest"})
    now[0] = 103.0
    assert s.take("cam01") == [("e1", {"class": "no_vest"})], "a fresh request is served"
    assert s.take("cam01") == [], "and only once"
    s.add("cam02", "e2", {}); now[0] = 110.0   # cam02 went offline: its next frame comes 10 s later
    assert s.take("cam02") == [] and s.expired == 1, "a stale request never gets a photo from a later scene"
    s.add("cam03", "e3", {}); now[0] = 116.0
    s.prune()                                   # cam03 sends no frames at all: the main loop drops it
    assert s.items == {} and s.expired == 2
    print("ok  snapshot requests expire after 5 s, on take and on prune; expiries counted")


def test_crops_cannot_pile_up():
    c = CropStore(keep_frames=50)
    for frame in range(1000):                  # a person whose crop the probe never reads, on every frame
        c.put((0, frame, 7), ("real", "crop"))
        c.put((1, frame, 9), ("real", "crop"))
    assert len(c) <= 2 * 51, len(c)
    assert c.get((0, 999, 7)) == ("real", "crop") and c.pop((0, 999, 7)) == ("real", "crop") and c.get((0, 999, 7)) is None
    assert c.pruned == 2 * 1000 - len(c) - 1
    print(f"ok  1,000 frames of unread crops on 2 cameras: only {len(c)} kept (last 50 frames each)")


def run():
    test_pixels_only_when_due(); test_drops_per_use_case(); test_snapshot_requests_expire(); test_crops_cannot_pile_up()
    print("PASS: routing, pixel timing, per-use-case drops, snapshot expiry and crop clean-up")


if __name__ == "__main__":
    run()
