"""The plugin worker with a misbehaving plugin, run in-process. Run: python3 -m tests.test_worker"""
import queue

from vp import worker
from vp.plugin import Frame, Plugin, event
from vp.plugins import REGISTRY


class Flaky(Plugin):
    """Says 'tick' once per frame; refuses settings with bad=True; fails on frames before fail_until."""
    name = "flaky_test"

    def __init__(self, camera_id, params):
        super().__init__(camera_id, params)
        if params.get("bad"):
            raise ValueError("zone has 1 point, needs at least 3")
        self.fail_until = params.get("fail_until", -1)

    def on_frame(self, frame):
        if frame.ts < self.fail_until:
            raise RuntimeError("something went wrong in the plugin")
        return [event(self, frame, "tick", severity="warning")]


def run():
    REGISTRY["flaky_test"] = Flaky
    cams = {"good": {"rule": "Test rule"}, "bad": {"bad": True, "rule": "Test rule"},
            "flaky": {"fail_until": 120.0, "rule": "Test rule", "severity": "critical"}}
    frames_q, events_q = queue.Queue(), queue.Queue()
    for n in range(0, 750):  # 150 s at 5 fps, every camera
        for cam in cams:
            frames_q.put(Frame(cam, n, n / 5, []))
    frames_q.put(None)
    worker.run("flaky_test", cams, frames_q, events_q)
    evs = []
    while not events_q.empty():
        evs.append(events_q.get())
    errs = [e for e in evs if e["class"] == "plugin_error"]
    ticks = {cam: sum(1 for e in evs if e["class"] == "tick" and e["camera_id"] == cam) for cam in cams}

    bad = [e for e in errs if e["camera_id"] == "bad"]
    assert len(bad) == 1 and bad[0]["extra"]["reason"] == "settings" and bad[0]["rule"] == "Test rule"
    print("ok  bad settings on one camera: one plugin_error for that camera only")
    assert ticks["good"] == 750, ticks
    print(f"ok  the healthy camera kept working: {ticks['good']} of 750 frames")
    flaky = [e for e in errs if e["camera_id"] == "flaky"]
    print("    flaky camera's reports:", [(e["ts"], e["extra"]["errors"], e["extra"]["resets"]) for e in flaky])
    assert [e["ts"] for e in flaky] == [0.0, 60.0], "at most one report a minute (600 errors, 2 reports)"
    rec = [e for e in evs if e["class"] == "plugin_recovered" and e["camera_id"] == "flaky"]
    print("    then recovered:", [(e["ts"], e["extra"]["errors"], e["extra"]["resets"]) for e in rec])
    assert len(rec) == 1 and rec[0]["ts"] == 120.0 and rec[0]["severity"] == "info"
    assert sum(e["extra"]["errors"] for e in flaky + rec) == 600, "every error is reported, none lost"
    assert rec[0]["extra"]["resets"] == 60, "fresh start every 10 errors in a row"

    assert ticks["flaky"] == 750 - 600, "it works again once the errors stop"
    assert all(e["severity"] == "critical" for e in evs if e["class"] == "tick" and e["camera_id"] == "flaky"), "rule severity kept"
    print(f"ok  600 errors in 2 min -> 2 reports, plugin restarted fresh every 10 errors, then {ticks['flaky']} good frames")
    del REGISTRY["flaky_test"]
    print("PASS: failures stay with their camera; errors reported once a minute; stuck plugins start fresh")


if __name__ == "__main__":
    run()
