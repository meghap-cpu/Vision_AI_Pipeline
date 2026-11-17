"""Runs one use case's plugin for every camera that has it, in its own process.
A camera's plugin only sees frames inside its schedule (if it has one). When the schedule ends, its state is cleared,
so it starts fresh next time, and schedule_paused / schedule_resumed are reported.
Failures stay with their camera: bad settings for one camera leave the others working (that camera reports one
plugin_error); a plugin error is reported at most once a minute per camera, with how many happened since; after
10 errors in a row the camera's plugin starts fresh; when it works again, plugin_recovered (info) reports the errors
not yet reported, so none go unseen."""
import signal
import time
import traceback

from vp import vlm
from vp.plugins import REGISTRY
from vp.schedule import Schedule

REPORT_EVERY_S = 60.0  # at most one plugin_error per camera per minute
RESET_AFTER = 10  # errors in a row before the camera's plugin starts fresh
STATS_EVERY_S = 60.0  # VLM counts to the main loop, for the health report


def _event(use_case, camera, ts, cls, severity="info", **extra):
    return {"use_case": use_case, "camera_id": camera, "ts": ts, "class": cls, "severity": severity,
            "confidence": 1.0, "track_id": None, "extra": extra}


class _Camera:
    """One camera's plugin, schedule and error counts."""
    def __init__(self, cls, cam, params):
        self.cls, self.cam, self.params = cls, cam, params
        self.meta = {k: params[k] for k in ("rule", "instance", "model") if params.get(k)}
        self.severity = params.get("severity")  # the rule decides how serious it is
        self.schedule = Schedule(params.get("schedule"), params.get("timezone"))
        self.plugin = cls(cam, params)
        self.active = None  # unknown until the first frame
        self.in_a_row, self.unreported, self.resets, self.last_report = 0, 0, 0, -1e9

    def step(self, use_case, frame):
        out = []
        try:
            now_active = self.schedule.active(frame.ts)
            if now_active != self.active:
                if self.active is not None:  # a real change, not the first frame
                    cls = "schedule_resumed" if now_active else "schedule_paused"
                    out.append({**_event(use_case, self.cam, frame.ts, cls), **self.meta})
                if not now_active:
                    self.plugin = self.cls(self.cam, self.params)  # cleared: starts fresh when it resumes
                self.active = now_active
            if not now_active:
                return out
            for ev in self.plugin.on_frame(frame):
                ev = {**self.meta, **ev}
                if self.severity and ev.get("severity") in ("warning", "critical"):
                    ev["severity"] = self.severity
                out.append(ev)
            if self.unreported:  # it works again: report what the once-a-minute limit held back
                out.append({**_event(use_case, self.cam, frame.ts, "plugin_recovered", errors=self.unreported,
                                     resets=self.resets), **self.meta})
                self.unreported = 0
            self.in_a_row = 0
        except Exception:
            self.in_a_row += 1
            self.unreported += 1
            if self.in_a_row >= RESET_AFTER:  # stuck in a bad state: start this camera's plugin fresh
                self.plugin, self.in_a_row, self.resets = self.cls(self.cam, self.params), 0, self.resets + 1
            if frame.ts - self.last_report >= REPORT_EVERY_S:
                out.append({**_event(use_case, self.cam, frame.ts, "plugin_error", "warning", errors=self.unreported,
                                     resets=self.resets, trace=traceback.format_exc(limit=3)), **self.meta})
                self.unreported, self.last_report = 0, frame.ts
        return out


def run(use_case, cameras, frames_q, events_q):
    """cameras: {camera_id: params}. Stops when it receives None."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl+C: the main process stops workers cleanly
    cls = REGISTRY[use_case]
    cams = {}
    for cam, params in cameras.items():
        try:
            cams[cam] = _Camera(cls, cam, params)
        except Exception:  # bad settings for this camera: say so once, keep the other cameras working
            meta = {k: params[k] for k in ("rule", "instance") if params.get(k)}
            events_q.put({**_event(use_case, cam, time.time(), "plugin_error", "warning", reason="settings",
                                   trace=traceback.format_exc(limit=3)), **meta})
    last_stats = time.time()
    while True:
        frame = frames_q.get()
        if frame is None:
            return
        c = cams.get(frame.camera_id)
        if c is None:
            continue  # this camera's settings failed (reported at start)
        for ev in c.step(use_case, frame):
            events_q.put(ev)
        if time.time() - last_stats >= STATS_EVERY_S:
            last_stats = time.time()
            s = vlm.shared_stats().get(use_case)
            if s:  # for the health report only: the main loop keeps it, it never goes to the outbox
                events_q.put({"stats": "vlm", "use_case": use_case, "values": s})
