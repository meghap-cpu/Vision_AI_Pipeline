"""Runs one use case's plugin for every camera that has it, in its own process.
A camera's plugin only sees frames inside its schedule (if it has one). When the schedule ends, its state is cleared,
so it starts fresh next time, and schedule_paused / schedule_resumed are reported."""
import signal
import traceback

from vp.plugins import REGISTRY
from vp.schedule import Schedule


def _note(use_case, camera, ts, cls):
    return {"use_case": use_case, "camera_id": camera, "ts": ts, "class": cls, "severity": "info",
            "confidence": 1.0, "track_id": None, "extra": {}}


def run(use_case, cameras, frames_q, events_q):
    """cameras: {camera_id: params}. Stops when it receives None."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl+C: the main process stops workers cleanly
    cls = REGISTRY[use_case]
    schedules = {cam: Schedule(p.get("schedule"), p.get("timezone")) for cam, p in cameras.items()}
    plugins = {cam: cls(cam, params) for cam, params in cameras.items()}
    meta = {cam: {k: p[k] for k in ("rule", "instance", "model") if p.get(k)} for cam, p in cameras.items()}
    active = {cam: None for cam in cameras}  # unknown until each camera's first frame
    while True:
        frame = frames_q.get()
        if frame is None:
            return
        cam = frame.camera_id
        now_active = schedules[cam].active(frame.ts)
        if now_active != active[cam]:
            if active[cam] is not None:  # a real change, not the first frame
                events_q.put({**_note(use_case, cam, frame.ts, "schedule_resumed" if now_active else "schedule_paused"), **meta[cam]})
            if not now_active:
                plugins[cam] = cls(cam, cameras[cam])  # cleared: starts fresh when the schedule resumes
            active[cam] = now_active
        if not now_active:
            continue
        try:
            for ev in plugins[cam].on_frame(frame):
                ev = {**meta[cam], **ev}
                if cameras[cam].get("severity") and ev.get("severity") in ("warning", "critical"):
                    ev["severity"] = cameras[cam]["severity"]  # the rule decides how serious it is
                events_q.put(ev)
        except Exception:
            events_q.put({"use_case": use_case, "camera_id": cam, "ts": frame.ts,
                          "class": "plugin_error", "severity": "warning",
                          "extra": {"trace": traceback.format_exc(limit=3)}})
