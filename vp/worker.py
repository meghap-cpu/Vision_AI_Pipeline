"""Runs one use case's plugin for every camera that has it, in its own process."""
import traceback

from vp.plugins import REGISTRY


def run(use_case, cameras, frames_q, events_q):
    """cameras: {camera_id: params}. Stops when it receives None."""
    cls = REGISTRY[use_case]
    plugins = {cam: cls(cam, params) for cam, params in cameras.items()}
    while True:
        frame = frames_q.get()
        if frame is None:
            return
        try:
            for ev in plugins[frame.camera_id].on_frame(frame):
                events_q.put(ev)
        except Exception:
            events_q.put({"use_case": use_case, "camera_id": frame.camera_id, "ts": frame.ts,
                          "class": "plugin_error", "severity": "warning",
                          "extra": {"trace": traceback.format_exc(limit=3)}})
