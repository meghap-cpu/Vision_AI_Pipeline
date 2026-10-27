"""The plugin interface: what every use case receives and returns."""
from dataclasses import dataclass


@dataclass
class Obj:
    track_id: int
    class_id: int
    confidence: float
    box: tuple  # x1, y1, x2, y2 as fractions of the frame (0 to 1)
    source: int = 1  # which model produced it: 1 = person detector, 3 = PPE model
    keypoints: tuple = None  # pose: 17 joints (x, y, score) as fractions of the frame, if pose ran


@dataclass
class Frame:
    camera_id: str
    frame_num: int
    ts: float  # capture time, seconds since epoch; all timers use this, never frame counts
    objects: list
    pixels: object = None  # the plugin's zone, small and gray; only if it asked


class Plugin:
    """One instance per camera; keeps that camera's state between frames."""
    name = "base"
    needs_pixels = False
    pixel_mode = "gray"  # "gray" (small zone) or "color" (up to 512 px, for the VLM)
    pixels_every_s = 0.0  # 0 = every analysed frame
    pixel_max_width = 512  # colour pixels: widest size sent
    pixel_mode = "gray"  # "gray" (small zone) or "color" (up to 512 px, for the VLM)
    pixels_every_s = 0.0  # 0 = every analysed frame
    pixel_max_width = 512  # colour pixels: widest size sent

    def __init__(self, camera_id: str, params: dict):
        self.camera_id, self.params = camera_id, params

    def on_frame(self, frame: Frame) -> list:
        """Return a list of events (empty when there is nothing to report)."""
        return []


def event(plugin, frame, cls, severity="info", confidence=0.0, track_id=None, incident=None, **extra):
    return {"use_case": plugin.name, "camera_id": frame.camera_id, "ts": frame.ts,
            "frame_num": frame.frame_num, "class": cls, "severity": severity,
            "confidence": round(float(confidence), 3), "track_id": track_id, "incident": incident, "extra": extra}
