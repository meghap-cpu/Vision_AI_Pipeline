"""Motion in a restricted zone. Ported from Motion-Detection, with two fixes:
- the background adapts slowly (the original locked one reference frame, so lighting changes
  became permanent motion);
- one alert per motion episode (the original alerted every 5 s for as long as anything moved).
"""
import cv2
import numpy as np

from vp.plugin import Plugin, event

KERNEL = np.ones((3, 3), np.uint8)


class MotionPlugin(Plugin):
    name = "motion"
    needs_pixels = True

    def __init__(self, camera_id, params):
        super().__init__(camera_id, params)
        self.zone = params.get("zone", [0, 0, 1, 1])
        self.blur = params.get("blur", 11)
        self.threshold = params.get("threshold", 25)
        self.min_area_frac = params.get("min_area_frac", 0.002)  # share of the zone a blob must cover
        self.adapt = params.get("adapt_rate", 0.02)  # how fast the background follows slow changes
        self.relearn_frac = params.get("relearn_frac", 0.6)  # this much change at once = lighting, not motion
        self.warmup = params.get("warmup_frames", 10)
        self.mode = params.get("alert_mode", "episode")  # "episode" or "cooldown"
        self.quiet_s = params.get("quiet_s", 10.0)  # episode mode: still this long = episode over
        self.cooldown = params.get("cooldown_s", 5.0)  # cooldown mode: minimum gap between alerts
        self.bg, self.frames = None, 0
        self.active, self.started, self.last_motion, self.last_alert = False, 0.0, 0.0, 0.0

    def _blobs(self, frame):
        """Returns (blob boxes as fractions of the zone, share of the zone changed), or None while learning."""
        gray = cv2.GaussianBlur(frame.pixels, (self.blur, self.blur), 0).astype(np.float32)
        self.frames += 1
        if self.bg is None:
            self.bg = gray
            return None
        mask = (cv2.absdiff(gray, self.bg) > self.threshold).astype(np.uint8) * 255
        mask = cv2.dilate(cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL), KERNEL, iterations=2)
        changed = cv2.countNonZero(mask) / mask.size
        if changed > self.relearn_frac:  # sudden change across the zone: relearn, don't alert
            self.bg = gray
            return None
        cv2.accumulateWeighted(gray, self.bg, self.adapt)
        if self.frames < self.warmup:
            return None
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = mask.shape
        boxes = [(round(x / w, 3), round(y / h, 3), round((x + bw) / w, 3), round((y + bh) / h, 3))
                 for x, y, bw, bh in (cv2.boundingRect(c) for c in contours
                                      if cv2.contourArea(c) >= self.min_area_frac * mask.size)]
        return boxes, changed

    def on_frame(self, frame):
        if frame.pixels is None:
            return []
        found = self._blobs(frame)
        boxes, changed = found if found else ([], 0.0)
        if boxes:
            self.last_motion = frame.ts
        details = dict(blobs=len(boxes), changed_frac=round(changed, 4), zone=self.zone, boxes_in_zone=boxes)

        if self.mode == "cooldown":
            if boxes and frame.ts - self.last_alert >= self.cooldown:
                self.last_alert = frame.ts
                return [event(self, frame, "motion", severity="warning", confidence=min(1.0, changed * 10), **details)]
            return []

        if boxes and not self.active:  # a new episode starts: alert once
            self.active, self.started = True, frame.ts
            return [event(self, frame, "motion_start", severity="warning", incident=f"{frame.ts:.1f}", confidence=min(1.0, changed * 10), **details)]
        if self.active and frame.ts - self.last_motion >= self.quiet_s:  # still long enough: episode over
            self.active = False
            return [event(self, frame, "motion_end", severity="info", incident=f"{self.started:.1f}", confidence=1.0,
                          duration_s=round(self.last_motion - self.started, 1), zone=self.zone)]
        return []
