"""Unattended object. Ported from housekeeping: a still object in the zone with nobody near it for a
set time. Changed from the repo:
- two backgrounds instead of one frozen reference: lighting drift is learned, a left object is not;
- the starting background is the median of the first seconds, so passing traffic is not baked in;
- ghost check: a still shape with fewer edges than the background it replaced is something that LEFT,
  so that patch is relearned instead of alerting;
- still people and vehicles (shared detector) are never left objects; one alert per object;
- all timing from frame timestamps.
"""
import math

import cv2
import numpy as np

from vp.plugin import Plugin, event

KERNEL = np.ones((3, 3), np.uint8)


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a, b):
    return _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def iou(a, b):
    i = _inter(a, b)
    return i / (_area(a) + _area(b) - i) if i else 0.0


def covered(a, b):  # share of box a covered by box b
    return _inter(a, b) / _area(a) if _area(a) else 0.0


class HousekeepingPlugin(Plugin):
    name = "housekeeping"
    needs_pixels = True

    def __init__(self, camera_id, params):
        super().__init__(camera_id, params)
        self.zone = params.get("zone", [0, 0, 1, 1])
        self.blur = params.get("blur", 11)
        self.threshold = params.get("threshold", 25)
        self.min_area_frac = params.get("min_area_frac", 0.002)
        self.fast = params.get("fast_rate", 0.1)  # absorbs anything still within ~3 s
        self.slow = params.get("slow_rate", 0.001)  # follows lighting drift only
        self.learn_s = params.get("learn_s", 5.0)  # learn the empty zone first (median of these frames)
        self.relearn_frac = params.get("relearn_frac", 0.6)
        self.ghost_check = params.get("ghost_check", True)
        self.ghost_ratio = params.get("ghost_ratio", 0.7)  # fewer edges than this share of the background = ghost
        self.alone_dist = params.get("alone_dist", 0.117)  # 150 px at 1280: nobody closer = alone
        # COCO classes never treated as left objects: person, bicycle, car, motorcycle, bus, truck
        self.ignore_classes = set(params.get("ignore_classes", [0, 1, 2, 3, 5, 7]))
        self.aspect = params.get("aspect", 720 / 1280)
        self.unattended_s = params.get("unattended_s", 20.0)
        self.quiet_s = params.get("quiet_s", 5.0)
        self.bg_slow = self.bg_fast = self.started = None
        self.learning, self.objects = [], []

    def on_frame(self, frame):
        if frame.pixels is None:
            return []
        gray = cv2.GaussianBlur(frame.pixels, (self.blur, self.blur), 0)
        if self.started is None:
            self.started = frame.ts
        if self.bg_slow is None:  # learning: collect frames, then take the median
            self.learning.append(gray)
            if frame.ts - self.started >= self.learn_s:
                self.bg_slow = np.median(np.stack(self.learning), axis=0).astype(np.float32)
                self.bg_fast, self.learning = self.bg_slow.copy(), []
            return []
        g = gray.astype(np.float32)
        diff_slow = (cv2.absdiff(g, self.bg_slow) > self.threshold).astype(np.uint8)
        diff_fast = (cv2.absdiff(g, self.bg_fast) > self.threshold).astype(np.uint8)
        if diff_slow.mean() > self.relearn_frac:  # lights changed across the zone: relearn
            self.bg_slow, self.bg_fast = g.copy(), g.copy()
            return []
        # open removes speckle; dilate joins pieces of one shape (a striped object must not become 3 alerts)
        still = cv2.dilate(cv2.morphologyEx(diff_slow & (1 - diff_fast), cv2.MORPH_OPEN, KERNEL),
                           KERNEL, iterations=3) * 255
        rects = self._rects(still)
        if self.ghost_check:
            rects = [r for r in rects if not self._ghost(gray, r)]
        cv2.accumulateWeighted(g, self.bg_fast, self.fast)
        cv2.accumulateWeighted(g, self.bg_slow, self.slow, mask=(1 - diff_slow))  # never absorb new things
        return self._track(frame, [self._to_frame(r, still.shape) for r in rects])

    def _rects(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= self.min_area_frac * mask.size]

    def _ghost(self, gray, rect):
        """Fewer edges now than in the background = something left this spot. Relearn it, don't alert."""
        x, y, w, h = rect
        pad = max(4, max(w, h) // 2)  # look around the shape too: a ghost's missing edges sit at its border
        y1, y2 = max(0, y - pad), min(gray.shape[0], y + h + pad)
        x1, x2 = max(0, x - pad), min(gray.shape[1], x + w + pad)
        now = np.count_nonzero(cv2.Canny(gray[y1:y2, x1:x2], 50, 150))
        before = np.count_nonzero(cv2.Canny(self.bg_slow[y1:y2, x1:x2].astype(np.uint8), 50, 150))
        if now < self.ghost_ratio * before:
            self.bg_slow[y:y + h, x:x + w] = gray[y:y + h, x:x + w]
            return True
        return False

    def _to_frame(self, rect, shape):
        """Pixel rectangle inside the zone crop -> box in whole-frame fractions."""
        x, y, w, h = rect
        mh, mw = shape
        zx1, zy1, zx2, zy2 = self.zone
        zw, zh = zx2 - zx1, zy2 - zy1
        return (zx1 + zw * x / mw, zy1 + zh * y / mh, zx1 + zw * (x + w) / mw, zy1 + zh * (y + h) / mh)

    def _track(self, frame, boxes):
        people = [o.box for o in frame.objects if o.source == 1 and o.class_id == 0]
        ignore = [o.box for o in frame.objects if o.source == 1 and o.class_id in self.ignore_classes]
        boxes = [b for b in boxes if not any(covered(b, p) > 0.5 for p in ignore)]  # still people and vehicles are not left objects
        events = []
        for b in boxes:
            obj = max(self.objects, key=lambda o: iou(o["box"], b), default=None)
            if obj is None or iou(obj["box"], b) < 0.3:
                obj = {"first": frame.ts, "alone_since": None, "alerted": False}
                self.objects.append(obj)
            obj.update(box=b, last=frame.ts)
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            near = any(math.hypot((p[0] + p[2]) / 2 - cx, (p[3] - cy) * self.aspect) <= self.alone_dist for p in people)
            if near:
                obj["alone_since"] = None
            elif obj["alone_since"] is None:
                obj["alone_since"] = frame.ts
            if (not obj["alerted"] and obj["alone_since"] is not None
                    and frame.ts - obj["alone_since"] >= self.unattended_s):
                obj["alerted"] = True
                events.append(event(self, frame, "unattended_object", severity="warning", incident=f"{obj['first']:.1f}", confidence=1.0,
                                    box=tuple(round(v, 3) for v in b), still_s=round(frame.ts - obj["first"], 1),
                                    alone_s=round(frame.ts - obj["alone_since"], 1)))
        for obj in [o for o in self.objects if frame.ts - o["last"] >= self.quiet_s]:
            self.objects.remove(obj)
            if obj["alerted"]:
                events.append(event(self, frame, "object_removed", severity="info", incident=f"{obj['first']:.1f}", confidence=1.0,
                                    box=tuple(round(v, 3) for v in obj["box"]),
                                    present_s=round(obj["last"] - obj["first"], 1)))
        return events
