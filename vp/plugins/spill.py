"""Spill, from the spill model (YOLO26n detection: 0 Person, 1 Spill; its detections carry spill_source = 5).
Ported from Spill_detection: a spill whose box is at least 40 % covered by a person is dropped (the model mistakes
people for spills); spills under 500 px are dropped (box area on a 1280x720 frame, as the original measured with this
boxes-only model); only spills whose centre is inside a zone count. One alert per spill.
Changed: a spill must be seen in confirm_hits checks over at least confirm_s before it alerts (the original alerted on
its first frame); spill_end when it has not been seen for gone_s, which closes the incident; simple tracking of our own
(spills barely move); people come from the pipeline's person detector, so no second person model runs.
The spill model may run only every few frames (interval), so "gone" is judged by time, never by missed frames."""
import cv2
import numpy as np

from vp.plugin import Plugin, event

FRAME_PX = 1280 * 720  # min_area_px is measured on a frame of this size, as in the original


def _cover(spill, person):
    """Share of the spill box covered by the person box."""
    w = min(spill[2], person[2]) - max(spill[0], person[0])
    h = min(spill[3], person[3]) - max(spill[1], person[1])
    area = (spill[2] - spill[0]) * (spill[3] - spill[1])
    return w * h / area if w > 0 and h > 0 and area > 0 else 0.0


def _iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


class SpillPlugin(Plugin):
    name = "spill"

    def __init__(self, camera_id, params):
        super().__init__(camera_id, params)
        g = params.get
        self.source, self.spill_class = g("spill_source", 5), g("spill_class", 1)
        self.min_conf = g("min_confidence", 0.45)
        self.min_area = g("min_area_px", 500) / FRAME_PX
        self.person_overlap = g("person_overlap", 0.4)
        self.zones = [np.array(z, np.float32) for z in g("zones", [])]
        self.confirm_hits, self.confirm_s = g("confirm_hits", 3), g("confirm_s", 2.0)
        self.gone_s, self.match_iou = g("gone_s", 30.0), g("match_iou", 0.3)
        self.severity = g("severity", "warning")
        self.tracks, self.next_id = {}, 1

    def _in_zone(self, box):
        if not self.zones:
            return True
        centre = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        return any(cv2.pointPolygonTest(z, centre, False) >= 0 for z in self.zones)

    def spills(self, frame):
        """This frame's spills that pass the original's filters."""
        people = [o.box for o in frame.objects if o.source == 1 and o.class_id == 0]
        out = []
        for o in frame.objects:
            if o.source != self.source or o.class_id != self.spill_class or o.confidence < self.min_conf:
                continue
            b = o.box
            if (b[2] - b[0]) * (b[3] - b[1]) < self.min_area:
                continue
            if any(_cover(b, p) >= self.person_overlap for p in people):
                continue
            if self._in_zone(b):
                out.append(o)
        return out

    def on_frame(self, frame):
        events = []
        for o in self.spills(frame):
            best = max(self.tracks.values(), key=lambda t: _iou(t["box"], o.box), default=None)
            if best is None or _iou(best["box"], o.box) < self.match_iou:
                best = self.tracks[self.next_id] = {"id": self.next_id, "first": frame.ts, "hits": 0, "conf": 0.0,
                                                    "alerted": False}
                self.next_id += 1
            best.update(box=o.box, last=frame.ts, hits=best["hits"] + 1, conf=max(best["conf"], o.confidence))
            if not best["alerted"] and best["hits"] >= self.confirm_hits and frame.ts - best["first"] >= self.confirm_s:
                best["alerted"] = True
                b = best["box"]
                events.append(event(self, frame, "spill_detected", severity=self.severity, confidence=best["conf"],
                                    track_id=best["id"], incident=f"{best['first']:.1f}:{best['id']}",
                                    box=[round(v, 4) for v in b], area_px=round((b[2] - b[0]) * (b[3] - b[1]) * FRAME_PX),
                                    seen_s=round(frame.ts - best["first"], 1), checks=best["hits"]))
        for tid, t in list(self.tracks.items()):
            if frame.ts - t["last"] >= self.gone_s:
                del self.tracks[tid]
                if t["alerted"]:
                    events.append(event(self, frame, "spill_end", severity="info", track_id=tid,
                                        incident=f"{t['first']:.1f}:{tid}", lasted_s=round(t["last"] - t["first"], 1)))
        return events
