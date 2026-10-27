"""Worker at height. Ported from woker-at-height: a person whose feet are inside a height zone opens an incident;
their crop (30 px padding at 1280 wide) goes to the VLM with the repo's PPE prompt and parsing rules. Missing PPE ->
high (at_height_violation); missing and near an edge -> critical (at_height_critical); no verdict in 10 s (3 s near
an edge) -> high (at_height_unverified); compliant workers are re-checked every 300 s; the incident closes 3 s after
the person leaves the zone. All timings and distances are settings: provisional values set by the development team.
Changed: the VLM call never blocks the video; one alert per level per incident."""
import cv2
import numpy as np

from vp.plugin import Plugin, event
from vp.vlm import VLMClient

PPE_PROMPT = (
    "You are a safety inspector. Examine this image of a worker carefully.\n"
    "Determine whether the worker is wearing:\n"
    "  1. A safety harness (full-body or chest harness)\n"
    "  2. A helmet or hard hat\n\n"
    "Reply ONLY with a JSON object in exactly this format, no other text:\n"
    '{"harness": true/false, "helmet": true/false, '
    '"harness_confidence": "high/medium/low", '
    '"helmet_confidence": "high/medium/low", '
    '"notes": "<one short sentence>"}'
)


def _as_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip().lower() in ("true", "yes"):
        return True
    if isinstance(v, str) and v.strip().lower() in ("false", "no"):
        return False
    return None


def parse_ppe(ans):
    """The repo's rules on a parsed reply. None if unusable."""
    if not isinstance(ans, dict):
        return None
    harness, helmet = _as_bool(ans.get("harness")), _as_bool(ans.get("helmet"))
    if harness is None and helmet is None:
        return None
    harness = False if harness is None else harness  # answered one item but not the other: assume missing
    helmet = False if helmet is None else helmet
    return {"harness": harness, "helmet": helmet,
            "harness_confidence": str(ans.get("harness_confidence", "low")),
            "helmet_confidence": str(ans.get("helmet_confidence", "low")),
            "notes": str(ans.get("notes", "") or ""), "violation": not (harness and helmet)}


def seg_dist(p, a, b):
    (px, py), (ax, ay), (bx, by) = p, a, b
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    t = 0.0 if length_sq == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    return ((px - ax - t * dx) ** 2 + (py - ay - t * dy) ** 2) ** 0.5


class WorkerAtHeightPlugin(Plugin):
    name = "worker_at_height"
    needs_pixels, pixel_mode, pixels_every_s, pixel_max_width = True, "color", 1.0, 1280  # full resolution, 1 per s
    uses_vlm = True

    def __init__(self, camera_id, params):
        super().__init__(camera_id, params)
        g = params.get
        self.zones = [np.array(z, np.float32) for z in g("zones", [])]
        self.edges = g("edges", [])
        self.aspect = g("aspect", 720 / 1280)
        self.edge_dist = g("edge_distance", 70 / 1280)                    # repo: 70 px at 1280 wide
        self.pad = g("crop_padding", 30 / 1280)                           # repo: 30 px at 1280 wide
        self.unverified_s = g("unverified_escalate_s", 10.0)
        self.edge_unverified_s = g("near_edge_unverified_escalate_s", 3.0)
        self.reverify_s = g("reverify_s", 300.0)
        self.grace_s = g("depart_grace_s", 3.0)
        self.image_max_age_s = g("image_max_age_s", 2.0)
        self.error_every_s = g("error_every_s", 60.0)
        v = g("vlm", {})
        self.client = VLMClient(v.get("url", "http://127.0.0.1:8000/v1"), v.get("model", "qwen3-vl"),
                                v.get("timeout_s", 30.0), max_in_flight=g("max_in_flight", 2))
        self.image, self.image_ts, self.incidents, self.last_error = None, -1e9, {}, -1e9

    def _place(self, box):
        """(at height, near an edge) for a person's feet, the bottom centre of their box."""
        feet = ((box[0] + box[2]) / 2, box[3])
        at_height = any(cv2.pointPolygonTest(z, feet, False) >= 0 for z in self.zones)
        p = (feet[0], feet[1] * self.aspect)
        near = any(seg_dist(p, (a[0], a[1] * self.aspect), (b[0], b[1] * self.aspect)) <= self.edge_dist
                   for line in self.edges for a, b in zip(line, line[1:]))
        return at_height, near

    def _crop(self, box):
        h, w = self.image.shape[:2]
        pad = self.pad * w
        x1, y1 = int(max(0, box[0] * w - pad)), int(max(0, box[1] * h - pad))
        x2, y2 = int(min(w, box[2] * w + pad)), int(min(h, box[3] * h + pad))
        return self.image[y1:y2, x1:x2].copy() if x2 > x1 and y2 > y1 else None

    def on_frame(self, frame):
        events = []
        if frame.pixels is not None:
            self.image, self.image_ts = frame.pixels, frame.ts
        for o in frame.objects:
            if o.source != 1 or o.class_id != 0:
                continue
            at_height, near = self._place(o.box)
            if not at_height:
                continue
            inc = self.incidents.setdefault(o.track_id, {"opened": frame.ts, "near_edge": False, "verdict": None,
                                                         "checked": None, "pending": None, "levels": set()})
            inc.update(last_in=frame.ts, box=o.box)
            inc["near_edge"] = inc["near_edge"] or near
            v = inc["verdict"]
            due = v is None or (not v["violation"] and frame.ts - inc["checked"] >= self.reverify_s)
            if due and inc["pending"] is None and self.image is not None \
                    and frame.ts - self.image_ts <= self.image_max_age_s:
                crop = self._crop(o.box)
                if crop is not None:
                    inc["pending"] = self.client.ask([crop], PPE_PROMPT, max_tokens=120)
        for tid, inc in list(self.incidents.items()):
            if inc["pending"] is not None and inc["pending"].done():
                events += self._verdict(frame, inc)
            events += self._escalate(frame, tid, inc)
            if frame.ts - inc["last_in"] >= self.grace_s:
                del self.incidents[tid]
                if inc["levels"]:
                    events.append(event(self, frame, "at_height_end", severity="info", incident=f"{tid}@{inc['opened']:.1f}", confidence=1.0, track_id=tid,
                                        box=tuple(round(x, 3) for x in inc["box"]),
                                        at_height_s=round(inc["last_in"] - inc["opened"], 1)))
        return events

    def _verdict(self, frame, inc):
        fut, inc["pending"] = inc["pending"], None
        try:
            r = fut.result()
        except Exception as e:
            if frame.ts - self.last_error >= self.error_every_s:
                self.last_error = frame.ts
                return [event(self, frame, "vlm_error", severity="warning", error=str(e)[:200])]
            return []
        verdict = parse_ppe(r["answer"])
        if verdict is not None:  # unreadable: stays unverified and is asked again
            inc["verdict"], inc["checked"], inc["vlm_seconds"] = verdict, frame.ts, r["seconds"]
        return []

    def _escalate(self, frame, tid, inc):
        v, levels, out = inc["verdict"], inc["levels"], []

        def emit(cls, severity):
            levels.add(cls)
            extra = dict(near_edge=inc["near_edge"], at_height_s=round(frame.ts - inc["opened"], 1))
            if v:
                extra.update({k: v[k] for k in ("harness", "helmet", "harness_confidence", "helmet_confidence", "notes")})
            out.append(event(self, frame, cls, severity=severity, incident=f"{tid}@{inc['opened']:.1f}", confidence=1.0, track_id=tid,
                             box=tuple(round(x, 3) for x in inc["box"]), **extra))

        if v and v["violation"]:
            if inc["near_edge"] and "at_height_critical" not in levels:
                emit("at_height_critical", "critical")
            elif not inc["near_edge"] and not levels & {"at_height_violation", "at_height_critical"}:
                emit("at_height_violation", "warning")
        elif v is None and "at_height_unverified" not in levels:
            limit = self.edge_unverified_s if inc["near_edge"] else self.unverified_s
            if frame.ts - inc["opened"] >= limit:
                emit("at_height_unverified", "warning")
        return out
