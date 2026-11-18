"""Worker at height. Ported from woker-at-height: a person whose feet are inside a height zone opens an incident;
their crop (30 px padding at 1280 wide) goes to the VLM with the repo's PPE prompt and parsing rules, plus a vest.
Priorities as in the repo: in the zone for entered_after_s (5 s) -> low (at_height_entered, sent as warning; skipped
if the incident is already high, so walk-throughs and tracker blips raise nothing); any of harness, helmet or
vest missing -> high (at_height_violation, sent as critical); missing and near an edge -> at_height_critical; no
verdict in 10 s (3 s near an edge) -> high (at_height_unverified, with why: no_answer, or too_small when the person
is under min_person_px tall and is never asked). Compliant workers are re-checked every 300 s. A crop is only cut
from the picture of the same frame as the boxes; after an unreadable answer or an error the person waits retry_s.
A camera with no zone is a settings error.
3 s after the person leaves: a low-only incident closes (at_height_end); one that reached high stays open for an
operator to review (at_height_left, a record only), as in the repo. All timings and distances are settings:
provisional values set by the development team. Keep the CampNeuron rule's severity blank, or it flattens these levels.
Changed: the VLM call never blocks the video; one alert per level per incident; re-checks stay in the same incident."""
import cv2
import numpy as np

from vp.plugin import Plugin, event
from vp import vlm

PPE_PROMPT = (
    "You are a safety inspector. Examine this image of a worker carefully.\n"
    "Determine whether the worker is wearing:\n"
    "  1. A safety harness (full-body or chest harness)\n"
    "  2. A helmet or hard hat\n"
    "  3. A high-visibility safety vest\n\n"
    "Reply ONLY with a JSON object in exactly this format, no other text:\n"
    '{"harness": true/false, "helmet": true/false, "vest": true/false, '
    '"harness_confidence": "high/medium/low", '
    '"helmet_confidence": "high/medium/low", '
    '"vest_confidence": "high/medium/low", '
    '"notes": "<one short sentence>"}'
)


ITEMS = ("harness", "helmet", "vest")


def parse_ppe(ans):
    """The repo's rules on a parsed reply. None if unusable."""
    if not isinstance(ans, dict):
        return None
    items = {k: vlm.as_bool(ans.get(k)) for k in ITEMS}
    if all(v is None for v in items.values()):
        return None
    items = {k: False if v is None else v for k, v in items.items()}  # answered some items but not all: assume missing
    return {**items, **{f"{k}_confidence": str(ans.get(f"{k}_confidence", "low")) for k in ITEMS},
            "notes": str(ans.get("notes", "") or ""), "violation": not all(items.values())}


def seg_dist(p, a, b):
    (px, py), (ax, ay), (bx, by) = p, a, b
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    t = 0.0 if length_sq == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    return ((px - ax - t * dx) ** 2 + (py - ay - t * dy) ** 2) ** 0.5


class WorkerAtHeightPlugin(Plugin):
    name = "worker_at_height"
    needs_pixels, pixel_mode, pixels_every_s, pixel_max_width = True, "color", 0.5, 1280  # full resolution, 2 per s
    uses_vlm = True

    def __init__(self, camera_id, params):
        super().__init__(camera_id, params)
        g = params.get
        self.zones = [np.array(z, np.float32) for z in g("zones", [])]
        if not self.zones:  # a camera with no height zone would silently never alert: the worker reports this
            raise ValueError("worker_at_height needs at least one zone")
        self.edges = g("edges", [])
        self.aspect = g("aspect", 720 / 1280)
        self.edge_dist = g("edge_distance", 70 / 1280)                    # repo: 70 px at 1280 wide
        self.pad = g("crop_padding", 30 / 1280)                           # repo: 30 px at 1280 wide
        self.unverified_s = g("unverified_escalate_s", 10.0)
        self.edge_unverified_s = g("near_edge_unverified_escalate_s", 3.0)
        self.reverify_s = g("reverify_s", 300.0)
        self.grace_s = g("depart_grace_s", 3.0)
        self.entered_after_s = g("entered_after_s", 5.0)  # time in the zone before the low alert
        self.min_person_px = g("min_person_px", 48)   # shorter than this in the picture: too small to judge PPE
        self.retry_s = g("retry_s", 1.0)              # pause before asking again after an unreadable answer or error
        self.error_every_s = g("error_every_s", 60.0)
        # one client for every camera of this use case in the worker, 2 slots per camera; when it is busy, ask()
        # returns None and the person is asked about again on a later frame, with a fresh crop
        self.client = vlm.shared(self.name, g("vlm", {}), camera_id, g("max_in_flight", 2))
        self.incidents, self.last_error = {}, -1e9

    def _place(self, box):
        """(at height, near an edge) for a person's feet, the bottom centre of their box."""
        feet = ((box[0] + box[2]) / 2, box[3])
        at_height = any(cv2.pointPolygonTest(z, feet, False) >= 0 for z in self.zones)
        p = (feet[0], feet[1] * self.aspect)
        near = any(seg_dist(p, (a[0], a[1] * self.aspect), (b[0], b[1] * self.aspect)) <= self.edge_dist
                   for line in self.edges for a, b in zip(line, line[1:]))
        return at_height, near

    def _crop(self, image, box):
        """The person with some padding, or None if they are too small in the picture to judge (stays unverified)."""
        h, w = image.shape[:2]
        if (box[3] - box[1]) * h < self.min_person_px:
            return None
        pad = self.pad * w
        x1, y1 = int(max(0, box[0] * w - pad)), int(max(0, box[1] * h - pad))
        x2, y2 = int(min(w, box[2] * w + pad)), int(min(h, box[3] * h + pad))
        return image[y1:y2, x1:x2].copy() if x2 > x1 and y2 > y1 else None

    def on_frame(self, frame):
        events = []
        due = []  # (deadline, incident) of people waiting for a check
        for o in frame.objects:
            if o.source != 1 or o.class_id != 0:
                continue
            at_height, near = self._place(o.box)
            if not at_height:
                continue
            inc = self.incidents.setdefault(o.track_id, {"opened": frame.ts, "near_edge": False, "verdict": None,
                                                         "checked": None, "pending": None, "levels": set(),
                                                         "retry_at": -1e9, "too_small": False})
            inc.update(last_in=frame.ts, box=o.box)
            inc["near_edge"] = inc["near_edge"] or near
            v = inc["verdict"]
            if inc["pending"] is None and frame.ts >= inc["retry_at"] and \
                    (v is None or (not v["violation"] and frame.ts - inc["checked"] >= self.reverify_s)):
                due.append((self._deadline(inc), inc))
        # ask only with this frame's own picture: the boxes are from the same moment, so the crop holds the person
        if due and frame.pixels is not None:
            for _, inc in sorted(due, key=lambda d: d[0]):  # most urgent first: near an edge, then the longest wait
                crop = self._crop(frame.pixels, inc["box"])
                inc["too_small"] = crop is None
                if crop is None:
                    continue
                inc["pending"] = self.client.ask([crop], PPE_PROMPT, max_tokens=160)
                if inc["pending"] is None:
                    break  # no slot free: the rest wait for a later frame
        for tid, inc in list(self.incidents.items()):
            if inc["pending"] is not None and inc["pending"].done():
                events += self._verdict(frame, inc)
            events += self._escalate(frame, tid, inc)
            if frame.ts - inc["last_in"] >= self.grace_s:
                del self.incidents[tid]
                if inc["levels"]:  # reached high: stays open for an operator (repo); low only: closes
                    cls = "at_height_left" if inc["levels"] - {"at_height_entered"} else "at_height_end"
                    events.append(event(self, frame, cls, severity="info", incident=f"{tid}@{inc['opened']:.1f}", confidence=1.0, track_id=tid,
                                        box=tuple(round(x, 3) for x in inc["box"]),
                                        at_height_s=round(inc["last_in"] - inc["opened"], 1)))
        return events

    def _deadline(self, inc):
        """When this person escalates to at_height_unverified; re-checks of compliant workers come last."""
        if inc["verdict"] is not None:
            return float("inf")
        return inc["opened"] + (self.edge_unverified_s if inc["near_edge"] else self.unverified_s)

    def _verdict(self, frame, inc):
        fut, inc["pending"] = inc["pending"], None
        try:
            r = fut.result()
        except Exception as e:
            inc["retry_at"] = frame.ts + self.retry_s
            if frame.ts - self.last_error >= self.error_every_s:
                self.last_error = frame.ts
                return [event(self, frame, "vlm_error", severity="info", error=str(e)[:200])]  # alert: vlm_down
            return []
        verdict = parse_ppe(r["answer"])
        if verdict is None:  # unreadable: stays unverified, asked again after a pause so it can't hog the slots
            inc["retry_at"] = frame.ts + self.retry_s
            self.client.note_unreadable()
        else:
            inc["verdict"], inc["checked"], inc["vlm_seconds"] = verdict, frame.ts, r["seconds"]
        return []

    def _escalate(self, frame, tid, inc):
        v, levels, out = inc["verdict"], inc["levels"], []

        def emit(cls, severity):
            levels.add(cls)
            extra = dict(near_edge=inc["near_edge"], at_height_s=round(frame.ts - inc["opened"], 1))
            if v:
                extra.update({k: v[k] for k in (*ITEMS, *(f"{i}_confidence" for i in ITEMS), "notes")})
            elif cls == "at_height_unverified":
                extra["why"] = inc["why"]
            out.append(event(self, frame, cls, severity=severity, incident=f"{tid}@{inc['opened']:.1f}", confidence=1.0, track_id=tid,
                             box=tuple(round(x, 3) for x in inc["box"]), **extra))

        if not levels and inc["last_in"] - inc["opened"] >= self.entered_after_s:
            emit("at_height_entered", "warning")  # low: stayed in the zone, nothing high yet
        if v and v["violation"]:
            if inc["near_edge"] and "at_height_critical" not in levels:
                emit("at_height_critical", "critical")
            elif not inc["near_edge"] and not levels & {"at_height_violation", "at_height_critical"}:
                emit("at_height_violation", "critical")  # high
        elif v is None and "at_height_unverified" not in levels:
            limit = self.edge_unverified_s if inc["near_edge"] else self.unverified_s
            if frame.ts - inc["opened"] >= limit:  # the operator sees why: too small to judge, or no answer in time
                inc["why"] = "too_small" if inc["too_small"] else "no_answer"
                emit("at_height_unverified", "critical")  # high
        return out
