"""Crowd detection. Ported from crowd_detection: DBSCAN on feet points, crowd identity by Jaccard
overlap of track IDs. Changed: one alert per crowd (start + end), a short confirmation time, and
all timing from frame timestamps. Also:
- a group is the same crowd if its track IDs overlap (match_jaccard) OR it stands in the same place (its box overlaps
  the crowd's last box by match_iou): trackers swap IDs most inside crowds, and a swap must not raise a second alert;
- the confirmation must be continuous: a group unseen for over gap_s starts its confirm_s again;
- each crowd takes at most one group per frame (biggest first), so a crowd that splits leaves a second crowd;
- incident names carry a counter, so two crowds first seen on the same frame stay apart."""
import numpy as np

from vp.plugin import Plugin, event


def clusters(points, eps, min_samples):
    """A small DBSCAN for a few dozen people: lists of point indexes, noise left out."""
    if not points:
        return []
    p = np.asarray(points, dtype=float)
    near = np.linalg.norm(p[:, None] - p[None, :], axis=2) <= eps
    core = near.sum(axis=1) >= min_samples
    label, cid = [-1] * len(p), 0
    for i in range(len(p)):
        if label[i] != -1 or not core[i]:
            continue
        label[i], stack = cid, [i]
        while stack:
            j = stack.pop()
            if core[j]:
                for k in np.flatnonzero(near[j]):
                    if label[k] == -1:
                        label[k] = cid
                        stack.append(k)
        cid += 1
    return [[i for i in range(len(p)) if label[i] == c] for c in range(cid)]


def jaccard(a, b):
    return len(a & b) / len(a | b) if a or b else 0.0


def iou(a, b):
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    inter = max(w, 0.0) * max(h, 0.0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class CrowdPlugin(Plugin):
    name = "crowd"

    def __init__(self, camera_id, params):
        super().__init__(camera_id, params)
        self.eps = params.get("eps", 0.234)  # group radius as a share of frame width (300 px at 1280)
        self.min_people = params.get("min_people", 4)
        self.aspect = params.get("aspect", 720 / 1280)  # frame height / width
        self.confirm_s = params.get("confirm_s", 2.0)  # a crowd must hold this long before alerting
        self.gap_s = params.get("gap_s", 1.0)  # unseen longer than this while confirming: start again
        self.quiet_s = params.get("quiet_s", 10.0)  # gone this long = crowd over
        self.match = params.get("match_jaccard", 0.5)  # this much ID overlap = the same crowd
        self.match_iou = params.get("match_iou", 0.3)  # or this much box overlap (same place) = the same crowd
        self.episodes, self.count = [], 0

    def _same(self, ep, ids, box):
        """How well a group matches a crowd (higher is better), or None if it is not the same crowd."""
        j, o = jaccard(ep["members"], ids), iou(ep["box"], box)
        return j + o if j >= self.match or o >= self.match_iou else None

    def on_frame(self, frame):
        people = [o for o in frame.objects if o.source == 1 and o.class_id == 0]
        feet = [((o.box[0] + o.box[2]) / 2, o.box[3] * self.aspect) for o in people]
        events, free = [], list(self.episodes)  # crowds not yet matched on this frame
        for idx in sorted(clusters(feet, self.eps, self.min_people), key=len, reverse=True):
            ids = {people[i].track_id for i in idx}
            box = tuple(round(f(people[i].box[k] for i in idx), 3) for k, f in ((0, min), (1, min), (2, max), (3, max)))
            scored = [(s, ep) for ep in free if (s := self._same(ep, ids, box)) is not None]
            if scored:
                ep = max(scored, key=lambda x: x[0])[1]
                free.remove(ep)
                if not ep["alerted"] and frame.ts - ep["last"] > self.gap_s:
                    ep.update(first=frame.ts, peak=0)  # not continuous: confirm again
            else:
                ep = {"first": frame.ts, "alerted": False, "peak": 0}
                self.episodes.append(ep)
            ep.update(members=ids, last=frame.ts, box=box, peak=max(ep["peak"], len(ids)))
            if not ep["alerted"] and frame.ts - ep["first"] >= self.confirm_s:
                self.count += 1
                ep["alerted"], ep["key"] = True, f"{ep['first']:.1f}-{self.count}"
                events.append(event(self, frame, "crowd_start", severity="warning", incident=ep["key"], confidence=1.0,
                                    people=len(ids), members=sorted(ids), box=box))
        for ep in [e for e in self.episodes if frame.ts - e["last"] >= self.quiet_s]:
            self.episodes.remove(ep)
            if ep["alerted"]:
                events.append(event(self, frame, "crowd_end", severity="info", incident=ep["key"], confidence=1.0,
                                    duration_s=round(ep["last"] - ep["first"], 1), peak_people=ep["peak"]))
        return events
