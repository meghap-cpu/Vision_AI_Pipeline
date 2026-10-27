"""Crowd detection. Ported from crowd_detection: DBSCAN on feet points, crowd identity by Jaccard
overlap of track IDs. Changed: one alert per crowd (start + end), a short confirmation time, and
all timing from frame timestamps."""
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


class CrowdPlugin(Plugin):
    name = "crowd"

    def __init__(self, camera_id, params):
        super().__init__(camera_id, params)
        self.eps = params.get("eps", 0.234)  # group radius as a share of frame width (300 px at 1280)
        self.min_people = params.get("min_people", 4)
        self.aspect = params.get("aspect", 720 / 1280)  # frame height / width
        self.confirm_s = params.get("confirm_s", 2.0)  # a crowd must hold this long before alerting
        self.quiet_s = params.get("quiet_s", 10.0)  # gone this long = crowd over
        self.match = params.get("match_jaccard", 0.5)  # this much ID overlap = the same crowd
        self.episodes = []

    def on_frame(self, frame):
        people = [o for o in frame.objects if o.source == 1 and o.class_id == 0]
        feet = [((o.box[0] + o.box[2]) / 2, o.box[3] * self.aspect) for o in people]
        events = []
        for idx in clusters(feet, self.eps, self.min_people):
            ids = {people[i].track_id for i in idx}
            box = tuple(round(f(people[i].box[k] for i in idx), 3) for k, f in ((0, min), (1, min), (2, max), (3, max)))
            ep = max(self.episodes, key=lambda e: jaccard(e["members"], ids), default=None)
            if ep is None or jaccard(ep["members"], ids) < self.match:
                ep = {"first": frame.ts, "alerted": False, "peak": 0}
                self.episodes.append(ep)
            ep.update(members=ids, last=frame.ts, box=box, peak=max(ep["peak"], len(ids)))
            if not ep["alerted"] and frame.ts - ep["first"] >= self.confirm_s:
                ep["alerted"] = True
                events.append(event(self, frame, "crowd_start", severity="warning", incident=f"{ep['first']:.1f}", confidence=1.0,
                                    people=len(ids), members=sorted(ids), box=box))
        for ep in [e for e in self.episodes if frame.ts - e["last"] >= self.quiet_s]:
            self.episodes.remove(ep)
            if ep["alerted"]:
                events.append(event(self, frame, "crowd_end", severity="info", incident=f"{ep['first']:.1f}", confidence=1.0,
                                    duration_s=round(ep["last"] - ep["first"], 1), peak_people=ep["peak"]))
        return events
