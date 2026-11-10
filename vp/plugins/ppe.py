"""PPE: hardhat and safety vest per tracked person, following CampNeuron's rules. Each item can be its own rule
(items: {hardhat: {...}, vest: {...}}) with its own rule name, place (instance), zones, schedule, severity, duration
and cooldown; without items, both are checked under the camera's rule. A person in the zone without an item for
duration_s raises an alert; still without it after cooldown_s, it is raised again (same incident).
- one merged PPE model (Hardhat, No_Hardhat, Person, no_safety_vest, safety_vest) on the whole frame;
- PPE boxes are matched to tracked people from the shared person detector;
- no head or vest box = UNKNOWN, never a violation; "missing" needs stability_frames [N, M] (N of the last M checks)
  or, without it, violation_share of the checks in the last window_s;
- min_confidence (per camera calibration) and min_person_h (min object size) from the detection type."""
from collections import deque

import cv2
import numpy as np

from vp.plugin import Plugin, event
from vp.schedule import Schedule

HEAD, BODY = "hardhat", "vest"


class PPEPlugin(Plugin):
    name = "ppe"

    def __init__(self, camera_id, params):
        super().__init__(camera_id, params)
        g = params.get
        self.ppe_source = g("ppe_source", 3)
        c = g("classes", {"Hardhat": 0, "No_Hardhat": 1, "no_safety_vest": 3, "safety_vest": 4})
        self.meaning = {c["Hardhat"]: (HEAD, True), c["No_Hardhat"]: (HEAD, False),
                        c["safety_vest"]: (BODY, True), c["no_safety_vest"]: (BODY, False)}
        self.min_conf = g("min_confidence", 0.4)
        self.min_person_h = g("min_person_h", 0.0)
        self.stability = g("stability_frames")  # [N, M]
        self.window_s, self.min_checks = g("window_s", 5.0), g("min_checks", 4)
        self.violation_share, self.forget_s = g("violation_share", 0.6), g("forget_s", 10.0)
        items = g("items")
        cfg = {i: (items[i] if items else {}) for i in (HEAD, BODY) if not items or i in items}
        self.items = {}
        for item, ic in cfg.items():
            self.items[item] = {
                "duration_s": ic.get("duration_s", g("duration_s", 10.0)),
                "cooldown_s": ic.get("cooldown_s", g("cooldown_s", 300.0)),
                "zones": [np.array(z, np.float32) for z in ic.get("zones", g("zones", []))],
                "schedule": Schedule(ic.get("schedule"), ic.get("timezone")),
                "severity": ic.get("severity", "warning"),
                "tags": {k: ic[k] for k in ("rule", "instance") if ic.get(k)}}
        self.people = {}

    @staticmethod
    def _owner(box, people, top, bottom):
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        best = None
        for p in people:
            x1, y1, x2, y2 = p.box
            if x1 <= cx <= x2 and y1 + top * (y2 - y1) <= cy <= y1 + bottom * (y2 - y1):
                area = (x2 - x1) * (y2 - y1)
                if best is None or area < best[0]:
                    best = (area, p)
        return best[1] if best else None

    @staticmethod
    def _in(zones, box):
        if not zones:
            return True
        feet = ((box[0] + box[2]) / 2, box[3])
        return any(cv2.pointPolygonTest(z, feet, False) >= 0 for z in zones)

    def _missing(self, checks):
        """True / False, or None when there isn't enough evidence (the timer neither starts nor stops)."""
        if self.stability:
            n, m = self.stability
            recent = list(checks)[-m:]
            return None if len(recent) < m else sum(1 for c in recent if not c[1]) >= n
        if len(checks) < self.min_checks:
            return None
        return sum(1 for c in checks if not c[1]) / len(checks) >= self.violation_share

    def on_frame(self, frame):
        people = [o for o in frame.objects if o.source == 1 and o.class_id == 0
                  and (o.box[3] - o.box[1]) >= self.min_person_h]
        ppe = [o for o in frame.objects if o.source == self.ppe_source and o.class_id in self.meaning
               and o.confidence >= self.min_conf]
        for p in people:
            st = self.people.setdefault(p.track_id, {"last": frame.ts, HEAD: deque(maxlen=200), BODY: deque(maxlen=200),
                                                     "since": {}, "alerted": {}})
            st["last"] = frame.ts
        for o in ppe:
            item, wearing = self.meaning[o.class_id]
            if item not in self.items:
                continue
            top, bottom = (0.0, 0.35) if item == HEAD else (0.15, 0.75)
            owner = self._owner(o.box, people, top, bottom)
            if owner is not None:
                self.people[owner.track_id][item].append((frame.ts, wearing, o.confidence))
        events, by_id = [], {p.track_id: p for p in people}
        for tid, st in list(self.people.items()):
            if frame.ts - st["last"] > self.forget_s:
                del self.people[tid]
                continue
            for item, ic in self.items.items():
                checks = st[item]
                while checks and frame.ts - checks[0][0] > self.window_s:
                    checks.popleft()
                if tid not in by_id or not ic["schedule"].active(frame.ts) or not self._in(ic["zones"], by_id[tid].box):
                    st["since"].pop(item, None)  # not visible, outside its hours or outside its zone: timer stops
                    continue
                missing = self._missing(checks)
                if missing is None:
                    continue
                if not missing:
                    st["since"].pop(item, None)
                    continue
                since = st["since"].setdefault(item, frame.ts)
                last = st["alerted"].get(item)
                if frame.ts - since >= ic["duration_s"] and (last is None or frame.ts - last >= ic["cooldown_s"]):
                    st["alerted"][item] = frame.ts
                    bad = [c for c in checks if not c[1]] or list(checks)
                    ev = event(self, frame, f"no_{item}", severity=ic["severity"], incident=f"{tid}-{item}",
                               confidence=round(sum(c[2] for c in bad) / len(bad), 2), track_id=tid,
                               box=tuple(round(v, 3) for v in by_id[tid].box), missing_s=round(frame.ts - since, 1),
                               checks=len(checks))
                    ev.update(ic["tags"])
                    events.append(ev)
        return events
