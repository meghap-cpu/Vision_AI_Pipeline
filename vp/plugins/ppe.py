"""PPE: hardhat and safety vest per tracked person. Ported from the PPE app with fixes:
- one merged PPE model (Hardhat, No_Hardhat, Person, no_safety_vest, safety_vest) on the whole frame;
- PPE boxes are matched to tracked people from the shared person detector;
- a person with no head or vest box is UNKNOWN, never a violation (the app treated it as one);
- a decision needs several checks over a few seconds, so one wrong frame cannot raise or hide an alert;
- one alert per person per missing item.
"""
from collections import deque

from vp.plugin import Plugin, event

HEAD, BODY = "hardhat", "vest"


class PPEPlugin(Plugin):
    name = "ppe"

    def __init__(self, camera_id, params):
        super().__init__(camera_id, params)
        self.ppe_source = params.get("ppe_source", 3)  # gie-unique-id of the PPE model
        c = params.get("classes", {"Hardhat": 0, "No_Hardhat": 1, "no_safety_vest": 3, "safety_vest": 4})
        # PPE class -> (item, wearing it?)
        self.meaning = {c["Hardhat"]: (HEAD, True), c["No_Hardhat"]: (HEAD, False),
                        c["safety_vest"]: (BODY, True), c["no_safety_vest"]: (BODY, False)}
        self.window_s = params.get("window_s", 5.0)  # evidence older than this is dropped
        self.min_checks = params.get("min_checks", 4)  # checks needed before deciding
        self.violation_share = params.get("violation_share", 0.6)  # share of "not wearing" needed to alert
        self.forget_s = params.get("forget_s", 10.0)  # a person unseen this long is forgotten
        self.people = {}  # track_id -> {"last": ts, HEAD: deque, BODY: deque, "alerted": set()}

    @staticmethod
    def _owner(box, people, top, bottom):
        """The person whose box holds the PPE box's centre between top..bottom (fractions of its height).
        If several do, the smallest (usually the nearest) wins."""
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        best = None
        for p in people:
            x1, y1, x2, y2 = p.box
            if x1 <= cx <= x2 and y1 + top * (y2 - y1) <= cy <= y1 + bottom * (y2 - y1):
                area = (x2 - x1) * (y2 - y1)
                if best is None or area < best[0]:
                    best = (area, p)
        return best[1] if best else None

    def on_frame(self, frame):
        people = [o for o in frame.objects if o.source == 1 and o.class_id == 0]
        ppe = [o for o in frame.objects if o.source == self.ppe_source and o.class_id in self.meaning]
        for p in people:
            self.people.setdefault(p.track_id, {"last": frame.ts, HEAD: deque(), BODY: deque(), "alerted": set()})["last"] = frame.ts
        for o in ppe:
            item, wearing = self.meaning[o.class_id]
            top, bottom = (0.0, 0.35) if item == HEAD else (0.15, 0.75)  # head: top third; vest: middle
            owner = self._owner(o.box, people, top, bottom)
            if owner is not None:
                self.people[owner.track_id][item].append((frame.ts, wearing))
        events = []
        by_id = {p.track_id: p for p in people}
        for tid, st in list(self.people.items()):
            if frame.ts - st["last"] > self.forget_s:
                del self.people[tid]
                continue
            for item in (HEAD, BODY):
                checks = st[item]
                while checks and frame.ts - checks[0][0] > self.window_s:
                    checks.popleft()
                if item in st["alerted"] or len(checks) < self.min_checks:
                    continue
                share = sum(1 for _, wearing in checks if not wearing) / len(checks)
                if share >= self.violation_share and tid in by_id:
                    st["alerted"].add(item)
                    events.append(event(self, frame, f"no_{item}", severity="warning", incident=f"{tid}-{item}", confidence=round(share, 2),
                                        track_id=tid, box=tuple(round(v, 3) for v in by_id[tid].box),
                                        checks=len(checks), share_missing=round(share, 2)))
        return events
