"""Man-down. Per-frame scoring ported faithfully from Mandown_Detection (core/skeleton.py, core/decision.py),
with its tuned weights and thresholds from config/config.yaml. Timing differs on purpose: seconds from frame
timestamps (the repo counted frames at 25 fps) with hysteresis on the smoothed score, as decision.py intends.
Joints (COCO-17): 0 nose, 5/6 shoulders, 11/12 hips, 13/14 knees, 15/16 ankles."""
from vp.plugin import Plugin, event
import math

PAIRS = ((5, 6), (11, 12), (13, 14), (15, 16))  # shoulders, hips, knees, ankles (symmetry)


def ramp(v, lo, hi):
    return min(1.0, max(0.0, (v - lo) / (hi - lo)))


class ManDownPlugin(Plugin):
    name = "man_down"

    def __init__(self, camera_id, params):
        super().__init__(camera_id, params)
        g = params.get
        self.fw, self.fh = g("frame_width", 1280), g("frame_height", 720)   # frame size in pixels
        # guards, as in the repo's code
        self.min_conf = g("min_detection_conf", 0.45)
        self.min_box_px = g("min_bbox_height_px", 80)        # repo code: bbox under 80 px is uncertain
        self.kp_guard = g("min_kp_visible_score", 0.25)      # hips / legs visible
        self.kp_vis = g("kp_visible_score", 0.30)            # repo skeleton.py _MIN_VIS
        self.min_kp = g("min_keypoints", 9)
        # scoring, repo values
        self.angle = (g("angle_upright_max", 40.0), g("angle_fallen_min", 78.0))
        self.box = (g("aspect_upright_max", 0.85), g("aspect_fallen_min", 1.30))
        self.head_fallen = g("head_height_fallen_min", 0.65)
        # "binary": repo rule, hips within 15% of the box height of the ankles = 1, else 0 (missed a side fall).
        # "graded" (provisional default): full credit at 15%, falling to 0 at 45%. Re-check on labelled clips.
        self.lower_mode = g("lower_body", "graded")
        self.weights = dict(angle=g("weight_angle", 0.35), box=g("weight_aspect", 0.10), head=g("weight_head", 0.15),
                            lower=g("weight_lower_body", 0.30), symmetry=g("weight_symmetry", 0.10))
        self.fallen_at, self.crouch_at = g("fallen_score_threshold", 0.72), g("crouching_score_threshold", 0.50)
        self.alpha = g("score_ema_alpha", 0.20)
        # timing in seconds (repo frames at 25 fps: dwell 150, recovery 75, unseen 60)
        self.dwell_s, self.recovery_s = g("dwell_s", 6.0), g("recovery_s", 3.0)
        self.forget_s, self.cooldown_s = g("forget_s", 2.4), g("cooldown_s", 30.0)
        self.people = {}
        self.debug_s = g("debug_every_s", 0)  # diagnostic only; 0 = off
        self.stats, self.last_debug = {"scored": 0, "unknown": 0, "max_score": 0.0}, None

    def score(self, o):
        """Repo decision.py: weighted score 0..1 with its signals, or None when uncertain."""
        kp = o.keypoints
        if kp is None or 0 <= o.confidence < self.min_conf:
            return None
        x1, y1, x2, y2 = o.box[0] * self.fw, o.box[1] * self.fh, o.box[2] * self.fw, o.box[3] * self.fh
        bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        if bh < self.min_box_px:
            return None                                                  # bbox too small
        pt = [(k[0] * self.fw, k[1] * self.fh, k[2]) for k in kp]
        if not (pt[11][2] >= self.kp_guard or pt[12][2] >= self.kp_guard):
            return None                                                  # hips not visible
        if not any(pt[i][2] >= self.kp_guard for i in (13, 14, 15, 16)):
            return None                                                  # legs not visible
        if bh / bw > 1.8:
            return 0.0, {"tall_box": 1.0}                               # clearly standing or sitting
        if sum(1 for p in pt if p[2] >= self.kp_vis) < self.min_kp:
            return None
        vis = lambda i: pt[i][2] >= self.kp_vis

        def mean_y(a, b):
            ys = [pt[i][1] for i in (a, b) if vis(i)]
            return sum(ys) / len(ys) if ys else None

        s = {}
        if all(vis(i) for i in (5, 6, 11, 12)):                          # spine: shoulder midpoint -> hip midpoint
            dx = (pt[11][0] + pt[12][0] - pt[5][0] - pt[6][0]) / 2
            dy = (pt[11][1] + pt[12][1] - pt[5][1] - pt[6][1]) / 2
            if abs(dx) > 1e-3 or abs(dy) > 1e-3:
                s["angle"] = ramp(math.degrees(math.atan2(abs(dx), abs(dy))), *self.angle)
        s["box"] = ramp(bw / bh, *self.box)
        if vis(0):                                                       # nose position down the box
            s["head"] = ramp((pt[0][1] - y1) / bh, 0.2, self.head_fallen)
        hip_y, ankle_y = mean_y(11, 12), mean_y(15, 16)
        if hip_y is not None and ankle_y is not None:                    # hips at ankle level?
            drop = (ankle_y - hip_y) / bh
            s["lower"] = ((1.0 if drop < 0.15 else 0.0) if self.lower_mode == "binary"
                          else 1.0 - ramp(drop, 0.15, 0.45))
        diffs = [abs(pt[a][1] - pt[b][1]) / bh for a, b in PAIRS if vis(a) and vis(b)]
        if len(diffs) >= 2:
            s["symmetry"] = min(sum(diffs) / len(diffs) / 0.4, 1.0)
        active = sum(self.weights[k] for k in s)
        if active < 0.3 * sum(self.weights.values()):
            return None                                                  # insufficient evidence
        final = sum(self.weights[k] * v for k, v in s.items()) / active
        if vis(0) and hip_y is not None and pt[0][1] > hip_y:            # head below hips: override
            final = max(final, 0.85)
            s["head_below_hips"] = 1.0
        return final, s

    def on_frame(self, frame):
        events = []
        for o in frame.objects:
            if o.source != 1 or o.class_id != 0:
                continue
            st = self.people.setdefault(o.track_id, {"ema": 0.0, "valid": frame.ts, "down": None, "up": None,
                                                     "alerted": False, "last_alert": -1e9})
            result = self.score(o)
            if result is None:
                self.stats["unknown"] += 1
                continue  # uncertain: neither counts toward falling nor toward recovering
            self.stats["scored"] += 1
            st["valid"] = frame.ts
            st["ema"] = self.alpha * result[0] + (1 - self.alpha) * st["ema"]
            if st["ema"] >= self.stats["max_score"]:
                self.stats["max_score"] = round(st["ema"], 2)
                self.stats["signals"] = {k: round(v, 2) for k, v in result[1].items()}
            if st["ema"] >= self.fallen_at:
                st["up"] = None
                st["down"] = st["down"] or frame.ts
                if (not st["alerted"] and frame.ts - st["down"] >= self.dwell_s
                        and frame.ts - st["last_alert"] >= self.cooldown_s):
                    st["alerted"], st["last_alert"] = True, frame.ts
                    events.append(event(self, frame, "man_down", severity="critical", incident=f"{o.track_id}@{st['last_alert']:.1f}", confidence=round(st["ema"], 2),
                                        track_id=o.track_id, box=tuple(round(v, 3) for v in o.box),
                                        down_s=round(frame.ts - st["down"], 1),
                                        signals={k: round(v, 2) for k, v in result[1].items()}))
            elif st["ema"] < self.crouch_at:
                st["down"] = None
                if st["alerted"]:
                    st["up"] = st["up"] or frame.ts
                    if frame.ts - st["up"] >= self.recovery_s:
                        st["alerted"], st["up"] = False, None
                        events.append(event(self, frame, "man_down_end", severity="info", incident=f"{o.track_id}@{st['last_alert']:.1f}", confidence=1.0,
                                            track_id=o.track_id, box=tuple(round(v, 3) for v in o.box)))
            # Crouching band (0.50-0.72): never starts a fall (repo: crouching_as_fallen false), but someone
            # already down stays down through small wobbles until clearly upright (below 0.50): hysteresis.
        for tid in [t for t, st in self.people.items() if frame.ts - st["valid"] > self.forget_s and not st["alerted"]]:
            del self.people[tid]
        if self.debug_s and (self.last_debug is None or frame.ts - self.last_debug >= self.debug_s):
            if self.last_debug is not None:
                events.append(event(self, frame, "pose_stats", severity="info", **self.stats))
            self.last_debug, self.stats = frame.ts, {"scored": 0, "unknown": 0, "max_score": 0.0}
        return events
