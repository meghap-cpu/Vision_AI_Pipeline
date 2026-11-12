"""Fire and smoke, judged by the VLM. Ported from vlm-fire-smoke: 4 frames sampled 1 s apart go to the VLM with
the repo's question; 5 positive batches in a row raise the alert, 10 negative in a row end it.
Changed: the VLM call never blocks the video; unreadable replies are unknown, not crashes; one alert per episode
(start + end); a VLM that cannot be reached raises a warning instead of looking like "no fire"."""
from vp.plugin import Plugin, event
from vp.vlm import VLMClient

PROMPT = ('Analyze ALL these frames. Reply ONLY with this JSON, nothing else: '
          '{"fire": true or false, "smoke": true or false}')


class FireSmokePlugin(Plugin):
    name = "fire_smoke"
    needs_pixels, pixel_mode, pixels_every_s = True, "color", 1.0  # one colour frame per second
    uses_vlm = True

    def __init__(self, camera_id, params):
        super().__init__(camera_id, params)
        g = params.get
        self.zone = g("zone", [0, 0, 1, 1])
        self.batch_size = g("batch_size", 4)
        self.on_after, self.off_after = g("positive_batches", 5), g("negative_batches", 10)
        self.error_every_s = g("error_every_s", 60.0)
        v = g("vlm", {})
        self.client = VLMClient(v.get("url", "http://127.0.0.1:8000/v1"), v.get("model", "qwen3-vl"),
                                v.get("timeout_s", 30.0), max_in_flight=1)
        self.batch, self.pending = [], None
        self.pos = self.neg = 0
        self.active, self.started, self.last_error = False, None, -1e9

    def on_frame(self, frame):
        events = []
        if frame.pixels is not None:
            self.batch = (self.batch + [frame.pixels])[-self.batch_size:]  # newest frames if the VLM is busy
        if self.pending is not None and self.pending.done():
            events += self._result(frame)
            self.pending = None
        if self.pending is None and len(self.batch) >= self.batch_size:
            self.pending, self.batch = self.client.ask(self.batch, PROMPT), []
        return events

    def _result(self, frame):
        try:
            r = self.pending.result()
        except Exception as e:  # VLM unreachable or failed: say so, at most once per error_every_s
            if frame.ts - self.last_error >= self.error_every_s:
                self.last_error = frame.ts
                return [event(self, frame, "vlm_error", severity="info", error=str(e)[:200])]  # alert: vlm_down
            return []
        ans = r["answer"]
        if not isinstance(ans, dict):
            return []  # unreadable reply: unknown, counts neither way
        fire, smoke = bool(ans.get("fire")), bool(ans.get("smoke"))
        if fire or smoke:
            self.pos, self.neg = self.pos + 1, 0
        else:
            self.pos, self.neg = 0, self.neg + 1
        if not self.active and self.pos >= self.on_after:
            self.active, self.started = True, frame.ts
            return [event(self, frame, "fire_smoke_start", severity="critical", incident=f"{self.started:.1f}", confidence=1.0,
                          fire=fire, smoke=smoke, zone=self.zone, vlm_seconds=r["seconds"])]
        if self.active and self.neg >= self.off_after:
            self.active = False
            return [event(self, frame, "fire_smoke_end", severity="info", incident=f"{self.started:.1f}", confidence=1.0,
                          duration_s=round(frame.ts - self.started, 1))]
        return []
