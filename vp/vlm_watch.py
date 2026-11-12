"""Raises vlm_down / vlm_up for the node, so a VLM outage is one alert instead of one warning per camera.

Every every_s it asks the VLM server "are you there?" (GET <url>/models, which vLLM and the stand-in answer).
After fails_to_alert failed checks in a row: vlm_down (warning). When it answers again: vlm_up (info), same
incident, which closes it. Runs in its own thread, so a slow or dead VLM never holds up the main loop.
"""
import threading
import time
import urllib.request
from urllib.parse import urlsplit


class VLMWatch:
    def __init__(self, url, outbox, every_s=30.0, fails_to_alert=2, timeout_s=5.0):
        self.url = url.rstrip("/") + "/models"
        self.server = urlsplit(url).netloc  # host:port, names the VLM server in alerts
        self.outbox, self.every, self.fails_to_alert, self.timeout = outbox, every_s, fails_to_alert, timeout_s
        self.fails, self.down_since = 0, None

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()
        print(f"vlm watch: checking {self.url} every {self.every:.0f} s", flush=True)

    def _loop(self):
        while True:
            for ev in self.check():
                print(f"health: node {ev['class']} {ev['extra']}", flush=True)
            time.sleep(self.every)

    def reachable(self):
        try:
            with urllib.request.urlopen(self.url, timeout=self.timeout) as r:
                return r.status == 200
        except Exception:
            return False

    def check(self, now=None):
        """One check; stores and returns any state change (at most one event)."""
        now = time.time() if now is None else now
        events = []
        if self.reachable():
            self.fails = 0
            if self.down_since is not None:
                events.append(self._event("vlm_up", "info", now, self.down_since, server=self.server,
                                          down_for_s=round(now - self.down_since)))
                self.down_since = None
        else:
            self.fails += 1
            if self.down_since is None and self.fails >= self.fails_to_alert:
                self.down_since = now
                events.append(self._event("vlm_down", "warning", now, now, server=self.server, url=self.url,
                                          failed_checks=self.fails))
        for ev in events:
            self.outbox.add(ev)
        return events

    def _event(self, cls, severity, ts, since, **extra):
        return {"use_case": "health", "camera_id": "node", "ts": ts, "class": cls, "severity": severity,
                "incident": f"vlm:{self.server}:down:{int(since)}", "confidence": 1.0, "track_id": None, "extra": extra}
