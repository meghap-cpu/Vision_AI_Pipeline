"""Raises camera_offline / camera_online health events, so a dead camera is never silent."""
import time


class CameraWatchdog:
    def __init__(self, camera_ids, outbox, offline_after_s=15.0):
        self.ids, self.outbox, self.offline_after = camera_ids, outbox, offline_after_s
        now = time.time()
        self.last_seen = [now] * len(camera_ids)  # counts as a grace period at startup
        self.offline_since = [None] * len(camera_ids)

    def seen(self, i):
        # Called from the video thread: one number written, nothing more.
        self.last_seen[i] = time.time()

    def check(self):
        """Called from the main loop; stores and returns any state changes."""
        now, events = time.time(), []
        for i, cam in enumerate(self.ids):
            silent = now - self.last_seen[i]
            if self.offline_since[i] is None and silent > self.offline_after:
                self.offline_since[i] = self.last_seen[i]
                events.append(self._event(cam, "camera_offline", "warning", now, silent_s=round(silent, 1)))
            elif self.offline_since[i] is not None and silent < 2.0:
                down = round(self.last_seen[i] - self.offline_since[i], 1)
                self.offline_since[i] = None
                events.append(self._event(cam, "camera_online", "info", now, offline_for_s=down))
        for ev in events:
            self.outbox.add(ev)
        return events

    @staticmethod
    def _event(cam, cls, severity, ts, **extra):
        return {"use_case": "health", "camera_id": cam, "ts": ts, "class": cls, "severity": severity,
                "confidence": 1.0, "track_id": None, "extra": extra}
