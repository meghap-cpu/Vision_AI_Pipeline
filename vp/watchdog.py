"""Raises camera_offline / camera_online health events, so a dead camera is never silent."""
import time


class CameraWatchdog:
    def __init__(self, camera_ids, outbox, offline_after_s=15.0, startup_grace_s=60.0, steady_s=10.0, clock=time.time):
        self.ids, self.outbox, self.offline_after = camera_ids, outbox, offline_after_s
        self.startup_grace, self.steady, self.clock = startup_grace_s, steady_s, clock
        self.started = now = clock()
        self.last_seen = [now] * len(camera_ids)
        self.offline_since = [None] * len(camera_ids)
        self.back_since = [None] * len(camera_ids)  # an offline camera's video came back at: online after steady_s

    def seen(self, i):
        # Called from the video thread: one number written, nothing more.
        self.last_seen[i] = self.clock()

    def check(self):
        """Called from the main loop; stores and returns any state changes."""
        now, events = self.clock(), []
        # at start-up cameras connect one by one (~3 s each): only the grace period counts as silence until then
        limit = self.startup_grace if now - self.started < self.startup_grace else self.offline_after
        for i, cam in enumerate(self.ids):
            silent = now - self.last_seen[i]
            if self.offline_since[i] is None:
                if silent > limit:
                    self.offline_since[i], self.back_since[i] = self.last_seen[i], None
                    events.append(self._event(cam, "camera_offline", "warning", now,
                                              f"offline:{int(self.offline_since[i])}", silent_s=round(silent, 1)))
            elif silent < 2.0:  # video again: online only once it has been steady for steady_s
                if self.back_since[i] is None:
                    self.back_since[i] = now
                elif now - self.back_since[i] >= self.steady:
                    since, down = self.offline_since[i], round(self.back_since[i] - self.offline_since[i], 1)
                    self.offline_since[i] = self.back_since[i] = None
                    events.append(self._event(cam, "camera_online", "info", now, f"offline:{int(since)}",
                                              offline_for_s=down))
            else:
                self.back_since[i] = None  # a blip: still the same offline incident
        for ev in events:
            self.outbox.add(ev)
        return events

    @staticmethod
    def _event(cam, cls, severity, ts, incident, **extra):
        return {"use_case": "health", "camera_id": cam, "ts": ts, "class": cls, "severity": severity, "incident": incident,
                "confidence": 1.0, "track_id": None, "extra": extra}
