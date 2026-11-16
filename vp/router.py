"""The probe's decisions, kept out of DeepStream's video thread so they can be tested without a camera or GPU.

Router: which use cases get a camera's frame, which of them need pixels right now, and frames dropped per use case.
SnapshotRequests: photos wanted per camera; a request older than max_age_s expires (an offline camera must never
  attach a photo taken minutes later, of a different scene).
CropStore: the enlarged crops pose ran on, per (camera, frame, object); entries older than keep_frames frames of their
  camera are removed, so nothing the probe never reads can pile up.
All three are safe to use from several threads."""
import threading
import time
from collections import Counter


class Router:
    def __init__(self, routes, pixel_zones):
        self.routes = routes  # per camera index: [use case, ...]
        self.pixel_zones = pixel_zones  # per camera index: {use case: (zone, mode, every_s, max_width)}
        self.last_pixels = {}  # (camera index, use case) -> when pixels were last sent
        self.dropped = Counter()  # use case -> frames dropped because its worker was full
        self._lock = threading.Lock()

    def plan(self, i, ts):
        """[(use case, pixel spec or None)] for camera i at time ts; marks the pixels it hands out as sent."""
        out = []
        with self._lock:
            for uc in self.routes[i]:
                spec = self.pixel_zones[i].get(uc)
                if spec is not None and ts - self.last_pixels.get((i, uc), -1e9) >= spec[2]:
                    self.last_pixels[(i, uc)] = ts
                    out.append((uc, spec))
                else:
                    out.append((uc, None))
        return out

    def drop(self, uc):
        with self._lock:
            self.dropped[uc] += 1

    def dropped_by_use_case(self):
        with self._lock:
            return dict(self.dropped)


class SnapshotRequests:
    def __init__(self, max_age_s=5.0, clock=time.time):
        self.max_age, self.clock = max_age_s, clock
        self.items, self.expired = {}, 0  # camera_id -> [(asked at, event_id, event)]
        self._lock = threading.Lock()

    def add(self, camera_id, event_id, ev):
        with self._lock:
            self.items.setdefault(camera_id, []).append((self.clock(), event_id, ev))

    def take(self, camera_id):
        """This camera's fresh requests as [(event_id, event)], removed; stale ones are dropped and counted."""
        with self._lock:
            reqs = self.items.pop(camera_id, None)
            if not reqs:
                return []
            now = self.clock()
            fresh = [(e, ev) for t, e, ev in reqs if now - t <= self.max_age]
            self.expired += len(reqs) - len(fresh)
            return fresh

    def prune(self):
        """Drops stale requests of cameras that send no frames (called from the main loop)."""
        with self._lock:
            now = self.clock()
            for cam in list(self.items):
                fresh = [r for r in self.items[cam] if now - r[0] <= self.max_age]
                self.expired += len(self.items[cam]) - len(fresh)
                if fresh:
                    self.items[cam] = fresh
                else:
                    del self.items[cam]


class CropStore:
    def __init__(self, keep_frames=50):
        self.keep = keep_frames
        self.items, self.latest, self.pruned = {}, {}, 0  # key -> value; camera -> newest frame number seen
        self._lock = threading.Lock()

    def put(self, key, value):
        cam, frame, _ = key
        with self._lock:
            self.items[key] = value
            if frame > self.latest.get(cam, -1):
                self.latest[cam] = frame
                old = [k for k in self.items if k[0] == cam and k[1] < frame - self.keep]
                for k in old:
                    del self.items[k]
                self.pruned += len(old)

    def get(self, key):
        with self._lock:
            return self.items.get(key)

    def pop(self, key):
        with self._lock:
            return self.items.pop(key, None)

    def __len__(self):
        with self._lock:
            return len(self.items)
