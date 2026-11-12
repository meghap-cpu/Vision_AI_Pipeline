"""Keeps the node's storage bounded. Every few minutes:
- deletes snapshots the hub already has, once older than snapshot_keep_hours;
- if the snapshot folder is still over snapshot_max_gb, deletes the oldest snapshots the hub already has;
- deletes sent events from the outbox after outbox_keep_days;
- warns once when free disk drops under disk_warn_free_gb (disk_low) or only undelivered snapshots exceed the cap
  (storage_full), and sends disk_ok / storage_ok with the same incident when it recovers, which closes it.
Anything not yet delivered to the hub is never deleted."""
import os
import shutil
import threading
import time


class Retention:
    def __init__(self, outbox, snapshot_dir, settings, every_s=300):
        g = settings.get
        self.outbox, self.dir, self.every = outbox, snapshot_dir, every_s
        self.keep_s = g("snapshot_keep_hours", 24) * 3600
        self.max_bytes = g("snapshot_max_gb", 2.0) * 1e9
        self.outbox_keep_s = g("outbox_keep_days", 7) * 86400
        self.warn_free = g("disk_warn_free_gb", 2.0) * 1e9
        self.open, self.stats = {}, {"snapshots_deleted": 0, "events_pruned": 0}  # open: problem -> since when
        if every_s:
            print(f"retention: snapshots {g('snapshot_keep_hours', 24)} h / {g('snapshot_max_gb', 2.0)} GB, "
                  f"sent events {g('outbox_keep_days', 7)} days; undelivered data is never deleted", flush=True)
            threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            try:
                self.run_once()
            except Exception as e:
                print(f"retention: {e}", flush=True)
            time.sleep(self.every)

    def _folder_bytes(self):
        if not os.path.isdir(self.dir):
            return 0
        return sum(e.stat().st_size for e in os.scandir(self.dir) if e.is_file())

    def _delete(self, paths):
        for p in paths:
            try:
                os.remove(p)
            except FileNotFoundError:
                pass
        self.outbox.forget_snapshots(paths)
        self.stats["snapshots_deleted"] += len(paths)

    def _condition(self, cls, bad, now, ok_cls, **extra):
        """One warning when a problem starts, one info (ok_cls) when it ends, both in the same incident."""
        since = self.open.get(cls)
        if bad and since is None:
            self.open[cls] = now
            self.outbox.add({"camera_id": "node", "use_case": "health", "class": cls, "severity": "warning",
                             "ts": now, "incident": f"{cls}:{int(now)}", "extra": extra})
        elif not bad and since is not None:
            del self.open[cls]
            self.outbox.add({"camera_id": "node", "use_case": "health", "class": ok_cls, "severity": "info",
                             "ts": now, "incident": f"{cls}:{int(since)}", "extra": {**extra, "lasted_s": round(now - since)}})

    def run_once(self, now=None):
        now = now or time.time()
        self._delete(self.outbox.delivered_snapshots(older_than=now - self.keep_s))   # old and delivered
        size = self._folder_bytes()
        if size > self.max_bytes:
            doomed = []
            for path in self.outbox.delivered_snapshots(older_than=float("inf")):   # every delivered one, oldest first
                if size <= self.max_bytes:
                    break
                try:
                    size -= os.path.getsize(path)
                except FileNotFoundError:
                    pass
                doomed.append(path)
            self._delete(doomed)  # one batch: the outbox is busy for a moment, not for thousands of updates
        full = size > self.max_bytes
        self._condition("storage_full", full, now, "storage_ok", snapshot_gb=round(size / 1e9, 3),
                        **({"note": "only undelivered snapshots left: is the hub reachable?"} if full else {}))
        self.stats["events_pruned"] += self.outbox.prune_sent(older_than=now - self.outbox_keep_s)
        free = shutil.disk_usage(self.dir if os.path.isdir(self.dir) else "/").free
        margin = 0.5e9 if "disk_low" in self.open else 0  # close only once 0.5 GB past the line: no flapping
        self._condition("disk_low", free < self.warn_free + margin, now, "disk_ok", free_gb=round(free / 1e9, 2))
        return dict(self.stats)
