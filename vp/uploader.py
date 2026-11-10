"""Sends the outbox to the hub: events first, then their snapshots, marked sent only once accepted.
Retries with growing waits (up to 30 s) while the hub is unreachable. The hub ignores an event_id it already
has, so a retry never creates a duplicate."""
import json
import os
import threading
import time
import urllib.request
from datetime import datetime, timezone


class Uploader:
    def __init__(self, outbox, url, batch=50, timeout_s=10.0):
        self.outbox, self.url, self.batch, self.timeout = outbox, url.rstrip("/"), batch, timeout_s
        self.hub_ok, self.sent_total = None, 0
        threading.Thread(target=self._run, daemon=True).start()

    def _post(self, path, data, content_type):
        req = urllib.request.Request(self.url + path, data=data, headers={"Content-Type": content_type})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return r.status

    @staticmethod
    def _event(r):
        when = r["ts"] or r["created"]
        return {"event_id": r["event_id"], "incident_id": r["incident_id"],
                "timestamp": datetime.fromtimestamp(when, timezone.utc).astimezone().isoformat(timespec="seconds"),
                "site_id": r["site_id"], "node_id": r["node_id"], "camera_id": r["camera_id"],
                "use_case": r["use_case"], "class": r["class"], "severity": r["severity"],
                "confidence": r["confidence"], "track_id": r["track_id"],
                "rule": r.get("rule"), "instance": r.get("instance"), "model": r.get("model"),
                "snapshot_ref": r["event_id"] if r["snapshot"] else None, "extra": json.loads(r["extra"] or "{}")}

    def _status(self, ok, err=None):
        if ok != self.hub_ok:
            if ok:
                print(f"uploader: hub reachable at {self.url}", flush=True)
            else:
                print(f"uploader: hub unreachable ({str(err)[:80]}); keeping {self.outbox.unsent()} events", flush=True)
            self.hub_ok = ok

    def _run(self):
        wait = 1.0
        while True:
            rows = []
            try:
                rows = self.outbox.pending(self.batch)
                if rows:
                    self._post("/api/v1/events", json.dumps([self._event(r) for r in rows]).encode(), "application/json")
                    for r in rows:
                        if r["snapshot"] and os.path.exists(r["snapshot"]):
                            with open(r["snapshot"], "rb") as f:
                                self._post(f"/api/v1/evidence/{r['event_id']}", f.read(), "image/jpeg")
                    self.outbox.mark_sent([r["event_id"] for r in rows])
                    self.sent_total += len(rows)
                self._status(True)
                wait = 1.0
            except Exception as e:
                self._status(False, e)
                wait = min(wait * 2, 10.0)
            if not (self.hub_ok and len(rows) == self.batch):  # a full batch means more are waiting: keep going
                time.sleep(wait)
