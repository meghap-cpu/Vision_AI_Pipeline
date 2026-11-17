"""Sends the outbox to the hub in two steps: events (marked sent once the hub accepts the batch), then photos on
their own. A photo the hub refuses (4xx) or whose file is gone is given up and counted, so it can never hold other
alerts back; a photo that arrives after its event is still sent. Network or hub errors retry with growing waits
(up to 10 s). The hub ignores an event_id it already has, so a retry never creates a duplicate."""
import json
import os
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone


class Uploader:
    def __init__(self, outbox, url, batch=50, timeout_s=10.0):
        self.outbox, self.url, self.batch, self.timeout = outbox, url.rstrip("/"), batch, timeout_s
        self.hub_ok, self.sent_total, self.photos_sent, self.photos_refused = None, 0, 0, 0
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

    def _send_events(self):
        rows = self.outbox.pending(self.batch)
        if rows:
            self._post("/api/v1/events", json.dumps([self._event(r) for r in rows]).encode(), "application/json")
            self.outbox.mark_sent([r["event_id"] for r in rows])
            self.sent_total += len(rows)
        return len(rows)

    def _send_photos(self, limit=20):
        photos = self.outbox.pending_photos(limit)
        for event_id, path in photos:
            if not path or not os.path.exists(path):
                self._give_up(event_id, "file missing")
                continue
            with open(path, "rb") as f:
                data = f.read()
            try:
                self._post(f"/api/v1/evidence/{event_id}", data, "image/jpeg")
            except urllib.error.HTTPError as e:
                if 400 <= e.code < 500:  # the hub will never take this one: give it up, don't block the rest
                    self._give_up(event_id, f"refused ({e.code})")
                    continue
                raise  # the hub is in trouble: try again later
            self.outbox.mark_photos([event_id], 1)
            self.photos_sent += 1
        return len(photos)

    def _give_up(self, event_id, why):
        self.outbox.mark_photos([event_id], 2)
        self.photos_refused += 1
        if self.photos_refused == 1 or self.photos_refused % 100 == 0:
            print(f"uploader: photo for {event_id} given up, {why} ({self.photos_refused} so far)", flush=True)

    def _run(self):
        wait = 1.0
        while True:
            busy = False
            try:
                busy = self._send_events() == self.batch
                busy = self._send_photos() == 20 or busy
                self._status(True)
                wait = 1.0
            except Exception as e:
                self._status(False, e)
                wait = min(wait * 2, 10.0)
            if not (self.hub_ok and busy):  # a full batch means more are waiting: keep going
                time.sleep(wait)
