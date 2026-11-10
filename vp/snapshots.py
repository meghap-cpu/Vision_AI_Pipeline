import time
import sqlite3
"""Saves event snapshots in a background thread. Each event draws its own evidence from standard fields:
people green (the event's people red), the event's box yellow, its zone blue, motion blobs orange."""
import os
import queue
import threading

import cv2

GREEN, RED, YELLOW, BLUE, ORANGE = (0, 200, 0), (0, 0, 255), (0, 255, 255), (255, 128, 0), (0, 140, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _px(box, w, h):
    return (int(box[0] * w), int(box[1] * h)), (int(box[2] * w), int(box[3] * h))


class Snapshots:
    errors = 0  # snapshots that failed (each one is skipped; the writer carries on)
    def __init__(self, outbox, folder):
        self.outbox, self.folder = outbox, folder
        os.makedirs(folder, exist_ok=True)
        self.q = queue.Queue(maxsize=32)
        self.dropped = 0
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, event_ids, rgba, objs, events):
        try:
            self.q.put_nowait((event_ids, rgba, objs, events))
        except queue.Full:
            self.dropped += 1

    def _run(self):
        """Never dies: a snapshot that fails is skipped, counted and logged; the next one is tried."""
        while True:
            item = self.q.get()
            try:
                self._write(*item)
            except Exception as e:
                self.errors += 1
                if self.errors == 1 or self.errors % 100 == 0:
                    print(f"snapshots: error {self.errors}: {type(e).__name__}: {str(e)[:150]}", flush=True)

    def _write(self, event_ids, rgba, objs, events):
        img = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        h, w = img.shape[:2]
        highlight = set()
        for ev in events:
            highlight.add(ev.get("track_id"))
            highlight.update(ev.get("extra", {}).get("members", []))
        for o in objs:
            p1, p2 = _px(o.box, w, h)
            if o.source != 1:  # boxes from other models (PPE items): thin, no track ID
                cv2.rectangle(img, p1, p2, (255, 200, 0), 1)
                continue
            color = RED if o.track_id in highlight else GREEN
            cv2.rectangle(img, p1, p2, color, 2)
            cv2.putText(img, str(o.track_id), (p1[0], max(12, p1[1] - 5)), FONT, 0.5, color, 1)
        for n, ev in enumerate(events):
            x = ev.get("extra", {})
            zone = x.get("zone")
            if zone:
                p1, p2 = _px(zone, w, h)
                cv2.rectangle(img, p1, p2, BLUE, 2)
                zw, zh = zone[2] - zone[0], zone[3] - zone[1]
                for b in x.get("boxes_in_zone", []):  # motion blobs, as fractions of the zone
                    fb = (zone[0] + b[0] * zw, zone[1] + b[1] * zh, zone[0] + b[2] * zw, zone[1] + b[3] * zh)
                    q1, q2 = _px(fb, w, h)
                    cv2.rectangle(img, q1, q2, ORANGE, 2)
            if x.get("box"):
                p1, p2 = _px(x["box"], w, h)
                cv2.rectangle(img, p1, p2, YELLOW, 3)
            label = f"{ev['use_case']}: {ev['class']}"
            cv2.putText(img, label, (10, 30 + 28 * n), FONT, 0.8, (0, 0, 0), 4)
            cv2.putText(img, label, (10, 30 + 28 * n), FONT, 0.8, YELLOW, 2)
        path = os.path.join(self.folder, f"{event_ids[0]}.jpg")
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        for attempt in range(5):  # the outbox may be busy for a moment: retry before giving up
            try:
                self.outbox.set_snapshot(event_ids, path)
                return
            except sqlite3.OperationalError:
                if attempt == 4:
                    raise
                time.sleep(1 + attempt)
