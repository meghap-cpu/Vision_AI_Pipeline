"""Local store-and-forward queue for events. Survives restarts; the uploader sends it to the hub."""
import json
import sqlite3
import threading
import time
import uuid

COLUMNS = ("event_id TEXT PRIMARY KEY, created REAL, node_id TEXT, site_id TEXT, camera_id TEXT, use_case TEXT, "
           "class TEXT, severity TEXT, confidence REAL, track_id INTEGER, ts REAL, extra TEXT, snapshot TEXT, "
           "sent INTEGER DEFAULT 0, incident_id TEXT")
FIELDS = ("event_id", "created", "node_id", "site_id", "camera_id", "use_case", "class", "severity", "confidence",
          "track_id", "ts", "extra", "snapshot", "incident_id")


class Outbox:
    def __init__(self, path, node_id, site_id):
        self.path, self.node_id, self.site_id = path, node_id, site_id
        self._local = threading.local()  # one SQLite connection per thread
        db = self._db()
        db.execute("PRAGMA journal_mode=WAL")
        with db:
            db.execute(f"CREATE TABLE IF NOT EXISTS events ({COLUMNS})")
            if "incident_id" not in {r[1] for r in db.execute("PRAGMA table_info(events)")}:
                db.execute("ALTER TABLE events ADD COLUMN incident_id TEXT")  # outbox from an older version

    def _db(self):
        if not hasattr(self._local, "db"):
            self._local.db = sqlite3.connect(self.path, timeout=5)
        return self._local.db

    def add(self, ev):
        event_id = str(uuid.uuid4())
        key = ev.get("incident")
        incident_id = f"{self.node_id}:{ev['camera_id']}:{ev['use_case']}:{key}" if key else event_id
        with self._db() as db:
            db.execute("INSERT INTO events (event_id, created, node_id, site_id, camera_id, use_case, class, severity, "
                       "confidence, track_id, ts, extra, incident_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                           event_id, time.time(), self.node_id, self.site_id, ev["camera_id"], ev["use_case"],
                           ev["class"], ev.get("severity", "info"), ev.get("confidence", 0.0), ev.get("track_id"),
                           ev.get("ts"), json.dumps(ev.get("extra", {})), incident_id))
        return event_id

    def set_snapshot(self, event_ids, path):
        with self._db() as db:
            db.executemany("UPDATE events SET snapshot=? WHERE event_id=?", [(path, e) for e in event_ids])

    def pending(self, limit=50, snapshot_wait_s=10.0):
        """Unsent events ready to go: snapshot attached, or none expected, or waited long enough for one."""
        rows = self._db().execute(
            f"SELECT {', '.join(FIELDS)} FROM events WHERE sent = 0 AND (snapshot IS NOT NULL "
            "OR severity NOT IN ('warning', 'critical') OR class LIKE '%_error' OR created < ?) "
            "ORDER BY CASE severity WHEN 'critical' THEN 0 WHEN 'warning' THEN 1 ELSE 2 END, created LIMIT ?", (time.time() - snapshot_wait_s, limit)).fetchall()
        return [dict(zip(FIELDS, r)) for r in rows]

    def mark_sent(self, event_ids):
        with self._db() as db:
            db.executemany("UPDATE events SET sent = 1 WHERE event_id = ?", [(e,) for e in event_ids])

    def unsent(self):
        return self._db().execute("SELECT count(*) FROM events WHERE sent = 0").fetchone()[0]
