"""Local store-and-forward queue for events. Survives restarts; the uploader sends it to the hub.
Each event's photo is tracked on its own (photo_sent: NULL none, 0 waiting, 1 uploaded, 2 refused or missing), so an
event is sent as soon as it is ready and a photo the hub refuses can never hold other alerts back."""
import json
import os
import sqlite3
import threading
import time
import uuid


def _plain(o):
    """For json.dumps: numpy numbers and arrays (and anything else odd) become plain values instead of an error."""
    for attr in ("item", "tolist"):
        if hasattr(o, attr):
            try:
                return getattr(o, attr)()
            except Exception:
                pass
    return str(o)


def _num(v, kind):
    return None if v is None else kind(v)  # numpy scalars -> plain numbers SQLite can store

COLUMNS = ("event_id TEXT PRIMARY KEY, created REAL, node_id TEXT, site_id TEXT, camera_id TEXT, use_case TEXT, "
           "class TEXT, severity TEXT, confidence REAL, track_id INTEGER, ts REAL, extra TEXT, snapshot TEXT, "
           "sent INTEGER DEFAULT 0, incident_id TEXT")
FIELDS = ("event_id", "created", "node_id", "site_id", "camera_id", "use_case", "class", "severity", "confidence",
          "track_id", "ts", "extra", "snapshot", "incident_id", "rule", "instance", "model")


class Outbox:
    def __init__(self, path, node_id, site_id):
        self.path, self.node_id, self.site_id = path, node_id, site_id
        self._local = threading.local()  # one SQLite connection per thread
        db = self._db()
        db.execute("PRAGMA journal_mode=WAL")
        with db:
            db.execute(f"CREATE TABLE IF NOT EXISTS events ({COLUMNS})")
            db.execute("CREATE INDEX IF NOT EXISTS events_snapshot ON events (snapshot)")
            db.execute("CREATE INDEX IF NOT EXISTS events_sent_created ON events (sent, created)")
            for col in ("rule", "instance", "model"):  # CampNeuron alert fields
                if col not in {r[1] for r in db.execute("PRAGMA table_info(events)")}:
                    db.execute(f"ALTER TABLE events ADD COLUMN {col} TEXT")
            if "incident_id" not in {r[1] for r in db.execute("PRAGMA table_info(events)")}:
                db.execute("ALTER TABLE events ADD COLUMN incident_id TEXT")  # outbox from an older version
            if "photo_sent" not in {r[1] for r in db.execute("PRAGMA table_info(events)")}:
                db.execute("ALTER TABLE events ADD COLUMN photo_sent INTEGER")  # older rows: NULL = handled the old way
            db.execute("CREATE INDEX IF NOT EXISTS events_sent_severity_created ON events (sent, severity, created)")
            db.execute("CREATE INDEX IF NOT EXISTS events_photo_sent ON events (photo_sent, created)")

    def _db(self):
        if not hasattr(self._local, "db"):
            self._local.db = sqlite3.connect(self.path, timeout=30)
        return self._local.db

    def add(self, ev):
        event_id = str(uuid.uuid4())
        key = ev.get("incident")
        incident_id = f"{self.node_id}:{ev['camera_id']}:{ev['use_case']}:{key}" if key else event_id
        with self._db() as db:
            db.execute("INSERT INTO events (event_id, created, node_id, site_id, camera_id, use_case, class, severity, "
                       "confidence, track_id, ts, extra, incident_id, rule, instance, model) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                           event_id, time.time(), self.node_id, self.site_id, ev["camera_id"], ev["use_case"],
                           ev["class"], ev.get("severity", "info"), _num(ev.get("confidence", 0.0), float),
                           _num(ev.get("track_id"), int), _num(ev.get("ts"), float),
                           json.dumps(ev.get("extra", {}), default=_plain), incident_id, ev.get("rule"), ev.get("instance"),
                           ev.get("model")))
        return event_id

    def set_snapshot(self, event_ids, path):
        with self._db() as db:
            db.executemany("UPDATE events SET snapshot=?, photo_sent=0 WHERE event_id=?", [(path, e) for e in event_ids])

    def pending(self, limit=50, snapshot_wait_s=10.0):
        """Unsent events ready to go (snapshot attached, none expected, or waited long enough for one): critical first,
        then warning, then the rest, oldest first. Three index reads, so the cost stays flat however many wait."""
        ready = ("(snapshot IS NOT NULL OR severity NOT IN ('warning', 'critical') OR class LIKE '%_error' "
                 "OR created < ?)")
        cutoff, out = time.time() - snapshot_wait_s, []
        for where, args in (("severity = 'critical'", ()), ("severity = 'warning'", ()),
                            ("severity NOT IN ('critical', 'warning')", ())):
            if len(out) >= limit:
                break
            rows = self._db().execute(
                f"SELECT {', '.join(FIELDS)} FROM events WHERE sent = 0 AND {where} AND {ready} "
                "ORDER BY created LIMIT ?", (*args, cutoff, limit - len(out))).fetchall()
            out += [dict(zip(FIELDS, r)) for r in rows]
        return out

    def pending_photos(self, limit=20):
        """[(event_id, path)] of photos waiting to upload, for events the hub already has; oldest first."""
        return self._db().execute("SELECT event_id, snapshot FROM events WHERE photo_sent = 0 AND sent = 1 "
                                  "ORDER BY created LIMIT ?", (limit,)).fetchall()

    def mark_photos(self, event_ids, state):
        """state 1 = uploaded, 2 = refused by the hub or file missing (given up)."""
        with self._db() as db:
            db.executemany("UPDATE events SET photo_sent = ? WHERE event_id = ?", [(state, e) for e in event_ids])

    def oldest_unsent_s(self):
        row = self._db().execute("SELECT min(created) FROM events WHERE sent = 0").fetchone()
        return round(time.time() - row[0]) if row and row[0] else 0

    def size_mb(self):
        """The outbox's files on disk (SQLite never shrinks them by itself)."""
        return round(sum(os.path.getsize(p) for p in (self.path, self.path + "-wal") if os.path.exists(p)) / 1e6, 1)

    def photos_waiting(self):
        return self._db().execute("SELECT count(*) FROM events WHERE photo_sent = 0").fetchone()[0]

    def mark_sent(self, event_ids):
        with self._db() as db:
            db.executemany("UPDATE events SET sent = 1 WHERE event_id = ?", [(e,) for e in event_ids])

    def unsent(self):
        return self._db().execute("SELECT count(*) FROM events WHERE sent = 0").fetchone()[0]

    def delivered_snapshots(self, older_than):
        """Snapshot files whose events have ALL reached the hub, newest event older than older_than; oldest first."""
        rows = self._db().execute(
            "SELECT snapshot FROM events WHERE snapshot IS NOT NULL GROUP BY snapshot "
            "HAVING min(sent) = 1 AND min(COALESCE(photo_sent, 1)) >= 1 AND max(created) < ? "  # photo done too
            "ORDER BY max(created)", (older_than,)).fetchall()
        return [r[0] for r in rows]

    def forget_snapshots(self, paths):
        with self._db() as db:
            db.executemany("UPDATE events SET snapshot = NULL WHERE snapshot = ?", [(p,) for p in paths])

    def prune_sent(self, older_than):
        with self._db() as db:
            return db.execute("DELETE FROM events WHERE sent = 1 AND COALESCE(photo_sent, 1) <> 0 AND created < ?",
                              (older_than,)).rowcount
