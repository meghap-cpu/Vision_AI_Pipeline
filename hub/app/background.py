"""Hub background jobs.
1. Kafka publisher: new alerts go to KAFKA_TOPIC (default safety.alerts) in CampNeuron's format, each marked
   published only after Kafka confirms it (acks=all). Kafka down: they wait in Postgres and are retried.
   Each message carries its event_id, so a rare re-send can be recognised as a duplicate.
2. Node monitor: a node whose health reports stop for NODE_DOWN_AFTER_S raises node_down (critical);
   its next report raises node_up, which closes the incident."""
import json
import os
import shutil
from pathlib import Path
import threading
import time
import uuid
from zoneinfo import ZoneInfo

from .alert_format import to_safety_alert

BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "")
TOPIC = os.environ.get("KAFKA_TOPIC", "safety.alerts")
PUBLIC_URL = os.environ.get("HUB_PUBLIC_URL", "http://localhost:8100").rstrip("/")
SITE_TZ = ZoneInfo(os.environ.get("SITE_TZ", "UTC"))
DOWN_AFTER_S = int(os.environ.get("NODE_DOWN_AFTER_S", "120"))
# Clean-up (3rd background job): evidence photos and old events, like the nodes' own retention
EVIDENCE = Path(os.environ.get("EVIDENCE_DIR", "/data/evidence"))
EVIDENCE_KEEP_DAYS = float(os.environ.get("EVIDENCE_KEEP_DAYS", "30"))
EVIDENCE_MAX_GB = float(os.environ.get("EVIDENCE_MAX_GB", "20"))
EVENTS_KEEP_DAYS = float(os.environ.get("EVENTS_KEEP_DAYS", "90"))
RECENT_S = 3600  # photos from the last hour are never deleted, even over the cap
HUB_ID = os.environ.get("HUB_ID", "hub")
DISK_WARN_GB = float(os.environ.get("HUB_DISK_WARN_GB", "2"))  # hub's own disk alert; 0 = off
ENDS = ["motion_end", "crowd_end", "object_removed", "man_down_end", "fire_smoke_end", "at_height_end", "node_up", "camera_online", "vlm_up", "disk_ok", "storage_ok"]


def node_event(c, node_id, site_id, cls, severity, since, extra):
    c.execute("INSERT INTO events (event_id, incident_id, ts, site_id, node_id, camera_id, use_case, class, severity, "
              "confidence, extra) VALUES (%s, %s, now(), %s, %s, 'node', 'health', %s, %s, 1.0, %s)",
              (uuid.uuid4(), f"{node_id}:health:node_down:{int(since.timestamp())}", site_id, node_id, cls, severity,
               json.dumps(extra)))


def _message(r):
    e = dict(r, event_id=str(r["event_id"]), timestamp=r["ts"].astimezone(SITE_TZ).isoformat(timespec="seconds"))
    return to_safety_alert(e, evidence_base=f"{PUBLIC_URL}/api/v1/evidence/")


def _publisher(pool):
    from confluent_kafka import Producer
    producer = Producer({"bootstrap.servers": BOOTSTRAP, "acks": "all", "enable.idempotence": True,
                         "message.timeout.ms": 20000, "linger.ms": 50})
    reachable = None
    while True:
        try:
            with pool.connection() as c:
                c.execute("UPDATE events SET published = true WHERE NOT published "   # info noise: never published
                          "AND severity NOT IN ('warning', 'critical') AND NOT (class = ANY(%s))", (ENDS,))
                rows = c.execute("SELECT * FROM events WHERE NOT published ORDER BY received LIMIT 200").fetchall()
            if not rows:
                time.sleep(1)
                continue
            done = []
            for r in rows:
                producer.produce(TOPIC, key=r["incident_id"], value=json.dumps(_message(r), ensure_ascii=False, default=str),
                                 on_delivery=lambda err, msg, eid=r["event_id"]: err is None and done.append(eid))
            producer.flush(25)
            if done:
                with pool.connection() as c:
                    c.execute("UPDATE events SET published = true WHERE event_id = ANY(%s)", (done,))
            ok = len(done) == len(rows)
            if ok != reachable:
                print(f"kafka: publishing to {TOPIC} at {BOOTSTRAP}" if ok else
                      f"kafka: cannot reach {BOOTSTRAP}; {len(rows) - len(done)} alerts waiting", flush=True)
                reachable = ok
            if not ok:
                time.sleep(5)
        except Exception as e:
            print(f"kafka publisher: {type(e).__name__}: {str(e)[:150]}", flush=True)
            time.sleep(5)


def _monitor(pool):
    while True:
        try:
            with pool.connection() as c:
                silent = c.execute("SELECT node_id, site_id, last_seen FROM nodes WHERE down_since IS NULL "
                                   "AND last_seen < now() - make_interval(secs => %s)", (DOWN_AFTER_S,)).fetchall()
                for n in silent:
                    c.execute("UPDATE nodes SET down_since = last_seen WHERE node_id = %s", (n["node_id"],))
                    node_event(c, n["node_id"], n["site_id"], "node_down", "critical", n["last_seen"],
                               {"last_seen": n["last_seen"].isoformat(), "down_after_s": DOWN_AFTER_S})
                    print(f"node {n['node_id']}: DOWN (no health report since {n['last_seen']:%H:%M:%S})", flush=True)
        except Exception as e:
            print(f"node monitor: {type(e).__name__}: {str(e)[:150]}", flush=True)
        time.sleep(15)


def _drop_evidence(c, ids):
    """Deletes these events' photos and marks them as having none. Returns bytes freed."""
    freed = 0
    for i in ids:
        f = EVIDENCE / f"{i}.jpg"
        try:
            freed += f.stat().st_size
            f.unlink()
        except FileNotFoundError:
            pass
    c.execute("UPDATE events SET has_evidence = false WHERE event_id = ANY(%s::uuid[])", (ids,))
    return freed


def evidence_bytes():
    with os.scandir(EVIDENCE) as it:
        return sum(e.stat().st_size for e in it if e.is_file())


def retention_once(pool):
    """1. photos of sent alerts older than EVIDENCE_KEEP_DAYS; 2. while photos exceed EVIDENCE_MAX_GB, the oldest
    photos of sent alerts (never from the last hour); 3. sent events older than EVENTS_KEEP_DAYS.
    A photo whose alert is not yet in Kafka is never deleted: CampNeuron will still need it."""
    stats = {"photos_deleted": 0, "gb_freed": 0.0, "events_deleted": 0}
    with pool.connection() as c:
        old = [r["event_id"] for r in c.execute(
            "SELECT event_id::text AS event_id FROM events WHERE has_evidence AND published "
            "AND ts < now() - make_interval(secs => %s)", (EVIDENCE_KEEP_DAYS * 86400,)).fetchall()]
        if old:
            stats["gb_freed"] += _drop_evidence(c, old) / 1e9
            stats["photos_deleted"] += len(old)
        size, cap = evidence_bytes(), EVIDENCE_MAX_GB * 1e9
        while size > cap:
            batch = [r["event_id"] for r in c.execute(
                "SELECT event_id::text AS event_id FROM events WHERE has_evidence AND published "
                "AND ts < now() - make_interval(secs => %s) ORDER BY ts LIMIT 500", (RECENT_S,)).fetchall()]
            if not batch:
                break  # what is left is unsent or from the last hour: kept
            take = []
            for i in batch:  # oldest first, and stop as soon as we are under the cap
                if size <= cap:
                    break
                try:
                    size -= (EVIDENCE / f"{i}.jpg").stat().st_size
                except FileNotFoundError:
                    pass
                take.append(i)
            stats["gb_freed"] += _drop_evidence(c, take) / 1e9
            stats["photos_deleted"] += len(take)
        stats["events_deleted"] = c.execute(
            "DELETE FROM events WHERE published AND NOT has_evidence AND ts < now() - make_interval(secs => %s)",
            (EVENTS_KEEP_DAYS * 86400,)).rowcount
    stats["gb_freed"] = round(stats["gb_freed"], 2)
    stats["photos_gb_now"] = round(evidence_bytes() / 1e9, 2)
    return stats


def hub_event(c, cls, severity, incident, extra):
    c.execute("INSERT INTO events (event_id, incident_id, ts, node_id, camera_id, use_case, class, severity, "
              "confidence, extra) VALUES (%s, %s, now(), %s, 'hub', 'health', %s, %s, 1.0, %s)",
              (uuid.uuid4(), incident, HUB_ID, cls, severity, json.dumps(extra)))


def disk_check(pool, free_bytes=None):
    """The hub's own disk: disk_low (warning) under DISK_WARN_GB free, disk_ok (same incident) once 0.5 GB over.
    The open/closed state is read from the events table, so a hub restart never repeats the alert."""
    if DISK_WARN_GB <= 0:
        return None
    free = shutil.disk_usage(EVIDENCE).free if free_bytes is None else free_bytes
    with pool.connection() as c:
        last = c.execute("SELECT class, incident_id FROM events WHERE camera_id = 'hub' AND class IN ('disk_low', 'disk_ok') "
                         "ORDER BY ts DESC, received DESC LIMIT 1").fetchone()
        low_open, gb = last is not None and last["class"] == "disk_low", round(free / 1e9, 2)
        if not low_open and free < DISK_WARN_GB * 1e9:
            hub_event(c, "disk_low", "warning", f"{HUB_ID}:health:disk_low:{uuid.uuid4().hex[:12]}", {"free_gb": gb})
            return "disk_low"
        if low_open and free >= (DISK_WARN_GB + 0.5) * 1e9:
            hub_event(c, "disk_ok", "info", last["incident_id"], {"free_gb": gb})
            return "disk_ok"
    return None


def _retention(pool):
    print(f"retention: photos {EVIDENCE_KEEP_DAYS:g} days / {EVIDENCE_MAX_GB:g} GB, events {EVENTS_KEEP_DAYS:g} days; "
          "photos of unsent alerts are never deleted", flush=True)
    while True:
        try:
            d = disk_check(pool)
            if d:
                print(f"hub disk: {d}", flush=True)
        except Exception as e:
            print(f"hub disk: {type(e).__name__}: {str(e)[:150]}", flush=True)
        try:
            s = retention_once(pool)
            if s["photos_deleted"] or s["events_deleted"]:
                print(f"retention: {s}", flush=True)
        except Exception as e:
            print(f"retention: {type(e).__name__}: {str(e)[:150]}", flush=True)
        time.sleep(600)


def start(pool):
    threading.Thread(target=_monitor, args=(pool,), daemon=True).start()
    threading.Thread(target=_retention, args=(pool,), daemon=True).start()
    if BOOTSTRAP:
        threading.Thread(target=_publisher, args=(pool,), daemon=True).start()
    else:
        print("kafka: KAFKA_BOOTSTRAP not set, alerts are not published", flush=True)
