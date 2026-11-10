"""Hub background jobs.
1. Kafka publisher: new alerts go to KAFKA_TOPIC (default safety.alerts) in CampNeuron's format, each marked
   published only after Kafka confirms it (acks=all). Kafka down: they wait in Postgres and are retried.
   Each message carries its event_id, so a rare re-send can be recognised as a duplicate.
2. Node monitor: a node whose health reports stop for NODE_DOWN_AFTER_S raises node_down (critical);
   its next report raises node_up, which closes the incident."""
import json
import os
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
ENDS = ["motion_end", "crowd_end", "object_removed", "man_down_end", "fire_smoke_end", "at_height_end", "node_up", "camera_online"]


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


def start(pool):
    threading.Thread(target=_monitor, args=(pool,), daemon=True).start()
    if BOOTSTRAP:
        threading.Thread(target=_publisher, args=(pool,), daemon=True).start()
    else:
        print("kafka: KAFKA_BOOTSTRAP not set, alerts are not published", flush=True)
