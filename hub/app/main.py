"""api-dashboard: the hub's API.
From the nodes: POST /api/v1/events (a list; an event_id already stored is ignored), POST /api/v1/evidence/{event_id}
(a JPEG), POST /api/v1/nodes/{node_id}/health, GET /api/v1/nodes/{node_id}/config (configs/<node_id>.yaml: later
built from CampNeuron's rules). Alerts go on to Kafka in CampNeuron's format (background.py).
Dashboard API: incidents grouped by incident_id; nodes and their status. /health checks the database.
No sign-in yet (camp network only)."""
import json
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from . import background, failover

DB_URL = os.environ.get("DATABASE_URL", "postgresql://vision:change-me@localhost:5432/vision")
EVIDENCE = Path(os.environ.get("EVIDENCE_DIR", "/data/evidence"))
CONFIGS = Path(os.environ.get("CONFIG_DIR", "/data/configs"))
MAX_JPEG = 5 * 1024 * 1024
UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
NODE_ID = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
LEVELS = {0: "info", 1: "warning", 2: "critical"}
SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
  event_id uuid PRIMARY KEY, incident_id text NOT NULL, ts timestamptz NOT NULL, site_id text, node_id text,
  camera_id text NOT NULL, use_case text NOT NULL, class text NOT NULL, severity text NOT NULL,
  confidence real, track_id bigint, has_evidence boolean NOT NULL DEFAULT false, extra jsonb,
  rule text, instance text, model text, published boolean NOT NULL DEFAULT false,
  received timestamptz NOT NULL DEFAULT now());
CREATE INDEX IF NOT EXISTS events_incident ON events (incident_id);
CREATE INDEX IF NOT EXISTS events_ts ON events (ts DESC);
CREATE INDEX IF NOT EXISTS events_unpublished ON events (received) WHERE NOT published;
CREATE TABLE IF NOT EXISTS nodes (node_id text PRIMARY KEY, site_id text, last_seen timestamptz NOT NULL,
  down_since timestamptz, report jsonb);
"""
pool = ConnectionPool(DB_URL, min_size=1, max_size=8, open=False, kwargs={"row_factory": dict_row})


@asynccontextmanager
async def lifespan(app):
    pool.open(wait=True, timeout=60)
    with pool.connection() as c:
        c.execute(SCHEMA)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    background.start(pool)
    yield
    pool.close()


app = FastAPI(title="api-dashboard", version="0.2", lifespan=lifespan)


@app.get("/health")
def health():
    try:
        with pool.connection() as c:
            c.execute("SELECT 1")
    except Exception as e:
        raise HTTPException(503, f"database unavailable: {str(e)[:200]}")
    return {"status": "ok", "database": "ok"}


@app.post("/api/v1/events")
async def receive_events(request: Request):
    batch = await request.json()
    if not isinstance(batch, list):
        raise HTTPException(422, "expected a list of events")
    new, rejected = 0, []
    with pool.connection() as c:
        for e in batch:
            try:
                track = e.get("track_id")
                with c.transaction():  # one bad event is rejected on its own; the rest of the batch is kept
                    cur = c.execute(
                        "INSERT INTO events (event_id, incident_id, ts, site_id, node_id, camera_id, use_case, class, "
                        "severity, confidence, track_id, extra, rule, instance, model) "
                        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (event_id) DO NOTHING",
                        (e["event_id"], e.get("incident_id") or e["event_id"], e["timestamp"], e.get("site_id"),
                         e.get("node_id"), e["camera_id"], e["use_case"], e["class"], e.get("severity", "info"),
                         e.get("confidence"), track if isinstance(track, int) and 0 <= track < 2 ** 63 else None,
                         json.dumps(e.get("extra", {})), e.get("rule"), e.get("instance"), e.get("model")))
                    new += cur.rowcount
            except Exception as err:
                rejected.append({"event_id": e.get("event_id") if isinstance(e, dict) else None, "error": str(err)[:120]})
    return {"accepted": len(batch) - len(rejected), "new": new, "rejected": rejected}


@app.post("/api/v1/evidence/{event_id}")
async def receive_evidence(event_id: str, request: Request):
    if not UUID.match(event_id):
        raise HTTPException(422, "bad event_id")
    data = await request.body()
    if len(data) > MAX_JPEG or not data.startswith(b"\xff\xd8"):
        raise HTTPException(422, "expected a JPEG under 5 MB")
    (EVIDENCE / f"{event_id}.jpg").write_bytes(data)
    with pool.connection() as c:
        c.execute("UPDATE events SET has_evidence = true WHERE event_id = %s", (event_id,))
    return {"ok": True}


@app.get("/api/v1/evidence/{event_id}")
def get_evidence(event_id: str):
    path = EVIDENCE / f"{event_id}.jpg"
    if not UUID.match(event_id) or not path.exists():
        raise HTTPException(404, "no evidence for this event")
    return FileResponse(path, media_type="image/jpeg")


@app.post("/api/v1/nodes/{node_id}/health")
async def receive_health(node_id: str, request: Request):
    if not NODE_ID.match(node_id):
        raise HTTPException(422, "bad node_id")
    report = await request.json()
    with pool.connection() as c:
        before = c.execute("SELECT down_since FROM nodes WHERE node_id = %s", (node_id,)).fetchone()
        c.execute("INSERT INTO nodes (node_id, site_id, last_seen, report) VALUES (%s, %s, now(), %s) "
                  "ON CONFLICT (node_id) DO UPDATE SET site_id = EXCLUDED.site_id, last_seen = now(), "
                  "report = EXCLUDED.report, down_since = NULL", (node_id, report.get("site_id"), json.dumps(report)))
        if before and before["down_since"]:  # it was down: close the incident
            background.node_event(c, node_id, report.get("site_id"), "node_up", "info", before["down_since"],
                                  {"down_since": before["down_since"].isoformat()})
    return {"ok": True}


@app.get("/api/v1/nodes/{node_id}/config", response_class=PlainTextResponse)
def node_config(node_id: str):
    path = CONFIGS / f"{node_id}.yaml"
    if not NODE_ID.match(node_id) or not path.exists():
        raise HTTPException(404, "no settings for this node")
    with pool.connection() as c:  # nodes that are down right now: their spare also gets their cameras
        down = {r["node_id"] for r in c.execute("SELECT node_id FROM nodes WHERE down_since IS NOT NULL").fetchall()}
    text, cover = failover.settings_for(node_id, CONFIGS, down)
    return PlainTextResponse(text, media_type="text/yaml")


@app.get("/api/v1/nodes")
def list_nodes():
    with pool.connection() as c:
        rows = c.execute("SELECT node_id, site_id, last_seen, down_since FROM nodes ORDER BY node_id").fetchall()
    return [dict(r, status="down" if r["down_since"] else "up") for r in rows]


@app.get("/api/v1/incidents")
def incidents(status: str = "open", hours: int = 24, limit: int = 100):
    """Incidents grouped by incident_id, worst and most recent first. status: open | closed | all."""
    level = "max(CASE severity WHEN 'critical' THEN 2 WHEN 'warning' THEN 1 ELSE 0 END)"
    closed = "bool_or(class = ANY(%(ends)s))"
    having = {"open": f"NOT {closed}", "closed": closed}.get(status, "true")
    with pool.connection() as c:
        rows = c.execute(f"""
            SELECT incident_id, min(ts) AS started, max(ts) AS latest, min(camera_id) AS camera_id,
                   min(use_case) AS use_case, min(rule) AS rule, min(instance) AS instance,
                   (array_agg(class ORDER BY ts DESC))[1] AS latest_class, {level} AS level, count(*) AS events,
                   {closed} AS closed,
                   (array_agg(event_id::text ORDER BY ts DESC) FILTER (WHERE has_evidence))[1] AS evidence_event
            FROM events WHERE ts > now() - make_interval(hours => %(hours)s)
            GROUP BY incident_id HAVING {level} >= 1 AND {having}
            ORDER BY {closed}, {level} DESC, max(ts) DESC LIMIT %(limit)s""",
                         {"ends": background.ENDS, "hours": hours, "limit": limit}).fetchall()
    for r in rows:
        r["level"] = LEVELS[r["level"]]
        r["evidence_url"] = f"/api/v1/evidence/{r['evidence_event']}" if r["evidence_event"] else None
    return rows


@app.get("/api/v1/incidents/{incident_id}")
def incident(incident_id: str):
    with pool.connection() as c:
        rows = c.execute("SELECT event_id::text, ts, class, severity, confidence, track_id, has_evidence, extra "
                         "FROM events WHERE incident_id = %s ORDER BY ts", (incident_id,)).fetchall()
    if not rows:
        raise HTTPException(404, "no such incident")
    return {"incident_id": incident_id, "events": rows}


@app.get("/", response_class=HTMLResponse)
def page():
    return """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Open incidents</title><style>
:root{color-scheme:light dark;font-family:system-ui,sans-serif}body{margin:1.5rem}h1{font-size:1.4rem}
table{border-collapse:collapse;width:100%}td,th{padding:.4rem .6rem;border-bottom:1px solid #8884;text-align:left;vertical-align:top}
.critical{color:#c62828;font-weight:600}.warning{color:#b26a00;font-weight:600}img{max-width:240px;border-radius:4px}
</style></head><body><h1>Open incidents <small id="n"></small></h1>
<table><thead><tr><th>Level</th><th>Rule</th><th>Where</th><th>Camera</th><th>Latest</th><th>Since</th><th>Events</th><th>Snapshot</th></tr></thead>
<tbody id="rows"></tbody></table><script>
async function load(){const r=await fetch('/api/v1/incidents?status=open');const d=await r.json();
document.getElementById('n').textContent='('+d.length+')';
document.getElementById('rows').innerHTML=d.map(i=>`<tr><td class="${i.level}">${i.level}</td><td>${i.rule||i.use_case}</td>
<td>${i.instance||''}</td><td>${i.camera_id}</td><td>${i.latest_class}</td><td>${new Date(i.started).toLocaleTimeString()}</td>
<td>${i.events}</td><td>${i.evidence_url?`<img src="${i.evidence_url}">`:''}</td></tr>`).join('');}
load();setInterval(load,5000);</script></body></html>"""
