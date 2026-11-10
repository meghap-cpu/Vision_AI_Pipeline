"""A stand-in for the hub's event intake, to test the uploader. Keeps events in hub.db (repeats of an event_id are
ignored) and evidence files in evidence/. Usage: python3 tools/standin_hub.py [--port 8100] [--dir /tmp/hub]"""
import argparse, json, os, sqlite3, sys, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vp.alert_format import to_safety_alert
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8100)
ap.add_argument("--dir", default="/tmp/hub")
ap.add_argument("--discard-evidence", action="store_true", help="accept snapshots without keeping them (long runs)")
a = ap.parse_args()
os.makedirs(os.path.join(a.dir, "evidence"), exist_ok=True)
db, lock = sqlite3.connect(os.path.join(a.dir, "hub.db"), check_same_thread=False), threading.Lock()
db.execute("CREATE TABLE IF NOT EXISTS events (event_id TEXT PRIMARY KEY, incident_id TEXT, camera_id TEXT, "
           "use_case TEXT, class TEXT, body TEXT)")


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        data = self.rfile.read(int(self.headers["Content-Length"]))
        if self.path == "/api/v1/events":
            new = 0
            with lock, db:
                for e in json.loads(data):
                    cur = db.execute("INSERT OR IGNORE INTO events VALUES (?,?,?,?,?,?)", (
                        e["event_id"], e["incident_id"], e["camera_id"], e["use_case"], e["class"], json.dumps(e)))
                    new += cur.rowcount
                    if cur.rowcount and e["severity"] in ("warning", "critical"):
                        with open(os.path.join(a.dir, "kafka_preview.jsonl"), "a") as k:
                            k.write(json.dumps(to_safety_alert(e), ensure_ascii=False) + "\n")
            print(f"events: {new} new", flush=True)
        elif self.path.startswith("/api/v1/nodes/") and self.path.endswith("/health"):
            node = self.path.split("/")[4]
            os.makedirs(os.path.join(a.dir, "health"), exist_ok=True)
            with open(os.path.join(a.dir, "health", node + ".json"), "wb") as f:
                f.write(data)
            with open(os.path.join(a.dir, "health", node + ".jsonl"), "ab") as f:
                f.write(data + b"\n")
        elif self.path.startswith("/api/v1/evidence/"):
            if not a.discard_evidence:  # long runs: accept snapshots without filling the disk
                with open(os.path.join(a.dir, "evidence", self.path.rsplit("/", 1)[-1] + ".jpg"), "wb") as f:
                    f.write(data)
        else:
            self.send_response(404); self.end_headers(); return
        out = b'{"ok": true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_GET(self):  # GET /api/v1/nodes/<node_id>/config: what the real hub will build from Neuron's rules
        parts = self.path.strip("/").split("/")
        path = os.path.join(a.dir, "configs", parts[3] + ".yaml") if len(parts) == 5 and parts[4] == "config" else None
        if not path or not os.path.exists(path):
            self.send_response(404); self.end_headers(); return
        out = open(path, "rb").read()
        self.send_response(200)
        self.send_header("Content-Type", "text/yaml")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *args):
        pass


print(f"stand-in hub on port {a.port}, storing in {a.dir}", flush=True)
ThreadingHTTPServer(("0.0.0.0", a.port), Handler).serve_forever()
