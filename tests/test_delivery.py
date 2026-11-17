"""Outbox + uploader against a pretend hub that goes down and refuses a photo. Run: python3 -m tests.test_delivery"""
import http.server, json, os, tempfile, threading, time

import numpy as np

from vp.outbox import Outbox
from vp.uploader import Uploader


class Hub:
    def __init__(self):
        self.events, self.photos, self.batches, self.refuse = [], [], [], set()
        hub = self

        class H(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                if self.path == "/api/v1/events":
                    batch = json.loads(body); hub.batches.append([e["severity"] for e in batch])
                    hub.events += [e["event_id"] for e in batch]; code = 200
                else:
                    eid = self.path.rsplit("/", 1)[1]
                    code = 422 if eid in hub.refuse else 200
                    if code == 200: hub.photos.append(eid)
                self.send_response(code); self.end_headers()
            def log_message(self, *a): pass
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def stop(self):
        self.httpd.shutdown(); self.httpd.server_close()


def wait_for(cond, s=25):
    end = time.time() + s
    while time.time() < end:
        if cond(): return True
        time.sleep(0.2)
    return False


def ev(cls, sev, **x):
    return {"camera_id": "cam01", "use_case": "test", "class": cls, "severity": sev, "ts": time.time(),
            "confidence": np.float32(0.87), "track_id": np.int64(7), "extra": x}


def run():
    d = tempfile.mkdtemp(); box = Outbox(os.path.join(d, "o.db"), "node-t", "camp")
    photo = os.path.join(d, "p.jpg"); open(photo, "wb").write(b"\xff\xd8 jpeg")

    e_np = box.add(ev("no_vest", "info", score=np.float32(0.5), box=np.array([0.1, 0.2, 0.3, 0.4], np.float32)))
    print("ok  an event full of numpy values saved (it raised before)")

    hub = Hub(); up = Uploader(box, f"http://127.0.0.1:{hub.port}")
    bad, good = box.add(ev("spill_detected", "warning")), box.add(ev("man_down", "critical"))
    box.set_snapshot([bad], photo); box.set_snapshot([good], photo)
    hub.refuse.add(bad)
    assert wait_for(lambda: good in hub.photos and box.photos_waiting() == 0), "photos settled"
    assert {e_np, bad, good} <= set(hub.events), "all events delivered"
    assert bad not in hub.photos and up.photos_refused == 1
    print("ok  a photo the hub refuses is given up and counted; every event and the other photo still delivered")

    port = hub.port; hub.stop()                       # the hub goes down
    late = [box.add(ev(f"e{i}", ["info", "warning", "critical"][i % 3])) for i in range(120)]
    assert wait_for(lambda: box.oldest_unsent_s() >= 11, 20) and box.unsent() == 120  # past the 10 s photo wait
    print(f"ok  hub down: 120 alerts wait, oldest {box.oldest_unsent_s()} s, nothing sent, nothing lost")
    hub = Hub.__new__(Hub); Hub.__init__(hub)        # back up, on a new port: point the uploader at it
    up.url = f"http://127.0.0.1:{hub.port}"
    assert wait_for(lambda: box.unsent() == 0, 30), f"still unsent: {box.unsent()}"
    first = hub.batches[0]
    assert first[:40] == ["critical"] * 40 and first[40:] == ["warning"] * 10, first
    print(f"ok  hub back: all 120 delivered in {len(hub.batches)} batches, the first one 40 critical + 10 warning")

    box.set_snapshot([late[0]], photo); hub.stop()   # a late photo while the hub is down
    assert box.delivered_snapshots(float("inf")) == [], "retention must not delete a photo still waiting to upload"
    print("ok  retention keeps a photo that is still waiting to upload")
    print("PASS: numpy-safe events; refused photos never block; backlog sent critical first; waiting photos kept")


if __name__ == "__main__":
    run()
