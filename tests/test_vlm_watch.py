"""VLM watch with a pretend VLM server switched off and on. Run: python3 -m tests.test_vlm_watch"""
import http.server
import threading

from vp.vlm_watch import VLMWatch


class Server:
    """A tiny pretend VLM that answers GET /v1/models, and can be stopped and started again on the same port."""
    def __init__(self, port=0):
        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200); self.end_headers(); self.wfile.write(b'{"data": []}')
            def log_message(self, *a):
                pass
        self.httpd = http.server.HTTPServer(("127.0.0.1", port), H)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def stop(self):
        self.httpd.shutdown(); self.httpd.server_close()


class Outbox:
    def __init__(self):
        self.events = []

    def add(self, ev):
        self.events.append(ev)


def run():
    vlm, box = Server(), Outbox()
    w = VLMWatch(f"http://127.0.0.1:{vlm.port}/v1", box, timeout_s=1.0)
    steps = []
    steps.append(("up", w.check(now=0)))
    vlm.stop()
    steps.append(("down 1", w.check(now=30)))      # one failed check: could be a blip, no alert
    steps.append(("down 2", w.check(now=60)))      # second failed check in a row: vlm_down
    steps.append(("down 3", w.check(now=90)))      # still down: no repeat alert
    vlm = Server(vlm.port)
    steps.append(("back", w.check(now=120)))       # answers again: vlm_up closes the incident
    steps.append(("up", w.check(now=150)))
    for name, evs in steps:
        print(f"{name:7s} -> {[(e['class'], e['severity'], e['incident']) for e in evs]}")
    got = [(e["class"], e["severity"]) for e in box.events]
    assert got == [("vlm_down", "warning"), ("vlm_up", "info")], f"unexpected events: {got}"
    assert box.events[0]["incident"] == box.events[1]["incident"] == f"vlm:127.0.0.1:{vlm.port}:down:60", \
        "down and up must share one incident, named after the server"
    assert box.events[1]["extra"]["down_for_s"] == 60
    print("PASS: one vlm_down after 2 failed checks, no repeats while down, vlm_up closes the same incident")
    vlm.stop()


if __name__ == "__main__":
    run()
