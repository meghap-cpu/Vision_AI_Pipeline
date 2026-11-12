"""VLM client with a backup: main server off and on again. Run: python3 -m tests.test_vlm_failover"""
import http.server, json, threading, time

import numpy as np

from vp.vlm import VLMClient


class PretendVLM:
    """Answers chat/completions with its own name, counts requests, and can stop and start on the same port."""
    def __init__(self, name, port=0):
        self.name, self.hits = name, 0
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                outer.hits += 1
                body = json.dumps({"choices": [{"message": {"content": json.dumps({"from": outer.name})}}]}).encode()
                self.send_response(200); self.end_headers(); self.wfile.write(body)

            def log_message(self, *a):
                pass
        self.httpd = http.server.HTTPServer(("127.0.0.1", port), H)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def stop(self):
        self.httpd.shutdown(); self.httpd.server_close()


def run():
    main, backup = PretendVLM("main"), PretendVLM("backup")
    c = VLMClient(f"http://127.0.0.1:{main.port}/v1", "m", timeout_s=2, backup_url=f"http://127.0.0.1:{backup.port}/v1",
                  retry_primary_s=1.0)
    img = np.zeros((8, 8, 3), np.uint8)
    ask = lambda: c.ask([img], "q").result()
    r = ask(); print("both up      ->", r["server"], r["answer"]); assert r["server"] == "primary"
    main_port = main.port; main.stop()
    r = ask(); print("main off     ->", r["server"], r["answer"]); assert r["server"] == "backup", "retried on the backup at once"
    hits = backup.hits
    r = ask(); print("still off    ->", r["server"], "(straight to the backup)"); assert r["server"] == "backup"
    assert backup.hits == hits + 1
    main = PretendVLM("main", main_port)
    r = ask(); print("main back    ->", r["server"], "(still within 1 s: stays on backup)"); assert r["server"] == "backup"
    time.sleep(1.1)
    r = ask(); print("after 1 s    ->", r["server"], r["answer"]); assert r["server"] == "primary", "tries the main one again"
    backup.stop(); main.stop()
    try:
        VLMClient(f"http://127.0.0.1:{main_port}/v1", "m", timeout_s=1, backup_url=f"http://127.0.0.1:{backup.port}/v1").ask([img], "q").result()
        raise AssertionError("both down must raise")
    except AssertionError:
        raise
    except Exception as e:
        print("both off     -> error:", type(e).__name__)
    print("PASS: backup answers when the main VLM is down, sticks to it, returns to the main one, errors only when both are down")


if __name__ == "__main__":
    run()
