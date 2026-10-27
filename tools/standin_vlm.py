"""A stand-in for the VLM server: OpenAI-compatible /v1/chat/completions, for testing without the real model.
Answers with the contents of /tmp/standin_vlm.json (default {"fire": false, "smoke": false}).
Usage: python3 tools/standin_vlm.py [--port 8001] [--delay 0.5] [--fenced]"""
import argparse, json, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ANSWER_FILE = "/tmp/standin_vlm.json"
ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8001)
ap.add_argument("--delay", type=float, default=0.5, help="seconds to wait before answering, like a real model")
ap.add_argument("--fenced", action="store_true", help="wrap the answer in markdown code fences, like some models do")
a = ap.parse_args()
count, lock = {"requests": 0}, threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def _send(self, obj):
        out = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        images = sum(1 for m in body.get("messages", []) if isinstance(m.get("content"), list)
                     for c in m["content"] if c.get("type") == "image_url")
        time.sleep(a.delay)
        try:
            answer = open(ANSWER_FILE).read().strip()
        except FileNotFoundError:
            answer = '{"fire": false, "smoke": false}'
        if a.fenced:
            answer = f"```json\n{answer}\n```"
        with lock:
            count["requests"] += 1
            n = count["requests"]
        print(f"request {n}: {images} image(s) -> {answer!r}", flush=True)
        self._send({"model": body.get("model"), "choices": [{"message": {"role": "assistant", "content": answer}}]})

    def do_GET(self):  # /v1/models, like vLLM
        self._send({"data": [{"id": "qwen3-vl"}]})

    def log_message(self, *args):
        pass


print(f"stand-in VLM on port {a.port} (delay {a.delay}s{', fenced' if a.fenced else ''})", flush=True)
ThreadingHTTPServer(("0.0.0.0", a.port), Handler).serve_forever()
