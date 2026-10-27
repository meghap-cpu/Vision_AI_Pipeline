"""Client for the VLM server (OpenAI-compatible chat completions, e.g. vLLM). Never blocks the caller:
ask() returns a Future at once; at most max_in_flight requests run together; replies are parsed robustly."""
import base64, json, re, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

import cv2


def to_jpeg_b64(rgb, quality=80):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode()


def parse_json(text):
    """The first {...} in a reply, tolerating markdown fences and extra words. None if there is none."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


class VLMClient:
    def __init__(self, url, model, timeout_s=30.0, max_in_flight=1):
        self.url = url.rstrip("/") + "/chat/completions"
        self.model, self.timeout = model, timeout_s
        self.pool = ThreadPoolExecutor(max_workers=max_in_flight)
        self.max_in_flight, self._busy, self._lock = max_in_flight, 0, threading.Lock()

    @property
    def full(self):
        """True when max_in_flight requests are already running: callers should wait, not queue more."""
        with self._lock:
            return self._busy >= self.max_in_flight

    def ask(self, images_rgb, prompt, max_tokens=60):
        with self._lock:
            self._busy += 1
        fut = self.pool.submit(self._call, images_rgb, prompt, max_tokens)
        fut.add_done_callback(self._done)
        return fut

    def _done(self, _):
        with self._lock:
            self._busy -= 1

    def _call(self, images_rgb, prompt, max_tokens):
        content = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + to_jpeg_b64(im)}}
                   for im in images_rgb]
        content.append({"type": "text", "text": prompt})
        body = json.dumps({"model": self.model, "messages": [{"role": "user", "content": content}],
                           "max_tokens": max_tokens, "temperature": 0.1,
                           "chat_template_kwargs": {"enable_thinking": False}}).encode()
        req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            reply = json.loads(r.read())["choices"][0]["message"]["content"]
        return {"answer": parse_json(reply), "raw": reply, "seconds": round(time.time() - t0, 2)}
