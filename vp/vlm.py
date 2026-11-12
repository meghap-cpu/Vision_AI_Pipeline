"""Client for the VLM server (OpenAI-compatible chat completions, e.g. vLLM). Never blocks the caller:
ask() returns a Future at once; at most max_in_flight requests run together; replies are parsed robustly.
With a backup_url: a failed request is retried on the backup at once, and the client stays on the backup for
retry_primary_s before trying the primary again. Each result says which server answered."""
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
    def __init__(self, url, model, timeout_s=30.0, max_in_flight=1, backup_url=None, retry_primary_s=60.0):
        self.urls = [u.rstrip("/") + "/chat/completions" for u in (url, backup_url) if u]
        self.url = self.urls[0]
        self.retry_primary_s, self._primary_failed_at = retry_primary_s, None
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

    def _order(self):
        """Primary first, unless it failed less than retry_primary_s ago: then the backup first."""
        with self._lock:
            recent = self._primary_failed_at is not None and time.time() - self._primary_failed_at < self.retry_primary_s
        return self.urls[1:] + self.urls[:1] if recent and len(self.urls) > 1 else list(self.urls)

    def _call(self, images_rgb, prompt, max_tokens):
        content = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + to_jpeg_b64(im)}}
                   for im in images_rgb]
        content.append({"type": "text", "text": prompt})
        body = json.dumps({"model": self.model, "messages": [{"role": "user", "content": content}],
                           "max_tokens": max_tokens, "temperature": 0.1,
                           "chat_template_kwargs": {"enable_thinking": False}}).encode()
        t0, error = time.time(), None
        for url in self._order():
            try:
                req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    reply = json.loads(r.read())["choices"][0]["message"]["content"]
            except Exception as e:  # this server failed: note it, try the next one
                error = e
                if url == self.urls[0]:
                    with self._lock:
                        self._primary_failed_at = time.time()
                continue
            if url == self.urls[0]:
                with self._lock:
                    self._primary_failed_at = None
            return {"answer": parse_json(reply), "raw": reply, "seconds": round(time.time() - t0, 2),
                    "server": "primary" if url == self.urls[0] else "backup"}
        raise error
