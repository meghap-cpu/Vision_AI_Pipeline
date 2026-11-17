"""Client for the VLM server (OpenAI-compatible chat completions, e.g. vLLM). Never blocks the caller:
ask() returns a Future at once, or None when max_in_flight questions are already running (the caller asks again
later, with a fresh image, instead of queueing a stale one). Replies are parsed robustly.
With a backup_url: a failed or timed-out request is retried on the backup at once, and the client stays on the
backup for retry_primary_s before trying the primary again. Each result says which server answered.
shared(): one client per server and use case in a worker process, so every camera shares its slots and its failover
memory (a stuck primary costs one timeout per use case, not one per camera). stats(): counts for the health report.
Note: timeout_s applies to each network read, not the whole reply; a server trickling bytes could take longer."""
import base64, json, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

import cv2

DEFAULT_TIMEOUT_S = 8.0  # a stuck VLM hands over to the backup well within worker at height's 10 s


def to_jpeg_b64(rgb, quality=80):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode()


def parse_json(text):
    """The first complete {...} object in a reply, tolerating markdown fences, extra words, braces inside notes and
    a second object after it. None if there is none."""
    decoder, text, i = json.JSONDecoder(), text or "", 0
    while (i := text.find("{", i)) != -1:
        try:
            obj, _ = decoder.raw_decode(text, i)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        i += 1
    return None


class VLMClient:
    def __init__(self, url, model, timeout_s=DEFAULT_TIMEOUT_S, max_in_flight=1, backup_url=None, retry_primary_s=60.0):
        self.urls = [u.rstrip("/") + "/chat/completions" for u in (url, backup_url) if u]
        self.url = self.urls[0]
        self.retry_primary_s, self._primary_failed_at = retry_primary_s, None
        self.model, self.timeout = model, timeout_s
        self.pool = ThreadPoolExecutor(max_workers=max_in_flight)
        self.max_in_flight, self._busy, self._lock = max_in_flight, 0, threading.Lock()
        # running totals: questions asked, answered, by the backup, failed, refused because full; answer seconds
        self.counts = {"asked": 0, "answered": 0, "by_backup": 0, "failed": 0, "busy": 0}
        self._seconds = 0.0

    @property
    def full(self):
        """True when max_in_flight requests are already running."""
        with self._lock:
            return self._busy >= self.max_in_flight

    def ask(self, images_rgb, prompt, max_tokens=60):
        """A Future for the answer, or None when all slots are busy (nothing is queued)."""
        with self._lock:
            if self._busy >= self.max_in_flight:
                self.counts["busy"] += 1
                return None
            self._busy += 1
            self.counts["asked"] += 1
        fut = self.pool.submit(self._call, images_rgb, prompt, max_tokens)
        fut.add_done_callback(self._done)
        return fut

    def _done(self, fut):
        with self._lock:
            self._busy -= 1
            if fut.exception() is not None:
                self.counts["failed"] += 1
            else:
                r = fut.result()
                self.counts["answered"] += 1
                self.counts["by_backup"] += r["server"] == "backup"
                self._seconds += r["seconds"]

    def stats(self):
        """Running totals since start, plus the average answer time and the questions running now."""
        with self._lock:
            n = self.counts["answered"]
            return {**self.counts, "avg_s": round(self._seconds / n, 2) if n else None, "in_flight": self._busy,
                    "max_in_flight": self.max_in_flight}

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
            except Exception as e:  # this server failed or timed out: note it, try the next one
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


_shared = {}  # (use case, url, backup, model, timeout, max_in_flight) -> client, for this process
_shared_lock = threading.Lock()


def shared(use_case, vlm, max_in_flight):
    """The one client this process uses for use_case with these VLM settings (vlm: url, backup_url, model, timeout_s).
    Cameras of one use case share it; a plugin that starts fresh gets the same client back, not a new thread pool."""
    key = (use_case, vlm.get("url", "http://127.0.0.1:8000/v1"), vlm.get("backup_url"), vlm.get("model", "qwen3-vl"),
           float(vlm.get("timeout_s", DEFAULT_TIMEOUT_S)), int(max_in_flight))
    with _shared_lock:
        if key not in _shared:
            _shared[key] = VLMClient(key[1], key[3], key[4], max_in_flight=key[5], backup_url=key[2])
        return _shared[key]


def shared_stats():
    """{use case: stats} for this process's shared clients (summed if a use case has several, e.g. two settings)."""
    out = {}
    with _shared_lock:
        clients = list(_shared.items())
    for key, c in clients:
        s, tot = c.stats(), out.setdefault(key[0], {})
        for k, v in s.items():
            if k == "avg_s":
                continue
            tot[k] = tot.get(k, 0) + v
        tot["_seconds"] = tot.get("_seconds", 0.0) + (s["avg_s"] or 0) * s["answered"]
    for tot in out.values():
        secs = tot.pop("_seconds")
        tot["avg_s"] = round(secs / tot["answered"], 2) if tot["answered"] else None
    return out
