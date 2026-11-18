"""VLM client: a stuck server, a busy client, awkward replies, one shared client per use case, and both VLM plugins
when the client is busy. Run: python3 -m tests.test_vlm_client"""
import http.server, json, threading, time

import numpy as np

from vp import vlm
from vp.plugin import Frame, Obj
from vp.plugins.fire_smoke import FireSmokePlugin
from vp.plugins.worker_at_height import WorkerAtHeightPlugin

IMG = np.zeros((8, 8, 3), np.uint8)


class SlowVLM:
    """Answers chat/completions after `delay` seconds with `reply` (text); counts requests."""
    def __init__(self, delay=0.0, reply='{"fire": false, "smoke": false}'):
        self.delay, self.reply, self.hits = delay, reply, 0
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                outer.hits += 1
                time.sleep(outer.delay)
                body = json.dumps({"choices": [{"message": {"content": outer.reply}}]}).encode()
                try:
                    self.send_response(200); self.end_headers(); self.wfile.write(body)
                except OSError:
                    pass  # the client gave up waiting

            def log_message(self, *a):
                pass
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def stop(self):
        self.httpd.shutdown(); self.httpd.server_close()


def test_awkward_replies():
    cases = {
        '{"fire": true}': {"fire": True},
        '```json\n{"fire": false, "smoke": true}\n```': {"fire": False, "smoke": True},
        'Sure! {"harness": true, "notes": "rope {left side}"} hope that helps': {"harness": True, "notes": "rope {left side}"},
        '{"helmet": true} {"helmet": false}': {"helmet": True},  # two objects: the first one
        'I see {braces} first, then {"vest": false}': {"vest": False},  # a broken one before a good one
        'no json here': None, '': None, None: None, '{"cut off": tr': None,
    }
    for text, want in cases.items():
        got = vlm.parse_json(text)
        assert got == want, f"{text!r}: got {got}, want {want}"
    print("ok  replies: fences, extra words, braces in notes, two objects, broken JSON")


def test_stuck_primary_hands_over_in_time():
    stuck, backup = SlowVLM(delay=5.0), SlowVLM()
    c = vlm.VLMClient(stuck.url, "m", timeout_s=1.0, backup_url=backup.url)
    t0 = time.time()
    r = c.ask([IMG], "q").result()
    took = time.time() - t0
    assert r["server"] == "backup" and took < 2.5, f"stuck primary: answered by {r['server']} after {took:.1f} s"
    t0 = time.time()
    r = c.ask([IMG], "q").result()
    assert r["server"] == "backup" and time.time() - t0 < 0.5, "next question goes straight to the backup"
    s = c.stats()
    assert s["asked"] == s["answered"] == s["by_backup"] == 2 and s["failed"] == 0, s
    stuck.stop(); backup.stop()
    print(f"ok  stuck primary: backup answered after {took:.1f} s (timeout 1 s), then straight to the backup; {s}")


def test_busy_client_refuses_instead_of_queueing():
    slow = SlowVLM(delay=0.5)
    c = vlm.VLMClient(slow.url, "m", timeout_s=2.0, max_in_flight=2)
    futs = [c.ask([IMG], "q") for _ in range(5)]
    assert all(f is not None for f in futs[:2]) and all(f is None for f in futs[2:]), "only max_in_flight run"
    assert c.full
    for f in futs[:2]:
        f.result()
    time.sleep(0.05)  # done callbacks
    assert not c.full and c.ask([IMG], "q") is not None, "a slot frees up once an answer arrives"
    s = c.stats()
    assert s["busy"] == 3 and s["asked"] == 3, s
    dead = vlm.VLMClient("http://127.0.0.1:9/v1", "m", timeout_s=0.5)
    try:
        dead.ask([IMG], "q").result()
    except Exception:
        pass
    time.sleep(0.05)
    assert dead.stats()["failed"] == 1 and not dead.full, "a failed question frees its slot and is counted"
    slow.stop()
    print(f"ok  busy: 2 running, 3 refused at once (nothing queued), slot freed after the answer; failures counted")


def test_shared_client_per_use_case():
    v = {"url": "http://127.0.0.1:9/v1", "timeout_s": 8}
    a, b = FireSmokePlugin("cam1", {"vlm": v}), FireSmokePlugin("cam2", {"vlm": v})
    assert a.client is b.client, "cameras of one use case share one client"
    assert a.client.max_in_flight == 2 and a.client.timeout == 8.0, "1 slot per fire camera, as before"
    assert FireSmokePlugin("cam1", {"vlm": v}).client is a.client, "a plugin that starts fresh gets the same client"
    assert a.client.max_in_flight == 2, "a fresh start does not add slots again"
    zone = [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]]
    w = [WorkerAtHeightPlugin(c, {"vlm": v, "zones": zone}) for c in ("cam3", "cam4")]
    assert w[0].client is w[1].client is not a.client, "each use case has its own"
    assert w[0].client.max_in_flight == 4, "2 slots per at-height camera: the capacity of one client per camera"
    assert vlm.shared("x", {}, "c", 1).timeout == vlm.DEFAULT_TIMEOUT_S == 8.0
    a.client.counts["asked"] += 3
    assert vlm.shared_stats()["fire_smoke"]["asked"] >= 3
    print("ok  shared: one client per use case with each camera's slots, reused after a fresh start; stats per use case")


class BusyThenFree:
    """A client stand-in: busy (None) for the first `busy` questions, then answers at once with `answer`."""
    def __init__(self, busy, answer):
        from concurrent.futures import Future
        self.Future, self.busy, self.answer, self.asked = Future, busy, answer, []

    def ask(self, images, prompt, max_tokens=60):
        self.asked.append(images)
        if self.busy > 0:
            self.busy -= 1
            return None
        f = self.Future()
        f.set_result({"answer": self.answer, "raw": "", "seconds": 0.1, "server": "primary"})
        return f


def test_plugins_ask_again_when_busy():
    fire = FireSmokePlugin("cam1", {"positive_batches": 1})
    fire.client = BusyThenFree(busy=3, answer={"fire": True, "smoke": False})
    got = []
    for n in range(8):
        img = np.full((4, 4, 3), n, np.uint8)
        got += [e["class"] for e in fire.on_frame(Frame("cam1", n, float(n), [], pixels=img))]
    assert len(fire.client.asked) == 4, f"asked on frames 3, 4, 5 (busy) and 6: {len(fire.client.asked)}"
    assert [int(im[0, 0, 0]) for im in fire.client.asked[-1]] == [3, 4, 5, 6], "the newest 4 frames, not stale ones"
    assert got == ["fire_smoke_start"], got

    params = {"zones": [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]]}
    wah = WorkerAtHeightPlugin("cam2", params)
    wah.client = BusyThenFree(busy=2, answer={"harness": True, "helmet": True, "vest": True})
    img = np.zeros((720, 1280, 3), np.uint8)
    for n in range(6):
        wah.on_frame(Frame("cam2", n, n / 5, [Obj(1, 0, 0.9, (0.4, 0.4, 0.5, 0.9))], pixels=img))
    assert len(wah.client.asked) == 3, f"asked again after each busy reply: {len(wah.client.asked)}"
    assert wah.incidents[1]["verdict"] is not None and not wah.incidents[1]["verdict"]["violation"]

    # one slot, two people waiting: the one near an edge (3 s limit) goes first although they arrived later
    wah = WorkerAtHeightPlugin("cam3", {**params, "edges": [[[0.0, 0.95], [1.0, 0.95]]]})
    wah.client = BusyThenFree(busy=0, answer={"harness": True, "helmet": True, "vest": True})
    wah.client.ask = lambda images, prompt, max_tokens=60: (wah.client.asked.append(images[0].shape), None)[1]
    wah.on_frame(Frame("cam3", 0, 0.0, [Obj(1, 0, 0.9, (0.1, 0.1, 0.2, 0.5))], pixels=img))  # far from the edge
    wah.client.asked.clear()
    wah.on_frame(Frame("cam3", 1, 1.0, [Obj(1, 0, 0.9, (0.1, 0.1, 0.2, 0.5)),
                                        Obj(2, 0, 0.9, (0.6, 0.5, 0.9, 0.95))], pixels=img))  # near the edge
    assert len(wah.client.asked) == 1, "stops asking once the client is busy"
    assert wah.client.asked[0][1] > 300, f"near-edge person (wide crop) asked first: {wah.client.asked}"
    print("ok  plugins: fire keeps its newest 4 frames and asks again; worker at height asks again with a fresh crop, "
          "most urgent person first")


def test_worker_sends_stats():
    import queue
    from vp import worker
    worker.STATS_EVERY_S, frames, events = 0.0, queue.Queue(), queue.Queue()
    frames.put(Frame("cam1", 0, 0.0, []))
    frames.put(None)  # the worker stops after one frame
    try:
        worker.run("fire_smoke", {"cam1": {"vlm": {"url": "http://127.0.0.1:9/v1"}}}, frames, events)
    finally:
        worker.STATS_EVERY_S = 60.0
    got = [events.get_nowait() for _ in range(events.qsize())]
    stats = [m for m in got if m.get("stats") == "vlm"]
    assert stats and stats[0]["use_case"] == "fire_smoke" and "asked" in stats[0]["values"], got
    assert all("class" not in m for m in stats), "a stats message is not an event"
    print(f"ok  worker: VLM stats go to the main loop as their own message: {stats[0]['values']}")


def run():
    test_awkward_replies()
    test_worker_sends_stats()
    test_stuck_primary_hands_over_in_time()
    test_busy_client_refuses_instead_of_queueing()
    test_shared_client_per_use_case()
    test_plugins_ask_again_when_busy()
    print("PASS: stuck VLM hands over in time; a busy client refuses instead of queueing; awkward replies read; "
          "one client per use case; plugins ask again when busy")


if __name__ == "__main__":
    run()
