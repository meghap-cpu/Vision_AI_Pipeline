"""Settings safety: the thorough check, rollback to last known good, and the watcher. Run: python3 -m tests.test_settings"""
import copy, http.server, os, tempfile, threading, time

import yaml

from vp import remote_config as rc
from vp.settings import from_dict

BASE = {"node_id": "n1", "site_id": "camp", "engine": {"max_batch": 4},
        "cameras": [{"id": "cam00", "uri": "rtsp://cam/0", "use_cases": {"motion": {}}}]}


def variant(**changes):
    d = copy.deepcopy(BASE)
    for path, value in changes.items():
        cam, key = path.split("__")
        d["cameras"][int(cam)]["use_cases"][key] = value
    return d


def test_thorough_check():
    bad = {
        "unknown use case": variant(**{"0__flying_pigs": {}}),
        "polygon with 2 points": variant(**{"0__motion": {"zones": [[[0.1, 0.1], [0.5, 0.5]]]}}),
        "not inside the frame": variant(**{"0__crowd": {"zone": [0.5, 0.5, 1.4, 0.9]}}),
        "schedule or timezone": variant(**{"0__motion": {"schedule": [{"from": "08:00", "to": "17:00"}],
                                                         "timezone": "Mars/Olympus_Mons"}}),
        "needs a VLM": variant(**{"0__fire_smoke": {}}),
        "not on this node": dict(copy.deepcopy(BASE), models=[{"name": "ppe", "gie_id": 3, "cameras": ["cam07"]}]),
        "gie_id must be unique": dict(copy.deepcopy(BASE), models=[{"name": "a", "gie_id": 3}, {"name": "b", "gie_id": 3}]),
    }
    for expect, d in bad.items():
        try:
            from_dict(d); raise AssertionError(f"accepted: {expect}")
        except ValueError as e:
            assert expect in str(e), (expect, str(e))
    from_dict(variant(**{"0__crowd": {"zone": [0, 0, 1, 1]}}))
    print(f"ok  thorough check refuses all {len(bad)} kinds of bad settings, with the reason; good settings pass")


class Hub:
    def __init__(self):
        self.text = ""
        hub = self
        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200); self.end_headers(); self.wfile.write(hub.text.encode())
            def log_message(self, *a): pass
        self.httpd = http.server.HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/config"


def test_rollback_and_watch():
    d = tempfile.mkdtemp(); hub = Hub()
    boot = os.path.join(d, "bootstrap.yaml"); open(boot, "w").write(f"node_id: n1\nconfig_url: {hub.url}\n")
    cache = os.path.join(d, "config_cache.yaml")
    A = yaml.safe_dump(BASE)
    B = yaml.safe_dump(dict(copy.deepcopy(BASE), cameras=BASE["cameras"] + [{"id": "cam01", "uri": "rtsp://cam/1"}]))
    hub.text = A
    s, src, vA, _, info = rc.load_node(boot, cache)
    rc.mark_good(vA, d)                                   # A ran 5 min: last known good
    assert os.path.exists(os.path.join(d, "config_good.yaml")) and info["events"] == []
    hub.text = B                                          # B is valid but (say) crashes the pipeline every start
    for n in range(1, 4):
        s, src, v, _, info = rc.load_node(boot, cache)
        assert len(s.cameras) == 2 and info["events"] == [], f"start {n} tries B"
    s, src, v, _, info = rc.load_node(boot, cache)      # 4th start
    assert len(s.cameras) == 1 and v == vA and info["events"][0]["class"] == "settings_rolled_back"
    assert info["events"][0]["extra"]["failed_starts"] == 3 and info["ignore"] == {rc._hash(B)}
    print(f"ok  new settings that failed to start 3 times are rolled back: '{src[:60]}...'")

    changed = threading.Event(); reports = []
    rc.watch(hub.url, vA, changed.set, every_s=0.2, ignore=info["ignore"], report=reports.append)
    time.sleep(1.0)
    assert not changed.is_set(), "the rolled-back settings must not trigger a restart loop"
    hub.text = "node_id: n1\nsite_id: camp\nengine: {max_batch: 4}\ncameras: [{id: cam00, uri: x, use_cases: {flying_pigs: {}}}]\n"
    time.sleep(1.0)
    assert not changed.is_set() and len(reports) == 1 and reports[0]["class"] == "settings_rejected"
    assert "unknown use case" in reports[0]["extra"]["reasons"]
    hub.text = yaml.safe_dump(dict(copy.deepcopy(BASE), site_id="camp-2"))  # genuinely new, valid settings
    assert changed.wait(2), "new valid settings must still restart the node"
    print("ok  watcher: ignores rolled-back settings, reports refused ones once (not every check), restarts for new ones")


def run():
    test_thorough_check(); test_rollback_and_watch()
    print("PASS: bad settings refused with reasons; crash-looping settings rolled back; no restart loops")


if __name__ == "__main__":
    run()
