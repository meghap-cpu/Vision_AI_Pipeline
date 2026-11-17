"""Settings from the hub (which mirrors Camp Neuron's rule engine), with a local copy and a last known good copy.
node.yaml is a bootstrap: node_id, config_url, and optionally full settings as a last resort.
Start: fetch config_url -> validate -> save a local copy -> use. Hub unreachable: use the local copy from last time;
none: use the bootstrap file. While running, a watcher checks the hub; valid new settings make the node exit with
RESTART_CODE so its supervisor (tools/run_node.sh, or Docker) restarts it with them.
Safety: settings the thorough check refuses are never used, and settings_rejected says why (once per version).
Settings that ran GOOD_AFTER_S become "last known good"; new settings that fail to start MAX_STARTS times are
replaced by the good ones (settings_rolled_back) and ignored at the hub until they change."""
import hashlib
import json
import os
import threading
import time
import urllib.request

import yaml

from vp.settings import from_dict, load

RESTART_CODE = 3


def _fetch(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode()


def _hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


GOOD_AFTER_S = 300  # settings that ran this long become "last known good"
MAX_STARTS = 3  # new settings that fail to reach GOOD_AFTER_S this many times are rolled back


def settings_event(cls, version, **extra):
    """A warning for the outbox about settings (reaches the hub and Kafka as "Edge node health")."""
    return {"camera_id": "node", "use_case": "settings", "class": cls, "severity": "warning", "ts": time.time(),
            "incident": f"settings:{version[:12]}", "extra": {"version": version[:12], **extra}}


def _write(path, text):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + ".tmp", "w") as f:
        f.write(text)
    os.replace(path + ".tmp", path)


def _trial(data_dir):
    try:
        with open(os.path.join(data_dir, "config_trial.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def load_node(bootstrap_path, cache_path="data/config_cache.yaml"):
    """Returns (settings, where they came from, content hash, config_url, info);
    info = {"events": warnings for the outbox, "ignore": hub versions the watcher must not restart for}."""
    with open(bootstrap_path) as f:
        boot = yaml.safe_load(f)
    url = boot.get("config_url")
    info = {"events": [], "ignore": set()}
    if not url:
        return load(bootstrap_path), "local file", None, None, info
    data_dir = os.path.dirname(cache_path) or "."
    try:
        text = _fetch(url)
        try:
            settings = from_dict(yaml.safe_load(text), check_files=True)  # the thorough check, before trusting it
        except Exception as e:
            info["events"].append(settings_event("settings_rejected", _hash(text), reasons=str(e)[:500]))
            raise RuntimeError(f"settings at the hub refused: {str(e)[:80]}")
        _write(cache_path, text)
        source = f"hub ({url})"
    except Exception as e:
        why = str(e)[:80]
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                text = f.read()
            settings, source = from_dict(yaml.safe_load(text)), f"local copy ({cache_path}); {why}"
        elif "cameras" in boot:
            return from_dict(boot), f"bootstrap file; {why}", None, url, info
        else:
            raise RuntimeError(f"no settings: {why}, and no local copy")
    version = _hash(text)
    t = _trial(data_dir)
    starts = t.get("starts", 0) + 1 if t.get("hash") == version else 1
    _write(os.path.join(data_dir, "config_trial.json"), json.dumps({"hash": version, "starts": starts}))
    good_path = os.path.join(data_dir, "config_good.yaml")
    if starts > MAX_STARTS and os.path.exists(good_path):
        with open(good_path) as f:
            good = f.read()
        if _hash(good) != version:  # these settings never ran GOOD_AFTER_S: go back to the last good ones
            info["events"].append(settings_event("settings_rolled_back", version, failed_starts=starts - 1,
                                                 now_running=_hash(good)[:12]))
            info["ignore"].add(version)
            return (from_dict(yaml.safe_load(good)), f"last known good ({good_path}); settings {version[:12]} "
                    f"failed to start {starts - 1} times", _hash(good), url, info)
    return settings, source, version, url, info


def mark_good(running_version, data_dir="data"):
    """Called once the node has run GOOD_AFTER_S: the running settings become last known good."""
    cache = os.path.join(data_dir, "config_cache.yaml")
    if running_version and os.path.exists(cache):
        with open(cache) as f:
            text = f.read()
        if _hash(text) == running_version:
            _write(os.path.join(data_dir, "config_good.yaml"), text)
    if _trial(data_dir).get("hash") == running_version:
        _write(os.path.join(data_dir, "config_trial.json"), json.dumps({"hash": running_version, "starts": 0}))


def watch(url, current_hash, on_change, every_s=60, ignore=(), report=None):
    """Checks the hub every every_s; calls on_change once when valid new settings appear. Settings in ignore
    (rolled back) never trigger a restart; refused settings are reported once per version."""
    reported = set()

    def loop():
        while True:
            time.sleep(every_s)
            try:
                text = _fetch(url)
            except Exception:
                continue  # hub unreachable: keep running with what we have
            version = _hash(text)
            if version == current_hash or version in ignore:
                continue
            try:
                from_dict(yaml.safe_load(text), check_files=True)
            except Exception as e:
                if version not in reported:
                    reported.add(version)
                    print(f"settings at the hub refused, keeping the current ones: {str(e)[:200]}", flush=True)
                    if report:
                        report(settings_event("settings_rejected", version, reasons=str(e)[:500]))
                continue
            on_change()
            return
    threading.Thread(target=loop, daemon=True).start()
