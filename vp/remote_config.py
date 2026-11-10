"""Settings from the hub (which mirrors Camp Neuron's rule engine), with a local copy.
node.yaml is a bootstrap: node_id, config_url, and optionally full settings as a last resort.
Start: fetch config_url -> validate -> save a local copy -> use. Hub unreachable: use the local copy from last time;
none: use the bootstrap file. While running, a watcher checks the hub; valid new settings make the node exit with
RESTART_CODE so its supervisor (tools/run_node.sh, or Docker) restarts it with them. Invalid settings are ignored."""
import hashlib
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


def load_node(bootstrap_path, cache_path="data/config_cache.yaml"):
    """Returns (settings, where they came from, content hash, config_url)."""
    with open(bootstrap_path) as f:
        boot = yaml.safe_load(f)
    url = boot.get("config_url")
    if not url:
        return load(bootstrap_path), "local file", None, None
    try:
        text = _fetch(url)
        settings = from_dict(yaml.safe_load(text))  # validate before trusting or saving it
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path + ".tmp", "w") as f:
            f.write(text)
        os.replace(cache_path + ".tmp", cache_path)
        return settings, f"hub ({url})", _hash(text), url
    except Exception as e:
        why = str(e)[:80]
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                text = f.read()
            return from_dict(yaml.safe_load(text)), f"local copy ({cache_path}); hub unreachable: {why}", _hash(text), url
        if "cameras" in boot:
            return from_dict(boot), f"bootstrap file; hub unreachable: {why}", None, url
        raise RuntimeError(f"no settings: hub unreachable ({why}) and no local copy")


def watch(url, current_hash, on_change, every_s=60):
    """Checks the hub every every_s; calls on_change once when valid new settings appear."""
    def loop():
        while True:
            time.sleep(every_s)
            try:
                text = _fetch(url)
            except Exception:
                continue  # hub unreachable: keep running with what we have
            if _hash(text) == current_hash:
                continue
            try:
                from_dict(yaml.safe_load(text))
            except Exception as e:
                print(f"settings at the hub are invalid, ignoring them: {str(e)[:120]}", flush=True)
                continue
            on_change()
            return
    threading.Thread(target=loop, daemon=True).start()
