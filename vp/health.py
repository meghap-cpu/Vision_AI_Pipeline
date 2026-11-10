"""Health reporter: every interval_s the node sends its health to the hub (POST /api/v1/nodes/<node_id>/health):
fps per camera, offline cameras, dropped frames, plugin workers, unsent alerts, GPU load, temperatures, memory, disk.
Best effort: an unreachable hub just misses a report, and missing reports are how the hub spots a dead node.
Values the device can't provide (e.g. Jetson sensors on a PC) are left out."""
import glob
import json
import os
import shutil
import threading
import time
import urllib.request
from datetime import datetime


class HealthReporter:
    def __init__(self, hub_url, node_id, site_id, status_fn, interval_s=30, data_dir="data"):
        self.url = hub_url.rstrip("/") + f"/api/v1/nodes/{node_id}/health"
        self.node_id, self.site_id, self.status_fn = node_id, site_id, status_fn
        self.interval, self.data_dir, self.started = interval_s, data_dir, time.time()
        # Known GPU load files (JetPack 6, then older layouts). Never search /sys recursively: it has link loops.
        self.gpu_load, self.temps = None, {}
        try:
            for pattern in ("/sys/devices/platform/bus@0/17000000.gpu/load", "/sys/devices/platform/17000000.gpu/load",
                            "/sys/devices/platform/17000000.ga10b/load", "/sys/devices/gpu.0/load"):
                if os.path.exists(pattern):
                    self.gpu_load = pattern
                    break
            for zone in glob.glob("/sys/class/thermal/thermal_zone*"):
                try:
                    self.temps[open(zone + "/type").read().strip()] = zone + "/temp"
                except OSError:
                    pass
        except Exception as e:  # health is best effort: it must never stop the pipeline
            print(f"health: sensor discovery failed ({e}); reporting without GPU/temperature", flush=True)
        self.sent = self.failed = 0
        print(f"health: reporting to the hub every {interval_s} s "
              f"(GPU load {'found' if self.gpu_load else 'not available'}, {len(self.temps)} temperature sensors)", flush=True)
        threading.Thread(target=self._loop, daemon=True).start()

    def report(self):
        h = {"node_id": self.node_id, "site_id": self.site_id,
             "time": datetime.now().astimezone().isoformat(timespec="seconds"),
             "uptime_s": int(time.time() - self.started)}
        h.update(self.status_fn())
        if self.gpu_load:
            try:
                h["gpu_load_pct"] = round(int(open(self.gpu_load).read()) / 10, 1)
            except Exception:
                pass
        temps = {}
        for name, path in self.temps.items():
            try:
                t = int(open(path).read()) / 1000
            except Exception:  # some sensor files can't be read (driver quirks): skip them
                continue
            if -40 < t < 150:
                temps[name] = round(t, 1)
        if temps:
            h["temperatures_c"] = temps
        mem = {}
        for line in open("/proc/meminfo"):
            k, v = line.split(":", 1)
            mem[k] = int(v.split()[0])
        h["memory_used_pct"] = round(100 * (1 - mem["MemAvailable"] / mem["MemTotal"]), 1)
        h["disk_free_gb"] = round(shutil.disk_usage(self.data_dir if os.path.isdir(self.data_dir) else "/").free / 1e9, 1)
        return h

    def _loop(self):
        while True:
            time.sleep(self.interval)
            try:
                req = urllib.request.Request(self.url, data=json.dumps(self.report()).encode(),
                                             headers={"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=5).read()
                self.sent += 1
            except Exception as e:  # best effort: the hub notices missing reports; say why in our own log
                self.failed += 1
                if self.failed == 1 or self.failed % 10 == 0:
                    print(f"health: report {self.failed} failed: {type(e).__name__}: {str(e)[:150]}", flush=True)
