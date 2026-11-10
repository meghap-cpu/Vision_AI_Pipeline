"""Report for a soak run: health reports, node log, outbox and hub. Usage: python3 tools/soak_report.py"""
import json, os, sqlite3, statistics
from datetime import datetime

H = [json.loads(l) for l in open("/tmp/hub/health/pilot-01.jsonl") if l.strip()]
t = [datetime.fromisoformat(h["time"]).timestamp() for h in H]
gaps = [b - a for a, b in zip(t, t[1:])]
print(f"soak: {len(H)} health reports over {(t[-1] - t[0]) / 3600:.1f} h | longest gap {max(gaps, default=0):.0f} s "
      f"(over 90 s means the node was down or restarting: {sum(g > 90 for g in gaps)})")
fps = [min(h["cameras_fps"].values()) for h in H if h.get("cameras_fps")]
print(f"slowest camera fps: median {statistics.median(fps):.1f}, worst {min(fps):.1f} | dropped frames at the end: "
      f"{H[-1].get('dropped_frames')}")
mem = [h["memory_used_pct"] for h in H]
q = len(mem) // 4
print(f"memory used: start {mem[0]}%, end {mem[-1]}%, max {max(mem)}% | by quarter: "
      f"{[round(statistics.mean(mem[i * q:(i + 1) * q]), 1) for i in range(4)]}  (a steady climb would mean a leak)")
disk = [h["disk_free_gb"] for h in H]
print(f"disk free: start {disk[0]} GB, end {disk[-1]} GB, lowest {min(disk)} GB")
gpu = [h["gpu_load_pct"] for h in H if "gpu_load_pct" in h]
print(f"GPU load (momentary samples): average {statistics.mean(gpu):.0f}%, max {max(gpu):.0f}%")
temps = {}
for h in H:
    for k, v in h.get("temperatures_c", {}).items():
        temps[k] = max(temps.get(k, 0), v)
print("hottest temperatures:", {k: f"{v} C" for k, v in sorted(temps.items())})
print(f"unsent alerts: max {max(h.get('unsent_alerts', 0) for h in H)}, end {H[-1].get('unsent_alerts')} | "
      f"cameras ever offline: {sorted({c for h in H for c in h.get('cameras_offline', [])}) or 'none'} | "
      f"workers ever down: {sorted({w for h in H for w, ok in (h.get('workers_alive') or {}).items() if not ok}) or 'none'}")
log = open("/tmp/soak/node.log").read()
print(f"node log: restarts {log.count('run_node: exited')}, tracebacks {log.count('Traceback')}, "
      f"health failures {sum(1 for l in log.splitlines() if l.startswith('health: report ') and ' failed' in l)}")
snaps = sum(e.stat().st_size for e in os.scandir("data/snapshots")) / 1e9 if os.path.isdir("data/snapshots") else 0
box = sqlite3.connect("data/outbox.db")
total, unsent = box.execute("select count(*), sum(sent = 0) from events").fetchone()
print(f"node: snapshot folder {snaps:.2f} GB (cap 2 GB) | outbox {total} events, {unsent} unsent")
print("alerts by use case (warning/critical):", dict(box.execute(
    "select use_case, count(*) from events where severity in ('warning','critical') group by use_case").fetchall()))
hub = sqlite3.connect("/tmp/hub/hub.db")
print("hub received by use case:", dict(hub.execute("select use_case, count(*) from events group by use_case").fetchall()))
