"""Replays a recording from tools/record_poses.py through one plugin with chosen settings; prints its alerts.
Usage: python3 tools/replay.py REC.jsonl PLUGIN ['{"setting": value}']"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vp.plugin import Frame, Obj
from vp.plugins import REGISTRY

rec, name = sys.argv[1], sys.argv[2]
params = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
plugin = REGISTRY[name]("replay", params)
counts, frames, prev_ts, per_track = {}, 0, None, {}
for line in open(rec):
    f = json.loads(line)
    frames += 1
    objs = [Obj(o["track_id"], 0, o["conf"], tuple(o["box"]), source=1,
                keypoints=tuple(tuple(k) for k in o["keypoints"]) if o["keypoints"] else None) for o in f["objects"]]
    events = plugin.on_frame(Frame("replay", f["frame"], f["ts"], objs))
    dt = 0.0 if prev_ts is None else f["ts"] - prev_ts
    prev_ts = f["ts"]
    for tid, st in getattr(plugin, "people", {}).items():  # per-track: highest score, time above 'fallen'
        m = per_track.setdefault(tid, [0.0, 0.0])
        m[0] = max(m[0], st["ema"])
        if st["ema"] >= getattr(plugin, "fallen_at", 9):
            m[1] += dt
    for ev in events:
        if ev["class"] == "pose_stats":
            continue
        counts[ev["class"]] = counts.get(ev["class"], 0) + 1
        x = ev["extra"]
        print(f"    t={f['ts']:6.1f}s  {ev['class']:13s} track {ev['track_id']}  "
              f"{'down_s=' + str(x['down_s']) if 'down_s' in x else ''}  {x.get('signals', '')}")
print(f"  {os.path.basename(rec)} | {name} {json.dumps(params) if params else '(defaults)'} | "
      f"{frames} frames | alerts: {counts or 'none'}")
for tid, (top, above) in sorted(per_track.items(), key=lambda kv: -kv[1][0])[:3]:
    print(f"    track {tid}: highest smoothed score {top:.2f}, {above:.1f} s above the fallen threshold")
