"""Makes a new node's settings from an existing node's: picks some cameras, gives them new IDs everywhere
(camera list, rules, calibrations), renumbers their test streams from cam0, and sets the node name, hub and VLM.

  python3 tools/make_node_config.py hub/configs/pilot-01.yaml hub/configs/pilot-02.yaml --node-id pilot-02 \
      --cams cam01,cam03,cam05 --first-id 16 --hub http://192.168.0.154:8100 \
      [--vlm http://HOST:8001/v1] [--vlm-backup http://HOST:8002/v1]
Camera IDs must be unique across the site, so pick --first-id past the IDs other nodes use."""
import argparse, re, yaml

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("dst")
ap.add_argument("--node-id", required=True); ap.add_argument("--cams", required=True)
ap.add_argument("--first-id", type=int, required=True); ap.add_argument("--hub", required=True)
ap.add_argument("--vlm"); ap.add_argument("--vlm-backup")
a = ap.parse_args()

d = yaml.safe_load(open(a.src))
by_id = {c["id"]: c for c in d["cameras"]}
picked = a.cams.split(",")
missing = [c for c in picked if c not in by_id]
if missing:
    raise SystemExit(f"not in {a.src}: {missing}")
new_id = {old: f"cam{a.first_id + k:02d}" for k, old in enumerate(picked)}

cams = []
for k, old in enumerate(picked):
    c = dict(by_id[old], id=new_id[old])
    c["uri"] = re.sub(r"/cam\d+$", f"/cam{k}", c["uri"])  # this node's own test streams start at cam0
    cams.append(c)
d["cameras"], d["node_id"] = cams, a.node_id
d.setdefault("hub", {})["url"] = a.hub.rstrip("/")
if a.vlm:
    d.setdefault("vlm", {})["url"] = a.vlm
if a.vlm_backup:
    d.setdefault("vlm", {})["backup_url"] = a.vlm_backup

models = []
for m in d.get("models") or []:  # extra models (PPE, pose) list the cameras they run on
    cs = [new_id[c] for c in m.get("cameras", []) if c in new_id]
    if cs:
        models.append(dict(m, cameras=cs))  # a model with none of our cameras is left out: no wasted GPU
dropped_models = [m.get("name") for m in d.get("models") or [] if not any(c in new_id for c in m.get("cameras", []))]
d["models"] = models

for dt in (d.get("detection_types") or {}).values():
    if isinstance(dt, dict) and isinstance(dt.get("calibrations"), dict):
        dt["calibrations"] = {new_id[k]: v for k, v in dt["calibrations"].items() if k in new_id}
kept = 0
for r in d.get("rules") or []:
    out = []
    for inst in r.get("instances") or []:
        if "camera" in inst:
            if inst["camera"] in new_id:
                out.append(dict(inst, camera=new_id[inst["camera"]]))
        elif "cameras" in inst:
            cs = [new_id[c] for c in inst["cameras"] if c in new_id]
            if cs:
                out.append(dict(inst, cameras=cs))
        else:
            out.append(inst)  # not tied to a camera
    r["instances"] = out
    kept += len(out)

# Self-check: no camera name from the source node may be left anywhere (stream paths like /cam0 are fine)
text = yaml.safe_dump(d, sort_keys=False, allow_unicode=True)
stray = sorted(set(re.findall(r"(?<![/\w])cam\d{2}\b", text)) - set(new_id.values()))
if stray:
    raise SystemExit(f"stopped, nothing written: these cameras are not on {a.node_id}: {stray}")

with open(a.dst, "w") as f:
    f.write(f"# Made by tools/make_node_config.py from {a.src} ({', '.join(f'{o}->{n}' for o, n in new_id.items())})\n")
    f.write(text)
print(f"{a.dst}: node {a.node_id}, {len(cams)} cameras {cams[0]['id']}..{cams[-1]['id']}, {kept} rule instances kept, "
      f"models {[m.get('name') for m in models]}" + (f" (left out: {dropped_models})" if dropped_models else "") + f", hub {d['hub']['url']}")
