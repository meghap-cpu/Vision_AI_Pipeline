"""Failover: while a node is down, its spare is also given that node's cameras.

configs/failover.yaml says who covers whom, each node's address, and how many cameras a node can run:

    max_cameras: 16                 # the engines' largest batch: more would force a slow engine rebuild
    nodes:
      pilot-01: {spare: pilot-02, address: 192.168.0.139}
      pilot-02: {spare: pilot-01, address: 192.168.0.154}

Nothing is stored: when a node asks for its settings, its own file is served plus the cameras of every partner
that is down right now (nodes.down_since), as many as fit. When the partner is back, the cameras drop out of the
settings, and the spare restarts without them (nodes restart when their settings change).
A camera written as 127.0.0.1 / localhost means "on that node's own machine", so it gets the node's address.
Camera IDs are unique across the site, so a camera keeps its name (and its alerts) whichever node watches it.
"""
import re
from pathlib import Path

import yaml

LOCAL = re.compile(r"//(127\.0\.0\.1|localhost)(?=[:/])")
# When a spare has no room for every camera, the most serious use cases are covered first
PRIORITY = {"fire_smoke": 0, "man_down": 1, "worker_at_height": 2, "ppe": 3, "crowd": 4, "housekeeping": 5,
            "motion": 6, "line": 6}
NO_RULE = 9


def load_map(configs: Path):
    f = configs / "failover.yaml"
    if not f.exists():
        return {"max_cameras": 16, "nodes": {}}
    m = yaml.safe_load(f.read_text()) or {}
    return {"max_cameras": int(m.get("max_cameras", 16)), "nodes": m.get("nodes") or {}}


def _camera_refs(inst):
    return inst.get("cameras") if "cameras" in inst else ([inst["camera"]] if "camera" in inst else [])


def _rank(partner):
    """Each camera's most serious use case (lower = more serious), from its own use cases and the rules."""
    rank = {c["id"]: min([PRIORITY.get(u, NO_RULE - 1) for u in (c.get("use_cases") or {})] or [NO_RULE])
            for c in partner.get("cameras", [])}
    for r in partner.get("rules") or []:
        for i in r.get("instances") or []:
            for cam in _camera_refs(i):
                if cam in rank:
                    rank[cam] = min(rank[cam], PRIORITY.get(r.get("detection"), NO_RULE - 1))
    return rank


def merge(spare, partner, address, room):
    """spare's settings plus up to `room` of partner's cameras. Returns (settings, taken IDs, not taken IDs)."""
    own = {c["id"] for c in spare.get("cameras", [])}
    cams = [dict(c) for c in partner.get("cameras", []) if c["id"] not in own]  # never two of the same camera
    for c in cams:
        if address:
            c["uri"] = LOCAL.sub(f"//{address}", c["uri"])
    rank = _rank(partner)
    cams.sort(key=lambda c: rank.get(c["id"], NO_RULE))  # stable: equal rank keeps the original order
    take, left = cams[:max(room, 0)], sorted(c["id"] for c in cams[max(room, 0):])
    taken = {c["id"] for c in take}
    if not take:
        return spare, [], left
    out = dict(spare, cameras=list(spare.get("cameras", [])) + take)

    models = [dict(m, cameras=list(m.get("cameras", []))) for m in spare.get("models") or []]
    for m in partner.get("models") or []:
        cs = [c for c in m.get("cameras", []) if c in taken]
        if not cs:
            continue
        mine = next((x for x in models if x.get("name") == m.get("name")), None)
        if mine:
            mine["cameras"] += cs
        else:
            models.append(dict(m, cameras=cs))
    out["models"] = models

    dts = {k: (dict(v) if isinstance(v, dict) else v) for k, v in (spare.get("detection_types") or {}).items()}
    for k, v in (partner.get("detection_types") or {}).items():
        cal = {c: x for c, x in ((v or {}).get("calibrations") or {}).items() if c in taken} if isinstance(v, dict) else {}
        if k not in dts:
            dts[k] = dict(v, calibrations=cal) if isinstance(v, dict) else v
        elif cal and isinstance(dts[k], dict):
            dts[k]["calibrations"] = {**(dts[k].get("calibrations") or {}), **cal}
    out["detection_types"] = dts

    rules = [dict(r, instances=list(r.get("instances") or [])) for r in spare.get("rules") or []]
    for r in partner.get("rules") or []:
        insts = []
        for i in r.get("instances") or []:
            refs = [c for c in _camera_refs(i) if c in taken]
            if refs:
                insts.append(dict(i, cameras=refs) if "cameras" in i else i)
        if not insts:
            continue
        mine = next((x for x in rules if x.get("name") == r.get("name")), None)
        if mine:
            mine["instances"] += insts
        else:
            rules.append(dict(r, instances=insts))
    out["rules"] = rules
    return out, sorted(taken), left


def settings_for(node_id, configs: Path, down_nodes):
    """(settings text, cover report) for node_id: its own file plus what it covers for down partners."""
    own_text = (configs / f"{node_id}.yaml").read_text()
    fmap = load_map(configs)
    partners = [p for p, v in fmap["nodes"].items() if (v or {}).get("spare") == node_id and p in down_nodes]
    if not partners:
        return own_text, []
    settings = yaml.safe_load(own_text)
    report = []
    for p in sorted(partners):
        pfile = configs / f"{p}.yaml"
        if not pfile.exists():
            continue
        room = fmap["max_cameras"] - len(settings.get("cameras", []))
        settings, taken, left = merge(settings, yaml.safe_load(pfile.read_text()), fmap["nodes"][p].get("address"), room)
        report.append({"for": p, "covered": taken, "not_covered": left})
    header = "".join(f"# failover: covering {r['for']}: {len(r['covered'])} cameras {r['covered']}"
                     f"{', NOT covered ' + str(r['not_covered']) if r['not_covered'] else ''}\n" for r in report)
    return header + yaml.safe_dump(settings, sort_keys=False, allow_unicode=True), report


def plan(down_node, configs: Path, down_nodes):
    """What happens when down_node goes down: (spare, covered IDs, not covered IDs). Used for the alert."""
    fmap = load_map(configs)
    spare = (fmap["nodes"].get(down_node) or {}).get("spare")
    if not spare or not (configs / f"{spare}.yaml").exists() or spare in down_nodes:
        cams = yaml.safe_load((configs / f"{down_node}.yaml").read_text()).get("cameras", []) \
            if (configs / f"{down_node}.yaml").exists() else []
        return spare, [], [c["id"] for c in cams]
    _, report = settings_for(spare, configs, set(down_nodes) | {down_node})
    r = next((x for x in report if x["for"] == down_node), {"covered": [], "not_covered": []})
    return spare, r["covered"], r["not_covered"]
