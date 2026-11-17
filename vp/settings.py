"""Reads and checks a node's settings file."""
from dataclasses import dataclass, field
import yaml


@dataclass
class Camera:
    id: str
    uri: str
    source_fps: int = 30
    analysis_fps: int = 5
    use_cases: dict = field(default_factory=dict)

    @property
    def drop_interval(self) -> int:
        # The decoder passes on every Nth frame: a 30 fps camera analysed at 5 fps gives 6.
        return max(1, round(self.source_fps / self.analysis_fps))


@dataclass
class Settings:
    node_id: str
    site_id: str
    engine: dict
    cameras: list
    data_dir: str = "data"
    camera_offline_after_s: float = 15.0
    models: list = field(default_factory=list)  # extra whole-frame models, e.g. PPE
    vlm: dict = field(default_factory=dict)  # VLM server: url, model, timeout_s
    hub: dict = field(default_factory=dict)  # hub event intake: url
    retention: dict = field(default_factory=dict)  # storage limits on the node
    model_versions: dict = field(default_factory=dict)  # use case -> model version label


def load(path: str, check_files=False) -> Settings:
    with open(path) as f:
        return from_dict(yaml.safe_load(f), check_files)


def _zone_problem(z):
    """A zone is a box [x1, y1, x2, y2] or a polygon of 3+ points [x, y], all in the frame (0 to 1)."""
    try:
        if len(z) == 4 and all(isinstance(v, (int, float)) for v in z):
            x1, y1, x2, y2 = z
            return None if 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1 else f"box {z} is not inside the frame"
        if len(z) < 3:
            return f"polygon with {len(z)} points (needs 3 or more)"
        if not all(len(pt) == 2 and -0.01 <= pt[0] <= 1.01 and -0.01 <= pt[1] <= 1.01 for pt in z):
            return "polygon has points outside the frame"
    except TypeError:
        return f"not a box or polygon: {str(z)[:40]}"
    return None


def check(s, check_files=False):
    """Everything that would crash the node or silently do nothing, as a list of plain-English problems.
    check_files: also check that settings files exist (on the jetson; tests skip it)."""
    from vp.plugins import REGISTRY
    from vp.schedule import Schedule
    import os
    out, ids = [], {c.id for c in s.cameras}
    for c in s.cameras:
        for uc, p in c.use_cases.items():
            if uc not in REGISTRY:
                out.append(f"{c.id}: unknown use case '{uc}'")
                continue
            p = p or {}
            zones = p.get("zones") or []
            for z in zones if isinstance(zones, list) else [zones]:
                if (why := _zone_problem(z)):
                    out.append(f"{c.id} {uc}: zone {why}")
            if p.get("zone") is not None and (why := _zone_problem(p["zone"])):
                out.append(f"{c.id} {uc}: zone {why}")
            try:
                Schedule(p.get("schedule"), p.get("timezone"))
            except Exception as e:
                out.append(f"{c.id} {uc}: schedule or timezone: {str(e)[:60]}")
            if getattr(REGISTRY[uc], "uses_vlm", False) and not s.vlm.get("url"):
                out.append(f"{c.id} {uc}: needs a VLM but settings have no vlm url")
    gie_ids = [m.get("gie_id") for m in s.models]
    if len(gie_ids) != len(set(gie_ids)) or 1 in gie_ids:
        out.append(f"models: gie_id must be unique and not 1 (the person detector's): {gie_ids}")
    for m in s.models:
        missing = [c for c in m.get("cameras", []) if c not in ids]
        if missing:
            out.append(f"model {m.get('name')}: cameras {missing} are not on this node")
        if check_files and not os.path.exists(m.get("config", "")):
            out.append(f"model {m.get('name')}: settings file {m.get('config')} not found")
    if check_files:
        for key in ("pgie_config", "tracker_config", "tracker_lib"):
            if s.engine.get(key) and not os.path.exists(s.engine[key]):
                out.append(f"engine: {key} {s.engine[key]} not found")
    return out


def from_dict(raw, check_files=False) -> Settings:
    """Settings from a dict; raises ValueError listing every problem (see check)."""
    from vp.rules import expand
    raw = expand(raw)  # CampNeuron's detection_types + rules -> per-camera settings
    cameras = [Camera(**c) for c in raw["cameras"]]
    ids = [c.id for c in cameras]
    if len(ids) != len(set(ids)):
        raise ValueError("camera ids must be unique")
    max_batch = raw["engine"]["max_batch"]
    if not 0 < len(cameras) <= max_batch:
        raise ValueError(f"{len(cameras)} cameras configured; the engine allows 1 to {max_batch}")
    s = Settings(raw["node_id"], raw["site_id"], raw["engine"], cameras, raw.get("data_dir", "data"),
                 raw.get("camera_offline_after_s", 15.0), raw.get("models", []), raw.get("vlm", {}), raw.get("hub", {}),
                 raw.get("retention", {}), raw.get("model_versions", {}))
    problems = check(s, check_files)
    if problems:
        raise ValueError(f"{len(problems)} problem(s): " + "; ".join(problems[:8]))
    return s
