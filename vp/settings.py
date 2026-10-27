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


def load(path: str) -> Settings:
    with open(path) as f:
        raw = yaml.safe_load(f)
    cameras = [Camera(**c) for c in raw["cameras"]]
    ids = [c.id for c in cameras]
    if len(ids) != len(set(ids)):
        raise ValueError("camera ids must be unique")
    max_batch = raw["engine"]["max_batch"]
    if not 0 < len(cameras) <= max_batch:
        raise ValueError(f"{len(cameras)} cameras configured; the engine allows 1 to {max_batch}")
    return Settings(raw["node_id"], raw["site_id"], raw["engine"], cameras, raw.get("data_dir", "data"),
                    raw.get("camera_offline_after_s", 15.0), raw.get("models", []), raw.get("vlm", {}), raw.get("hub", {}))
