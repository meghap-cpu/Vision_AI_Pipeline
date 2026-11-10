"""CampNeuron's settings shape -> the pipeline's per-camera settings, so the plugins don't change.
detection_types (AI team, "Detection Types" screen): model, version, confidence, stability [N, M] (N of M frames),
  min_object_px, calibrations {camera: {confidence}}, use_case (default: the detection type's key).
rules (CampNeuron, "Safety Rules" screen): name, detection, severity (High/Medium/Low), status (Active/Draft),
  item (PPE: hardhat or vest), instances [{camera, instance (place name), zones, schedule, duration_s, cooldown_s, ...}].
Old-style cameras[].use_cases still work and can be mixed in."""
SEVERITY = {"high": "critical", "critical": "critical", "medium": "warning", "low": "warning", "warning": "warning"}
PER_ITEM = {"ppe"}  # one plugin, one rule per item


def expand(raw):
    rules = raw.get("rules") or []
    if not rules:
        return raw
    types, height = raw.get("detection_types") or {}, (raw.get("engine") or {}).get("height", 720)
    cams = {c["id"]: c for c in raw.get("cameras", [])}
    for rule in rules:
        if str(rule.get("status", "active")).lower() != "active":
            continue  # Draft or paused rules don't run
        dt = types[rule["detection"]]
        uc = dt.get("use_case", rule["detection"])
        severity = SEVERITY.get(str(rule.get("severity", "")).lower())
        for inst in rule.get("instances", []):
            cam = cams[inst["camera"]]
            params = cam.setdefault("use_cases", {}).setdefault(uc, {}) or {}
            cam["use_cases"][uc] = params
            calib = (dt.get("calibrations") or {}).get(cam["id"], {})
            params["model"] = f"{dt.get('model', uc)} {dt.get('version', '')}".strip()
            if calib.get("confidence", dt.get("confidence")) is not None:
                params["min_confidence"] = calib.get("confidence", dt.get("confidence"))
            if dt.get("stability"):
                params["stability_frames"] = list(dt["stability"])
            if dt.get("min_object_px"):
                params["min_person_h"] = dt["min_object_px"] / height
            own = {k: v for k, v in inst.items() if k != "camera"}
            own.update(rule=rule["name"], **({"severity": severity} if severity else {}))
            if uc in PER_ITEM:
                params.setdefault("items", {})[rule["item"]] = own
            else:
                params.update(own)
    return raw
