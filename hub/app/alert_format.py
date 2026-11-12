"""CampNeuron's alert message (Kafka topic safety.alerts), built from an event stored on the hub.
Keep in step with vp/alert_format.py on the nodes (same fields)."""
DURATIONS = ("missing_s", "down_s", "at_height_s", "duration_s", "people", "present_s")


def to_safety_alert(e, evidence_base):
    x, model, conf = e.get("extra") or {}, e.get("model"), e.get("confidence")
    msg = {"rule": e.get("rule") or e["use_case"], "instance": e.get("instance") or e["camera_id"],
           "camera": e["camera_id"], "track_id": e.get("track_id"), "severity": e["severity"], "event": e["class"],
           "model": f"{model} · conf {conf:.2f}" if model and conf is not None else model,
           "snapshot": f"{evidence_base}{e['event_id']}" if e.get("has_evidence") else None, "clip": None,
           "time": e["timestamp"], "incident_id": e.get("incident_id"), "event_id": e["event_id"],
           "node": e.get("node_id"), "site": e.get("site_id")}
    if e["use_case"] == "health" and e["camera_id"] == "hub":  # the hub's own disk_low / disk_ok
        msg.update(rule="Hub health", instance=e.get("node_id"), camera=None)
    elif e["use_case"] == "health" and e["camera_id"] == "node":  # node_down / node_up: name the node
        msg.update(rule="Edge node health", instance=e.get("node_id"), camera=None)
    elif e["use_case"] == "health":  # camera_offline / camera_online: name the camera
        msg.update(rule="Camera health", instance=e["camera_id"])
    msg.update({k: x[k] for k in DURATIONS if k in x})
    return msg
