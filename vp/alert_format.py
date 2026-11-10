"""CampNeuron's alert message (Kafka topic safety.alerts), built from an edge event as the hub receives it.
The hub publishes this to Kafka; the Jetsons keep sending their full events to the hub."""
DURATIONS = ("missing_s", "down_s", "at_height_s", "duration_s", "people", "present_s")


def to_safety_alert(e, evidence_base="evidence/"):
    x, model, conf = e.get("extra") or {}, e.get("model"), e.get("confidence")
    msg = {"rule": e.get("rule") or e["use_case"], "instance": e.get("instance") or e["camera_id"],
           "camera": e["camera_id"], "track_id": e.get("track_id"), "severity": e["severity"], "event": e["class"],
           "model": f"{model} · conf {conf:.2f}" if model and conf is not None else model,
           "snapshot": f"{evidence_base}{e['event_id']}.jpg" if e.get("snapshot_ref") else None, "clip": None,
           "time": e["timestamp"], "incident_id": e.get("incident_id"), "event_id": e["event_id"]}
    msg.update({k: x[k] for k in DURATIONS if k in x})
    return msg
