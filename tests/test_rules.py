"""CampNeuron's settings shape: expansion into per-camera settings, and PPE with per-item rules.
Run: python3 -m tests.test_rules"""
from vp.plugin import Frame, Obj
from vp.plugins.ppe import PPEPlugin
from vp.rules import expand


def run():
    raw = {"engine": {"height": 720},
           "cameras": [{"id": "cam08", "uri": "x"}, {"id": "cam11", "uri": "y"}],
           "detection_types": {"ppe": {"model": "PPE", "version": "v2.3", "confidence": 0.75, "stability": [8, 10],
                                       "min_object_px": 48, "calibrations": {"cam08": {"confidence": 0.8}}},
                               "fire_smoke": {"model": "Fire & Smoke", "version": "v3.0"},
                               "line": {"model": "Line Crossing", "version": "v1.0", "use_case": "line_crossing"}},
           "rules": [{"name": "Helmet Compliance", "detection": "ppe", "item": "hardhat", "severity": "High",
                      "instances": [{"camera": "cam08", "instance": "Electric room", "duration_s": 10}]},
                     {"name": "Safety Vest", "detection": "ppe", "item": "vest", "severity": "Medium",
                      "instances": [{"camera": "cam08", "instance": "Electric room yard"}]},
                     {"name": "Fire and Smoke", "detection": "fire_smoke", "severity": "High",
                      "instances": [{"camera": "cam11", "instance": "Kitchen"}]},
                     {"name": "Line Crossing - Gates", "detection": "line", "status": "Draft",
                      "instances": [{"camera": "cam11"}]}]}
    cams = {c["id"]: c["use_cases"] for c in expand(raw)["cameras"]}
    ppe = cams["cam08"]["ppe"]
    assert ppe["model"] == "PPE v2.3" and ppe["min_confidence"] == 0.8 and ppe["stability_frames"] == [8, 10]
    assert abs(ppe["min_person_h"] - 48 / 720) < 1e-9
    assert ppe["items"]["hardhat"]["rule"] == "Helmet Compliance" and ppe["items"]["hardhat"]["severity"] == "critical"
    assert ppe["items"]["vest"]["instance"] == "Electric room yard" and ppe["items"]["vest"]["severity"] == "warning"
    assert cams["cam11"]["fire_smoke"] == {"model": "Fire & Smoke v3.0", "instance": "Kitchen",
                                           "rule": "Fire and Smoke", "severity": "critical"}
    assert "line_crossing" not in cams["cam11"], "Draft rules must not run"
    print("expansion: models, calibration, stability, min size, per-item rules, severities, Draft skipped: OK")

    plugin = PPEPlugin("cam08", ppe)
    box = (0.40, 0.20, 0.50, 0.80)      # 0.6 of the frame tall: bigger than 48 px
    tiny = (0.70, 0.50, 0.72, 0.55)     # 36 px tall: ignored
    got = []
    for n in range(45):                 # 3 checks/s for 15 s
        t = n / 3
        objs = [Obj(2, 0, 0.9, box, source=1), Obj(3, 0, 0.9, tiny, source=1),
                Obj(0, 1, 0.85, (0.41, 0.20, 0.49, 0.28), source=3),   # No_Hardhat, conf 0.85 (>= 0.8)
                Obj(0, 1, 0.70, (0.41, 0.20, 0.49, 0.28), source=3),   # No_Hardhat, conf 0.70 (< 0.8): ignored
                Obj(0, 1, 0.9, (0.70, 0.50, 0.72, 0.51), source=3)]    # on the tiny person: ignored
        for ev in plugin.on_frame(Frame("cam08", n, t, objs)):
            got.append((round(t, 1), ev["class"], ev["track_id"], ev.get("rule"), ev.get("instance"), ev["severity"]))
    print("events:", got)
    assert len(got) == 1 and got[0][1:] == ("no_hardhat", 2, "Helmet Compliance", "Electric room", "critical"), got
    assert 12.5 <= got[0][0] <= 13.5, got  # 10 checks to be sure (~3 s) + 10 s duration
    print("PASS: CampNeuron's rules expand correctly; PPE alerts under its own rule, place and severity after "
          "8 of 10 frames + 10 s; low confidence, tiny people and the vest (no vest boxes) don't alert")


if __name__ == "__main__":
    run()
