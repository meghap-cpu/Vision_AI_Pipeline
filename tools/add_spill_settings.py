"""Adds the spill model, detection type and rule to a node's settings file, keeping its comments.
Usage: python3 add_spill_settings.py hub/configs/pilot-01.yaml cam05,cam06"""
import sys, pathlib
path, cams = pathlib.Path(sys.argv[1]), sys.argv[2].split(",")
t = path.read_text()
if "name: Spill Detection" in t:
    print(f"{path}: already done"); sys.exit()
model = ("- name: spill\n  config: config/pgie_spill.txt\n  gie_id: 5\n"
         "  interval: 4   # every 5th frame (~1 check/s per camera at 5 fps); no cameras list = all cameras\n")
dtype = "  spill:\n    model: Spill\n    version: v1 (yolo26n)\n    confidence: 0.45\n"
rule = ("- name: Spill Detection\n  detection: spill\n  severity: Medium\n  status: Active\n  instances:\n"
        + "".join(f"  - camera: {c}\n    instance: Spill area {i + 1}\n" for i, c in enumerate(cams)))
for anchor in ("\nvlm:\n", "\nrules:\n"):
    if t.count(anchor) != 1:
        sys.exit(f"{path}: expected one '{anchor.strip()}', found {t.count(anchor)}; stopping, nothing written")
t = t.replace("\nvlm:\n", "\n" + model + "vlm:\n").replace("\nrules:\n", "\n" + dtype + "rules:\n")
t = t.rstrip("\n") + "\n" + rule
path.write_text(t); print(f"{path}: spill model, detection type and rule on {cams} added")
