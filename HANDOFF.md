# Vision platform: handoff (as of Day 12, 10 Nov 2026)

Read this first. It carries the work from the previous chats (Days 1–12).

## Goal

AI safety monitoring for camps. CCTV cameras → Jetson edge nodes (detect) → hub (collect) → Kafka `safety.alerts` → CampNeuron (incidents, escalation). Settings flow back: CampNeuron rule engine → hub → nodes. Target: **~100 cameras**, ~16 per edge Jetson, alerts within seconds, unattended for days, nothing lost when the network or hub drops. Target picture: the "journey of a frame" diagram (11 steps) and the 3-container sketch the user shared on Day 12.

## Environment (pilot Jetson, everything runs here)

- Jetson AGX Orin 64GB · JetPack 6.2.1 · L4T R36.4.7 · CUDA 12.6 · TensorRT 10.3 · DeepStream 7.1 · pyds 1.2.0 · system Python 3.10 (`/usr/bin/python3`, run with `conda deactivate`)
- Project: `~/vision-pipeline` (GitHub `meghap-cpu/Vision_AI_Pipeline`, pushed Day 12; `data/`, `models/`, logs, `hub/.env` ignored)
- Models and engines: `~/vision-trial/` (YOLO26s, PPE merged RF-DETR, RTMPose); test clip `tools/test_clip.mp4`
- Test cameras: `tools/start_test_cameras.sh 16` (mediamtx + ffmpeg loops, RTSP :8554); stand-in VLM: `nohup /usr/bin/python3 tools/standin_vlm.py --port 8001 --delay 1.7 > /tmp/standin_vlm.log 2>&1 &` (default answer now covers fire, harness, helmet, vest; `/tmp/standin_vlm.json` overrides it)
- **After any reboot: start the test cameras and the stand-in VLM first.** They are not services. Day 12 started with the Jetson rebooted at 10:38 and both missing, which looked like 16 dead cameras and false worker-at-height alerts.
- Real VLM: vLLM Qwen3-VL-4B on a separate Jetson `192.168.0.119:8010` (currently DOWN)
- eMMC 57 GB, **7.7 GB free on Day 12** (was 9–12 GB). Likely old hub images from `--build`; check `docker image prune` before the next soak. NVMe SSD not fitted yet (needed for the 21 GB pipeline container)
- Times: Jetson clock is IST (+05:30); hub/Kafka times are Dubai (+04:00, `SITE_TZ`)

## Architecture on the node (`vp/`)

- `main.py` – DeepStream trunk: nvurisrcbin per camera (reconnect every 10 s; one camera retried about every 3 s) → nvstreammux → PGIE YOLO26s → nvtracker (NvSORT) → PPE PGIE (whole frame, helmet + vest only) → RTMPose SGIE → probe → plugin worker processes; memory guard (exit 3 above 8 GB)
- `settings.py` + `rules.py` – CampNeuron `detection_types` + `rules` → per-camera `use_cases`. A rule's severity overrides plugin warning/critical, so **keep "Harness at Height" severity blank** (it has its own levels)
- `remote_config.py` – settings from the hub, local copy `data/config_cache.yaml`, exit 3 on change
- `worker.py` – one process per use case; schedules
- `watchdog.py` – camera_offline (warning) / camera_online now share one incident (`pilot-01:camXX:health:offline:<since>`); camera_online closes it
- Plugins: motion, crowd, housekeeping, ppe (helmet + vest), man_down, fire_smoke (VLM), worker_at_height (VLM), test
- **worker_at_height (Day 12, as the original repo + vest):** VLM checks harness, helmet, vest; any one missing = violation. `at_height_entered` = Low (sent as warning) after `entered_after_s` (5 s) in the zone, skipped if already High; `at_height_violation` / `at_height_critical` (near edge) / `at_height_unverified` (no answer in 10 s, 3 s near edge) = High (critical). On leaving: Low-only incident closes (`at_height_end`); one that reached High stays open for review (`at_height_left`, info, not an END). Re-check every 300 s inside the same incident
- `outbox.py`, `uploader.py`, `snapshots.py`, `retention.py` (warns `disk_low` under 2 GB free, hourly; never deletes undelivered), `health.py` (every 30 s), `alert_format.py`, `vlm.py`, `schedule.py`
- Run: systemd `vision-node` (starts `config/bootstrap.yaml` at boot); `tools/run_node.sh` for dev
- Drills: `tools/net_drill.sh [s]` (blocks ports 8100/8554/8001 on lo, always restores, then restarts the ffmpeg loops), `tools/disk_drill.sh [leave_gb] [hold_s]` (fills to leave_gb free, always frees)

## Hub on the Jetson (`hub/`, docker compose)

- `api-dashboard` (FastAPI) + `postgres` + local `kafka`, port 8100
- `app/background.py`: Kafka publisher (warning/critical + ENDS only; info is marked published without sending), node monitor. `ENDS` now includes `camera_online`
- `app/alert_format.py`: camera health → rule "Camera health", node → "Edge node health" (Day 11 fix, confirmed applied and rebuilt Day 12)
- `hub/.env`: **NODE_DOWN_AFTER_S=90** on the test hub (30 equalled the report interval and gave false node_down/up in the same second); 120 in production. Changing `.env` needs `docker compose up -d api-dashboard`
- Postgres check: `sg docker -c "docker compose exec -T postgres psql -U vision -d vision -c \"…\""` (faster and more reliable than reading Kafka with the console consumer, which hangs)
- `/api/v1/incidents` LIMIT 100 is filled by crowd incidents from the looping test clip; query Postgres to find a specific incident

## Proven (pilot gate)

24 h soak, storage cap, camera failure, hub outage, VLM outage, settings change + hub down at start, reboot (18 s), node_down/up, end to end to Kafka. **Day 12:** network cut 2 min (simulated on one box: nothing lost 17/17, node down/up, all 16 cams back), disk nearly full (1.5 GB free 5.5 min: disk_low sent, nothing lost 401+1), camera health incidents open and close, healthy restart gives 0 false camera_offline.

## Decisions on Day 12 (user)

- ~100 cameras; fire and smoke on all; worker at height only where zones exist
- VLM on its own Jetsons, shared; a backup VLM Jetson takes over if one fails
- Alert time targets are set by us per incident type (market research + severity), not by the manager. Table with `[__ sec]` in the design guide
- PPE = helmet + vest by model. Worker at height = harness + helmet + vest by VLM
- Worker at height: Low on entry (after 5 s dwell), High on any missing item; video clips later; tracker "let's see"

## Sizing for 100 cameras (estimate)

~7 edge Jetsons, 2–4 VLM Jetsons (fire ≈ 25 checks/s; one VLM Jetson measured 6.2/s with only 4 concurrent, fire only, 512 px), hub, spare. First thing when the VLM Jetson is back: rerun `tools/vlm_bench.py` part C with 8/16/32 concurrent. Network ~25–30 Mbit/s.

## Open items

- Memory: native growth (0.5–1.6 GB/day across runs), bounded by the guard; source unknown
- Realistic run (real VLM, camp footage, NVMe SSD); alert time targets to fill
- Near-edge 3 s limit: 3 unverified Highs in a test (answer 1.7 s + frame wait) — safety team
- Vest added to the VLM prompt — verify on the real VLM
- Small fixes: one "VLM down" alert instead of one per fire camera per 60 s; disk_low needs a "disk OK" close; backup VLM address per node; Kafka duration field `missing_s` vs design's `no_helmet_seconds`
- Gaps vs target picture: helmet/vest classifier on person crops (biggest), clips (30 s buffer), containers (SSD), config-agent split (optional), camera credentials encrypted, local alarm, red zone and spill models, crack/corrosion
- Manager message drafted Day 12 (user answered most herself). Still open: hub on its own Jetson; false-positive feedback; man-down/crowd/housekeeping/motion as CampNeuron rules (suggested yes)
- Waiting on hardware/data: NVMe SSD, VLM Jetson, labelled man-down clips, camp PPE images, camp footage, safety team timings

## How we work (keep this)

- One step at a time: short plan table, then one command block, then "done when" checks. User pastes output; wait for `--- all done` before reading results.
- Patch scripts check for exactly one match and print `patched` / `already done`. Large patches are attached as files (saved to `~/Downloads`, copied to `tools/`); test them on a copy of the code before handing over.
- Long runs write to a log file, never through `| grep | tail`.
- Never `rm -rf` a previous run's data: archive it (`tools/soak_start.sh`). `/tmp` clears on reboot.
- Read the real code before fixing it; don't guess. Compare against the original repos when porting (the user shared `woker-at-height.zip` on Day 12).
- Everything on the Jetson, not the PC.
- Documents: feature log per day in the user's style (short, plain, "jetson", "cam", "sec", " - " after the name). Style OK'd Day 12; design guide rewritten in it on Day 12.

## Living documents

- Feature log (Day 12 added): https://claude.ai/artifact/86Jomn6Vv4MAWmZ6MRhYbk
- Design guide (rewritten Day 12, topology diagram redrawn): https://claude.ai/artifact/RdQb3WdEg6U8rgyYyKD2Jg
- Deck (slide 9 "every edge node gets its own VLM" is out of date): https://claude.ai/artifact/Ky3zpQfxcAWD8iCEjefEPt

## Next steps

1. Fill the alert time targets (design guide table).
2. Small fixes: VLM down alert, disk_low close, backup VLM address.
3. When the VLM Jetson is back: concurrency benchmark, vest check, realistic run.
4. Free eMMC space (`docker image prune`), then decide on the next soak.
5. Update deck slide 9 (VLM on its own Jetsons).
