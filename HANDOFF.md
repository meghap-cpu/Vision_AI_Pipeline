# Vision platform: handoff (as of Day 11, 6 Nov 2026)

Read this first. It carries the work from the previous chat (Days 1–11).

## Goal

AI safety monitoring for camps. CCTV cameras → Jetson edge nodes (detect) → hub (collect) → Kafka `safety.alerts` → CampNeuron (incidents, escalation). Settings flow back: CampNeuron rule engine → hub → nodes. Target: 64+ cameras, ~16 per Jetson, alerts within seconds, unattended for days, nothing lost when the network or hub drops.

## Environment (pilot Jetson, everything runs here)

- Jetson AGX Orin 64GB · JetPack 6.2.1 · L4T R36.4.7 · CUDA 12.6 · TensorRT 10.3 · DeepStream 7.1 · pyds 1.2.0 · system Python 3.10 (`/usr/bin/python3`, run with `conda deactivate`)
- Project: `~/vision-pipeline` (GitHub `meghap-cpu/Vision_AI_Pipeline`, last push Day 9; `data/`, `models/`, logs ignored)
- Models and engines: `~/vision-trial/` (YOLO26s, PPE merged RF-DETR, RTMPose); test clip `tools/test_clip.mp4`
- Test cameras: `tools/start_test_cameras.sh 16` (mediamtx + ffmpeg loops, RTSP :8554); stand-in VLM: `tools/standin_vlm.py --port 8001 --delay 1.7` with answer file `/tmp/standin_vlm.json`
- Real VLM: vLLM Qwen3-VL-4B on a separate Jetson `192.168.0.119:8010` (currently DOWN)
- eMMC 57 GB, ~9–12 GB free. NVMe SSD not fitted yet (needed for the pipeline container, 21 GB image)
- User is in the `docker` group as of Day 11; until re-login use `sg docker -c '...'`

## Architecture on the node (`vp/`)

- `main.py` – DeepStream trunk: nvurisrcbin per camera → nvstreammux → PGIE YOLO26s (people + vehicles) → nvtracker (NvSORT) → PPE PGIE (whole frame, routed cameras, nvdspreprocess) → RTMPose SGIE (man-down cameras, enlarged crop) → probe → plugin worker processes; memory guard (exit 3 above `retention.max_memory_gb`, 8 GB)
- `settings.py` + `rules.py` – settings; `rules.py` expands CampNeuron's `detection_types` + `rules` into per-camera `use_cases` (old style still works)
- `remote_config.py` – settings from the hub (`config_url`), local copy `data/config_cache.yaml`, exit code 3 on change
- `worker.py` – one process per use case; schedules (pause, clear state, `schedule_paused/resumed`); injects rule/instance/model; rule severity overrides warning/critical
- Plugins (`vp/plugins/`): motion, crowd, housekeeping, ppe (per-item rules, stability N of M, min size, duration/cooldown, zones), man_down, fire_smoke (VLM), worker_at_height (VLM), test
- `outbox.py` (SQLite, indexes), `uploader.py` (critical first, retries ≤10 s), `snapshots.py` (writer never dies), `retention.py` (2 GB snapshots, 7 days sent events, never deletes undelivered), `health.py` (every 30 s), `watchdog.py`, `alert_format.py`, `vlm.py`, `schedule.py`
- Run: `tools/run_node.sh` (dev) or systemd `vision-node` (installed, starts `config/bootstrap.yaml` at boot)

## Hub on the Jetson (`hub/`, docker compose)

- `api-dashboard` (FastAPI) + `postgres` + local `kafka` (apache/kafka 3.8.0), port 8100
- Endpoints: `POST /api/v1/events`, `POST /api/v1/evidence/{id}`, `POST /api/v1/nodes/{id}/health`, `GET /api/v1/nodes/{id}/config` (serves `hub/configs/<node>.yaml`, CampNeuron shape), `GET /api/v1/nodes`, `GET /api/v1/incidents`, `/health`, `/`
- `app/background.py`: Kafka publisher (CampNeuron format, published only after ack), node monitor (`node_down` critical / `node_up`)
- `hub/.env`: POSTGRES_PASSWORD, SITE_TZ=Asia/Dubai, NODE_DOWN_AFTER_S=30 (test; 120 in production). Production also needs HUB_PUBLIC_URL and KAFKA_BOOTSTRAP for CampNeuron's Kafka
- Last fix: camera_offline/online should publish as rule "Camera health" with the camera named (patch given; check it was applied and the hub rebuilt)

## Proven (pilot gate)

24 h soak (no restarts, fps 5, storage bounded), camera failure, hub outage, VLM outage, settings change + hub down at start, reboot (node back in 18 s from local copy, cameras reconnect), end to end node → hub → Kafka in CampNeuron's format, node_down/up.

## Open items

- Memory: native growth 0.5–1 GB/day, not fragmentation; bounded by the guard; source unknown (elimination runs later)
- Pilot drills still to do: network cable out and back; disk nearly full live; realistic run with real VLM and camp footage
- Small: stand-in VLM should answer all question types by default; `debug_every_s` off for man-down in production
- Waiting on the manager: CampNeuron Kafka address and access; harness check = PPE model or VLM; worker-at-height rule severity (it escalates itself); man-down, crowd, housekeeping as rules; video clips (his design has them; we deferred); NvSORT; false-positive feedback from CampNeuron
- Waiting on hardware/data: NVMe SSD, VLM Jetson, labelled man-down clips, camp PPE images, safety team timings
- Later: spill and machinery models, crack/corrosion batch, sending new models to nodes, sign-in/TLS, multi-node failover

## CampNeuron (manager's design)

Screens: Detection Types (AI team: model, version, confidence, stability "N of M frames", min object px, calibrations, false positive %, pilot/production, publish), Safety Rules (rule → detection type → instances per site/camera, schedule, severity, Active/Draft), Incident Master, Incident Listing (source AI Detection/Manual). Kafka message: rule, instance, camera, track_id, severity, model "PPE v2.3 · conf 0.88", snapshot, clip, time with local offset.

## How we work (keep this)

- One step at a time: short plan table, then one command block, then "done when" checks. User pastes output.
- Patch scripts check for exactly one match and print `patched` / `already done`.
- Long runs write to a log file, never through `| grep | tail` (output gets swallowed).
- Never `rm -rf` a previous run's data: archive it (`tools/soak_start.sh`). `/tmp` clears on reboot.
- Read the real code before fixing it; don't guess.
- Everything on the Jetson, not the PC.
- Documents: feature log per day in the user's own style (short, plain, "jetson", "cam", "sec", " - " after the name); design guide still to be rewritten in that style once she OKs the log.

## Living documents

- Feature log: https://claude.ai/artifact/86Jomn6Vv4MAWmZ6MRhYbk
- Design guide: https://claude.ai/artifact/RdQb3WdEg6U8rgyYyKD2Jg
- Deck: https://claude.ai/artifact/Ky3zpQfxcAWD8iCEjefEPt

## Next steps

1. Check the hub "Camera health" fix is applied.
2. Drills: network cable out and back; disk nearly full live.
3. Design guide rewrite in the user's style.
4. Send the manager the open questions above.
