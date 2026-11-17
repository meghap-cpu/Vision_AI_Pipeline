"""Runs a node: pipeline from settings, frames to plugin workers, events to the outbox with snapshots."""
import multiprocessing as mp
from dataclasses import replace
import os
import queue
import signal
import sys
import threading
import time
from collections import Counter

import gi
gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst
import numpy as np
import pyds

from vp import worker
from vp.health import HealthReporter
from vp.outbox import Outbox
from vp.retention import Retention
from vp.uploader import Uploader
from vp.pipeline import build
from vp.plugin import Frame, Obj
from vp.pixels import crop_color, crop_zone
from vp.pose import decode, pose_box, tensors
from vp.plugins import REGISTRY
from vp.remote_config import GOOD_AFTER_S, RESTART_CODE, load_node, mark_good, watch
from vp.pipeline_config import missing_engines
from vp.router import CropStore, Router, SnapshotRequests
from vp.supervisor import ErrorCounter, RestartPolicy
from vp.settings import load
from vp.snapshots import Snapshots
from vp.watchdog import CameraWatchdog
from vp.vlm_watch import VLMWatch


class Worker:
    """One use case's plugin process. Every (re)start gets fresh queues, so a killed
    process can never leave a stuck lock behind for its replacement."""

    def __init__(self, ctx, use_case, cameras):
        self.ctx, self.use_case, self.cameras = ctx, use_case, cameras
        self.process = self.frames_q = self.events_q = None
        self.start()

    def start(self):
        for q in (self.frames_q, self.events_q):
            if q is not None:
                q.cancel_join_thread()
        self.frames_q = self.ctx.Queue(maxsize=200)
        self.events_q = self.ctx.Queue()
        self.process = self.ctx.Process(target=worker.run, daemon=True,
                                        args=(self.use_case, self.cameras, self.frames_q, self.events_q))
        self.process.start()

    def stop(self):
        try:
            self.frames_q.put_nowait(None)
        except queue.Full:
            pass
        self.process.join(2)
        if self.process.is_alive():
            self.process.terminate()
        for q in (self.frames_q, self.events_q):
            q.cancel_join_thread()  # don't block exit flushing frames nobody will read


def start_workers(settings, ctx):
    by_use_case = {}
    for cam in settings.cameras:
        for uc, params in cam.use_cases.items():
            if uc not in REGISTRY:
                raise ValueError(f"camera {cam.id}: unknown use case '{uc}'")
            extra = {"vlm": settings.vlm} if getattr(REGISTRY[uc], "uses_vlm", False) else {}
            if settings.model_versions.get(uc) and not (params or {}).get("model"):
                extra["model"] = settings.model_versions[uc]
            by_use_case.setdefault(uc, {})[cam.id] = dict(params or {}, **extra)
    return {uc: Worker(ctx, uc, cams) for uc, cams in by_use_case.items()}


def _rss_mb():
    """This process's memory (RSS) in MB."""
    with open("/proc/self/status") as f:
        return next(int(l.split()[1]) for l in f if l.startswith("VmRSS")) / 1024


def main(path="config/node.yaml"):
    settings, source, config_hash, config_url, load_info = load_node(path)
    print(f"settings from: {source}", flush=True)
    restart = {"needed": False}
    failure = {"code": 0}  # 1 after a pipeline error or end of stream: the service or Docker restarts the node
    cams = settings.cameras
    W, H = settings.engine["width"], settings.engine["height"]
    os.makedirs(settings.data_dir, exist_ok=True)
    outbox = Outbox(os.path.join(settings.data_dir, "outbox.db"), settings.node_id, settings.site_id)
    snaps = Snapshots(outbox, os.path.join(settings.data_dir, "snapshots"))
    watchdog = CameraWatchdog([c.id for c in cams], outbox, settings.camera_offline_after_s,
                              settings.engine.get("camera_startup_grace_s", 60.0))
    uploader = Uploader(outbox, settings.hub["url"]) if settings.hub.get("url") else None
    retention = Retention(outbox, os.path.join(settings.data_dir, "snapshots"), settings.retention)
    if settings.vlm.get("url") and any(getattr(REGISTRY[uc], "uses_vlm", False) for c in cams for uc in c.use_cases):
        for url in (settings.vlm["url"], settings.vlm.get("backup_url")):  # one watch per VLM server
            if url:
                VLMWatch(url, outbox).start()  # one vlm_down alert per server, not one per camera
    status_fns = {  # each health value is read on its own: one failing never stops the report (see health.py)
        "settings_source": lambda: source, "settings_version": lambda: config_hash[:12],  # UI can confirm a change
        "cameras_fps": lambda: status.get("fps"), "dropped_frames": lambda: stats["dropped"],
        "dropped_by_use_case": lambda: router.dropped_by_use_case(),
        "probe_errors": lambda: sum(n for k, n in errors.counts.items() if k.startswith("probe")),
        "loop_errors": lambda: {k: n for k, n in errors.counts.items() if not k.startswith("probe")},
        "worker_restarts": lambda: dict(supervisor.restarts),
        "snapshots_expired": lambda: snap_requests.expired, "pose_crops_held": lambda: len(crop_store),
        "oldest_unsent_s": outbox.oldest_unsent_s, "photos_waiting": outbox.photos_waiting,
        "photos_refused": lambda: uploader.photos_refused if uploader else 0, "snapshot_drops": lambda: snaps.dropped,
        "cameras_offline": lambda: [cams[i].id for i, s in enumerate(watchdog.offline_since) if s is not None],
        "workers_alive": lambda: status.get("workers"), "unsent_alerts": outbox.unsent, "outbox_mb": outbox.size_mb,
        "snapshot_errors": lambda: snaps.errors, "node_rss_mb": lambda: status.get("rss_mb"),
        "snapshot_queue": lambda: snaps.q.qsize(), "pipeline_overruns": lambda: dict(overruns),
        "vlm": lambda: dict(vlm_stats)}  # per use case: asked, answered, by_backup, failed, busy, avg_s (since start)
    vlm_stats = {}
    health = (HealthReporter(settings.hub["url"], settings.node_id, settings.site_id, status_fns,
        settings.hub.get("health_every_s", 30), settings.data_dir) if settings.hub.get("url") else None)
    workers = start_workers(settings, mp.get_context("spawn"))
    supervisor = RestartPolicy()  # crashed workers restart at once, then after 5, 10, 20, 40, 60 s
    routes = [list(cam.use_cases) for cam in cams]
    pixel_zones = [{uc: ((p or {}).get("zone", [0, 0, 1, 1]), REGISTRY[uc].pixel_mode,
                         (p or {}).get("sample_every_s", REGISTRY[uc].pixels_every_s), REGISTRY[uc].pixel_max_width)
                    for uc, p in cam.use_cases.items() if REGISTRY[uc].needs_pixels} for cam in cams]
    router = Router(routes, pixel_zones)  # which use cases get each frame, and when they get pixels

    Gst.init(None)
    for name, engine in missing_engines(settings):  # DeepStream would quietly rebuild it (~25 min, every start)
        print(f"engine missing for {name}: {engine} - build it on this jetson (tools/install_node.sh, step 6)", flush=True)
        outbox.add({"camera_id": "node", "use_case": "pipeline", "class": "engine_missing", "severity": "warning",
                    "ts": time.time(), "incident": f"engine:{name}", "extra": {"model": name, "engine": engine}})
    overruns = Counter()  # queue name -> batches dropped because the next stage was too slow
    pipe, probe_pad, model_elements = build(settings, on_overrun=lambda q: overruns.update([q]))
    pose_models = [m for m in settings.models if m.get("crop") == "pose"]
    pose_cams = ({i for i, c in enumerate(cams) if c.id in pose_models[0]["cameras"]}
                 if pose_models and pose_models[0].get("cameras") else None)
    HIDDEN = 99  # a person on a camera without pose is briefly marked as this class, so pose skips them
    crop_store = CropStore()  # (camera, frame, object id) -> (real box, crop pose saw); old entries pruned
    errors = ErrorCounter(log=lambda m: print(m, flush=True))  # probes and timers: an error never escapes

    def safe(fn):
        """A probe runs in DeepStream's video thread and must never raise there."""
        def wrapped(pad, info):
            return errors.run(f"probe {fn.__name__}", fn, pad, info, default=Gst.PadProbeReturn.OK)
        return wrapped

    def each_person(info, classes):
        batch = pyds.gst_buffer_get_nvds_batch_meta(hash(info.get_buffer()))
        lf = batch.frame_meta_list
        while lf is not None:
            fm = pyds.NvDsFrameMeta.cast(lf.data)
            lo = fm.obj_meta_list
            while lo is not None:
                om = pyds.NvDsObjectMeta.cast(lo.data)
                if om.unique_component_id == 1 and om.class_id in classes:
                    yield (fm.pad_index, fm.frame_num, om.object_id), om
                try:
                    lo = lo.next
                except StopIteration:
                    break
            try:
                lf = lf.next
            except StopIteration:
                break

    def enlarge(pad, info):  # before pose: the larger crop RTMPose expects, on pose cameras only
        for key, om in each_person(info, (0,)):
            if pose_cams is not None and key[0] not in pose_cams:
                om.class_id = HIDDEN
                continue
            r = om.rect_params
            real = (r.left, r.top, r.width, r.height)
            r.left, r.top, r.width, r.height = crop = pose_box(*real, W, H)
            crop_store.put(key, (real, crop))
        return Gst.PadProbeReturn.OK

    def restore(pad, info):  # after pose: everyone is a person again, with their real box
        for key, om in each_person(info, (0, HIDDEN)):
            om.class_id = 0
            saved = crop_store.get(key)
            if saved:
                r = om.rect_params
                r.left, r.top, r.width, r.height = saved[0]
        return Gst.PadProbeReturn.OK

    for m in pose_models:
        model_elements[m["name"]].get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, safe(enlarge))
        model_elements[m["name"]].get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, safe(restore))

    def pose_of(fm, om):
        if not pose_models or om.unique_component_id != 1 or om.class_id != 0:
            return None
        saved = crop_store.pop((fm.pad_index, fm.frame_num, om.object_id))
        t = tensors(om, pose_models[0]["gie_id"]) if saved else None
        return decode(t, saved[1], W, H) if t else None
    frames = [0] * len(cams)
    stats = {"dropped": 0, "events": Counter(), "last_event": None}
    status = {}  # latest per-camera fps and workers, for health reports
    mem_trace = None
    if os.environ.get("VP_TRACEMALLOC"):  # diagnostic: which lines hold growing memory
        import tracemalloc
        tracemalloc.start(3)
        mem_trace = {"base": None, "t": time.time()}
    snap_requests = SnapshotRequests(settings.engine.get("snapshot_max_age_s", 5.0))  # photos wanted, per camera

    def probe(pad, info):
        # Runs on the video thread: copy what is needed, hand it off, return at once.
        buf = info.get_buffer()
        batch = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        l_frame = batch.frame_meta_list
        while l_frame is not None:
            fm = pyds.NvDsFrameMeta.cast(l_frame.data)
            i, cam_id = fm.pad_index, cams[fm.pad_index].id
            frames[i] += 1
            watchdog.seen(i)
            want = snap_requests.take(cam_id)
            if routes[i] or want:
                objs, l_obj = [], fm.obj_meta_list
                while l_obj is not None:
                    om = pyds.NvDsObjectMeta.cast(l_obj.data)
                    r = om.rect_params
                    objs.append(Obj(om.object_id, om.class_id, om.confidence,
                                    (r.left / W, r.top / H, (r.left + r.width) / W, (r.top + r.height) / H), source=om.unique_component_id, keypoints=pose_of(fm, om)))
                    try:
                        l_obj = l_obj.next
                    except StopIteration:
                        break
                ts = fm.ntp_timestamp / 1e9 if fm.ntp_timestamp else time.time()
                frame = Frame(cam_id, fm.frame_num, ts, objs)
                plan = router.plan(i, ts)  # [(use case, pixel spec if its pixels are due now)]
                surface = None
                if want or any(spec for _, spec in plan):  # map the frame's pixels only when someone needs them now
                    surface = pyds.get_nvds_buf_surface(hash(buf), fm.batch_id)
                for uc, spec in plan:
                    item = frame
                    if spec is not None:  # this plugin's pixels are due: only its zone
                        zone, mode, _, max_w = spec
                        px = crop_color(surface, zone, max_w) if mode == "color" else crop_zone(surface, zone)
                        item = replace(frame, pixels=px)
                    try:
                        workers[uc].frames_q.put_nowait(item)
                    except queue.Full:
                        stats["dropped"] += 1
                        router.drop(uc)
                if want:  # copy this camera's pixels once, for the events waiting on a snapshot
                    rgba = np.array(surface, copy=True, order="C")
                    snaps.submit([e for e, _ in want], rgba, objs, [ev for _, ev in want])
                if surface is not None:
                    pyds.unmap_nvds_buf_surface(hash(buf), fm.batch_id)
            try:
                l_frame = l_frame.next
            except StopIteration:
                break
        return Gst.PadProbeReturn.OK

    probe_pad.add_probe(Gst.PadProbeType.BUFFER, safe(probe))

    MAX_EVENTS_PER_DRAIN = 500  # per worker: a flood can't freeze the main loop; the rest wait 0.5 s

    def drain_once():
        snap_requests.prune()  # requests of cameras that send no frames expire here
        for ev in watchdog.check():
            stats["events"]["health:" + ev["class"]] += 1
            stats["last_event"] = ev
            print(f"health: {ev['camera_id']} {ev['class']} {ev['extra']}", flush=True)
        for uc, w in workers.items():
            if not w.process.is_alive():
                if supervisor.check(uc, False):
                    print(f"worker {uc} stopped (exit code {w.process.exitcode}); restart {supervisor.restarts[uc]}; "
                          f"if it stops again, next restart after {supervisor.next_wait(uc)} s", flush=True)
                    w.start()
                continue
            supervisor.check(uc, True)
            for _ in range(MAX_EVENTS_PER_DRAIN):
                try:
                    ev = w.events_q.get_nowait()
                except queue.Empty:
                    break
                if "stats" in ev:  # a worker's VLM counts: kept for the health report, never an alert
                    vlm_stats[ev["use_case"]] = ev["values"]
                    continue
                is_error = ev["class"] == "plugin_error"
                stats["events"][ev["use_case"] + (":error" if is_error else "")] += 1
                stats["last_event"] = ev
                event_id = outbox.add(ev)
                if not is_error and not ev["class"].endswith("_error") and ev.get("severity") in ("warning", "critical"):
                    snap_requests.add(ev["camera_id"], event_id, ev)

    def drain():
        errors.run("drain", drain_once)
        return True  # a GLib timer that returns anything else, or raises, is removed for good

    last = {"t": time.time(), "f": frames.copy()}

    def report_once():
        now = time.time()
        dt = now - last["t"]
        fps = [(frames[i] - last["f"][i]) / dt for i in range(len(cams))]
        last["t"], last["f"] = now, frames.copy()
        status.update(fps={c.id: round(f, 1) for c, f in zip(cams, fps)},
                      workers={uc: w.process.is_alive() for uc, w in workers.items()})
        if mem_trace is not None and time.time() - mem_trace["t"] > 600:
            snap = tracemalloc.take_snapshot()
            if mem_trace["base"] is None:
                print("memory tracer: baseline taken (after 10 min warm-up)", flush=True)
            else:
                for st in snap.compare_to(mem_trace["base"], "lineno")[:8]:
                    print(f"memory growth: {str(st)[:160]}", flush=True)
            mem_trace.update(base=mem_trace["base"] or snap, t=time.time())
        status["rss_mb"] = rss = round(_rss_mb())
        limit = settings.retention.get("max_memory_gb")
        if limit and rss > limit * 1024 and not restart["needed"]:  # safety net for slow native memory growth
            print(f"memory guard: main process at {rss} MB, over {limit} GB: restarting cleanly", flush=True)
            restart["needed"] = True
            GLib.idle_add(loop.quit)
        rest = fps[1:] or [0]
        ev = stats["last_event"]
        last_ev = f"{ev['camera_id']} {ev['class']} track {ev.get('track_id')}" if ev else "none"
        print(f"[{settings.node_id}] fps {cams[0].id}={fps[0]:.1f} others {min(rest):.1f}-{max(rest):.1f} | "
              f"events {dict(stats['events'])} | dropped {stats['dropped']} | "
              f"snapshot drops {snaps.dropped} | unsent {outbox.unsent()} | last: {last_ev}", flush=True)
        return True

    loop = GLib.MainLoop()

    def on_message(bus, msg):
        if msg.type == Gst.MessageType.ERROR:
            err, _ = msg.parse_error()
            where = msg.src.get_path_string()
            if ":src-" in where:  # one camera failed: report it, keep the others running
                print(f"camera error at {where}: {err.message}", flush=True)
            else:
                print(f"pipeline error at {where}: {err.message}", flush=True)
                failure["code"] = 1
                loop.quit()
        elif msg.type == Gst.MessageType.EOS:
            print("pipeline ended (end of stream)", flush=True)
            failure["code"] = 1
            loop.quit()

    bus = pipe.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_message)
    GLib.timeout_add(500, drain)
    GLib.timeout_add_seconds(10, lambda: errors.run("report", report_once) or True)
    for sig in (signal.SIGINT, signal.SIGTERM):
        GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, sig, loop.quit)
    print(f"[{settings.node_id}] starting {len(cams)} cameras, plugins: {list(workers)}", flush=True)
    if config_url:
        def settings_changed():
            print("settings changed at the hub: restarting to apply them", flush=True)
            restart["needed"] = True
            GLib.idle_add(loop.quit)
        watch(config_url, config_hash, settings_changed, settings.hub.get("config_check_s", 60),
              ignore=load_info["ignore"], report=outbox.add)

        def settings_proven():  # ran GOOD_AFTER_S: these settings become last known good
            errors.run("settings", mark_good, config_hash, os.path.dirname(settings.data_dir + "/") or "data")
            return False  # once
        GLib.timeout_add_seconds(GOOD_AFTER_S, settings_proven)
    for ev in load_info["events"]:  # settings refused at the hub, or rolled back: tell the hub
        print(f"settings: {ev['class']} {ev['extra']}", flush=True)
        outbox.add(ev)
    pipe.set_state(Gst.State.PLAYING)
    loop.run()
    pipe.set_state(Gst.State.NULL)
    for w in workers.values():
        w.stop()
    if restart["needed"]:
        sys.exit(RESTART_CODE)
    if failure["code"]:
        print("exiting with code 1, so the service or Docker restarts the node", flush=True)
        sys.exit(failure["code"])


if __name__ == "__main__":
    main(*sys.argv[1:])
