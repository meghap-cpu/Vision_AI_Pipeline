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
from vp.remote_config import RESTART_CODE, load_node, watch
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
    settings, source, config_hash, config_url = load_node(path)
    print(f"settings from: {source}", flush=True)
    restart = {"needed": False}
    cams = settings.cameras
    W, H = settings.engine["width"], settings.engine["height"]
    os.makedirs(settings.data_dir, exist_ok=True)
    outbox = Outbox(os.path.join(settings.data_dir, "outbox.db"), settings.node_id, settings.site_id)
    snaps = Snapshots(outbox, os.path.join(settings.data_dir, "snapshots"))
    watchdog = CameraWatchdog([c.id for c in cams], outbox, settings.camera_offline_after_s)
    uploader = Uploader(outbox, settings.hub["url"]) if settings.hub.get("url") else None
    retention = Retention(outbox, os.path.join(settings.data_dir, "snapshots"), settings.retention)
    if settings.vlm.get("url") and any(getattr(REGISTRY[uc], "uses_vlm", False) for c in cams for uc in c.use_cases):
        VLMWatch(settings.vlm["url"], outbox).start()  # one vlm_down alert for the node, not one per camera
    health = (HealthReporter(settings.hub["url"], settings.node_id, settings.site_id, lambda: {
        "settings_source": source, "cameras_fps": status.get("fps"), "dropped_frames": stats["dropped"],
        "cameras_offline": [cams[i].id for i, s in enumerate(watchdog.offline_since) if s is not None],
        "workers_alive": status.get("workers"), "unsent_alerts": outbox.unsent(),
        "snapshot_errors": snaps.errors, "node_rss_mb": status.get("rss_mb"), "snapshot_queue": snaps.q.qsize()},
        settings.hub.get("health_every_s", 30), settings.data_dir) if settings.hub.get("url") else None)
    workers = start_workers(settings, mp.get_context("spawn"))
    routes = [list(cam.use_cases) for cam in cams]
    pixel_zones = [{uc: ((p or {}).get("zone", [0, 0, 1, 1]), REGISTRY[uc].pixel_mode,
                         (p or {}).get("sample_every_s", REGISTRY[uc].pixels_every_s), REGISTRY[uc].pixel_max_width)
                    for uc, p in cam.use_cases.items() if REGISTRY[uc].needs_pixels} for cam in cams]
    last_pixels = {}  # (camera index, use case) -> when pixels were last sent

    Gst.init(None)
    pipe, probe_pad, model_elements = build(settings)
    pose_models = [m for m in settings.models if m.get("crop") == "pose"]
    pose_cams = ({i for i, c in enumerate(cams) if c.id in pose_models[0]["cameras"]}
                 if pose_models and pose_models[0].get("cameras") else None)
    HIDDEN = 99  # a person on a camera without pose is briefly marked as this class, so pose skips them
    crops, crops_lock = {}, threading.Lock()  # (camera, frame, object id) -> (real box, crop pose saw)

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
            with crops_lock:
                crops[key] = (real, crop)
        return Gst.PadProbeReturn.OK

    def restore(pad, info):  # after pose: everyone is a person again, with their real box
        for key, om in each_person(info, (0, HIDDEN)):
            om.class_id = 0
            with crops_lock:
                saved = crops.get(key)
            if saved:
                r = om.rect_params
                r.left, r.top, r.width, r.height = saved[0]
        return Gst.PadProbeReturn.OK

    for m in pose_models:
        model_elements[m["name"]].get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, enlarge)
        model_elements[m["name"]].get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, restore)

    def pose_of(fm, om):
        if not pose_models or om.unique_component_id != 1 or om.class_id != 0:
            return None
        with crops_lock:
            saved = crops.pop((fm.pad_index, fm.frame_num, om.object_id), None)
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
    want_snap, snap_lock = {}, threading.Lock()  # camera_id -> [(event_id, track_id)]

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
            with snap_lock:
                want = want_snap.pop(cam_id, None)
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
                surface = None
                if want or pixel_zones[i]:
                    surface = pyds.get_nvds_buf_surface(hash(buf), fm.batch_id)
                for uc in routes[i]:
                    item = frame
                    if uc in pixel_zones[i]:  # this plugin needs pixels: only its zone, at its rate
                        zone, mode, every, max_w = pixel_zones[i][uc]
                        if ts - last_pixels.get((i, uc), -1e9) >= every:
                            last_pixels[(i, uc)] = ts
                            px = crop_color(surface, zone, max_w) if mode == "color" else crop_zone(surface, zone)
                            item = replace(frame, pixels=px)
                    try:
                        workers[uc].frames_q.put_nowait(item)
                    except queue.Full:
                        stats["dropped"] += 1
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

    probe_pad.add_probe(Gst.PadProbeType.BUFFER, probe)

    def drain():
        for ev in watchdog.check():
            stats["events"]["health:" + ev["class"]] += 1
            stats["last_event"] = ev
            print(f"health: {ev['camera_id']} {ev['class']} {ev['extra']}", flush=True)
        for uc, w in workers.items():
            if not w.process.is_alive():
                print(f"worker {uc} stopped (exit code {w.process.exitcode}); restarting", flush=True)
                w.start()
                continue
            while True:
                try:
                    ev = w.events_q.get_nowait()
                except queue.Empty:
                    break
                is_error = ev["class"] == "plugin_error"
                stats["events"][ev["use_case"] + (":error" if is_error else "")] += 1
                stats["last_event"] = ev
                event_id = outbox.add(ev)
                if not is_error and not ev["class"].endswith("_error") and ev.get("severity") in ("warning", "critical"):
                    with snap_lock:
                        want_snap.setdefault(ev["camera_id"], []).append((event_id, ev))
        return True

    last = {"t": time.time(), "f": frames.copy()}

    def report():
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
                loop.quit()
        elif msg.type == Gst.MessageType.EOS:
            loop.quit()

    bus = pipe.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_message)
    GLib.timeout_add(500, drain)
    GLib.timeout_add_seconds(10, report)
    for sig in (signal.SIGINT, signal.SIGTERM):
        GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, sig, loop.quit)
    print(f"[{settings.node_id}] starting {len(cams)} cameras, plugins: {list(workers)}", flush=True)
    if config_url:
        def settings_changed():
            print("settings changed at the hub: restarting to apply them", flush=True)
            restart["needed"] = True
            GLib.idle_add(loop.quit)
        watch(config_url, config_hash, settings_changed, settings.hub.get("config_check_s", 60))
    pipe.set_state(Gst.State.PLAYING)
    loop.run()
    pipe.set_state(Gst.State.NULL)
    for w in workers.values():
        w.stop()
    if restart["needed"]:
        sys.exit(RESTART_CODE)


if __name__ == "__main__":
    main(*sys.argv[1:])
