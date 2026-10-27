"""Compares an nvtracker config on the test clip: speed with N streams (default 16), plus ID stability and
tracker-vs-detector box agreement on stream 0. Usage: python3 tools/tracker_compare.py CONFIG.yml [STREAMS]"""
import os, statistics, sys, time
from collections import defaultdict
import gi
gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst
import pyds

DS = "/opt/nvidia/deepstream/deepstream-7.1"
CLIP = os.path.expanduser("~/vision-pipeline/tools/test_clip.mp4")
cfg, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 16
UNTRACKED, VEHICLES = 0xFFFFFFFFFFFFFFFF, (2, 3, 5, 7)
Gst.init(None)
srcs = " ".join(f"nvurisrcbin uri=file://{CLIP} ! m.sink_{i}" for i in range(n))
pipe = Gst.parse_launch(
    f"{srcs} nvstreammux name=m batch-size={n} width=1280 height=720 ! "
    f"nvinfer config-file-path={os.path.expanduser('~/vision-pipeline/config/pgie_yolo26.txt')} batch-size={n} ! "
    f"nvtracker name=trk ll-lib-file={DS}/lib/libnvds_nvmultiobjecttracker.so ll-config-file={cfg} ! fakesink sync=false")
track_frames, ious, st = defaultdict(int), defaultdict(list), {"frames": 0, "t0": None}


def iou(a, b):  # boxes as left, top, width, height
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / (a[2] * a[3] + b[2] * b[3] - inter) if inter else 0.0


def probe(pad, info):
    batch = pyds.gst_buffer_get_nvds_batch_meta(hash(info.get_buffer()))
    lf = batch.frame_meta_list
    while lf is not None:
        fm = pyds.NvDsFrameMeta.cast(lf.data)
        st["frames"] += 1
        st["t0"] = st["t0"] or time.time()
        if fm.pad_index == 0:
            lo = fm.obj_meta_list
            while lo is not None:
                o = pyds.NvDsObjectMeta.cast(lo.data)
                if o.object_id != UNTRACKED and (o.class_id == 0 or o.class_id in VEHICLES):
                    kind = "person" if o.class_id == 0 else "vehicle"
                    track_frames[(kind, o.object_id)] += 1
                    d = o.detector_bbox_info.org_bbox_coords
                    if d.width > 0 and d.height > 0:  # detected this frame: compare the tracker's box with it
                        r = o.rect_params
                        ious[kind].append(iou((r.left, r.top, r.width, r.height), (d.left, d.top, d.width, d.height)))
                try:
                    lo = lo.next
                except StopIteration:
                    break
        try:
            lf = lf.next
        except StopIteration:
            break
    return Gst.PadProbeReturn.OK


pipe.get_by_name("trk").get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, probe)
loop = GLib.MainLoop()
bus = pipe.get_bus(); bus.add_signal_watch()
bus.connect("message", lambda b, m: loop.quit() if m.type in (Gst.MessageType.EOS, Gst.MessageType.ERROR) else None)
pipe.set_state(Gst.State.PLAYING); loop.run(); pipe.set_state(Gst.State.NULL)
fps = st["frames"] / (time.time() - st["t0"])
out = [f"{os.path.basename(cfg).replace('config_tracker_', '').replace('.yml', ''):14s}", f"fps(16 streams)={fps:6.0f}"]
for kind in ("person", "vehicle"):
    lengths = [v for (k, _), v in track_frames.items() if k == kind]
    if lengths:
        short = sum(1 for v in lengths if v < 10) / len(lengths) * 100
        ov = sorted(ious[kind])
        out.append(f"{kind}: ids={len(lengths):4d} mean_len={statistics.mean(lengths):5.1f} short={short:4.0f}% "
                   f"box_iou median={statistics.median(ov):.3f} worst10%={ov[len(ov) // 10]:.3f}")
print("RESULT", " | ".join(out), flush=True)
