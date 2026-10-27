"""Records what the pipeline sees in a video file: each tracked person with box, confidence and pose,
frame by frame, as JSON lines. Replay it through any plugin with tools/replay.py to tune in seconds.
Usage: python3 tools/record_poses.py VIDEO OUT.jsonl [--fps 5] [--config config/node.yaml]"""
import argparse, json, os, sys

import gi
gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst
import cv2, pyds

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vp.pose import decode, pose_box, tensors
from vp.settings import load

ap = argparse.ArgumentParser()
ap.add_argument("video"); ap.add_argument("out")
ap.add_argument("--fps", type=float, default=5.0, help="analysis rate to record at")
ap.add_argument("--config", default="config/node.yaml")
a = ap.parse_args()
s = load(a.config)
e, W, H = s.engine, s.engine["width"], s.engine["height"]
pose_m = next(m for m in s.models if m.get("crop") == "pose")
src_fps = cv2.VideoCapture(a.video).get(cv2.CAP_PROP_FPS) or 25.0
drop = max(1, round(src_fps / a.fps))

Gst.init(None)
pipe = Gst.parse_launch(
    f"nvurisrcbin uri=file://{os.path.abspath(a.video)} drop-frame-interval={drop} ! m.sink_0 "
    f"nvstreammux name=m batch-size=1 width={W} height={H} ! nvinfer config-file-path={e['pgie_config']} ! "
    f"nvtracker ll-lib-file={e['tracker_lib']} ll-config-file={e['tracker_config']} ! "
    f"nvinfer name=pose config-file-path={pose_m['config']} ! fakesink sync=false")
real, out, count = {}, open(a.out, "w"), {"frames": 0, "people": 0}


def frames_with_people(info):
    batch = pyds.gst_buffer_get_nvds_batch_meta(hash(info.get_buffer()))
    lf = batch.frame_meta_list
    while lf is not None:
        fm = pyds.NvDsFrameMeta.cast(lf.data)
        people, lo = [], fm.obj_meta_list
        while lo is not None:
            om = pyds.NvDsObjectMeta.cast(lo.data)
            if om.unique_component_id == 1 and om.class_id == 0:
                people.append(om)
            try:
                lo = lo.next
            except StopIteration:
                break
        yield fm, people
        try:
            lf = lf.next
        except StopIteration:
            break


def enlarge(pad, info):  # before pose: the crop RTMPose expects
    for fm, people in frames_with_people(info):
        for om in people:
            r = om.rect_params
            real[(fm.frame_num, om.object_id)] = (r.left, r.top, r.width, r.height)
            r.left, r.top, r.width, r.height = pose_box(r.left, r.top, r.width, r.height, W, H)
    return Gst.PadProbeReturn.OK


def record(pad, info):  # after pose: decode with the crop used, restore the real box, write one line per frame
    for fm, people in frames_with_people(info):
        row = []
        for om in people:
            r = om.rect_params
            crop = (r.left, r.top, r.width, r.height)
            l, t, w, h = real.pop((fm.frame_num, om.object_id), crop)
            r.left, r.top, r.width, r.height = l, t, w, h
            tens = tensors(om, pose_m["gie_id"])
            row.append({"track_id": om.object_id, "conf": round(om.confidence, 3),
                        "box": [round(l / W, 4), round(t / H, 4), round((l + w) / W, 4), round((t + h) / H, 4)],
                        "keypoints": decode(tens, crop, W, H) if tens else None})
        out.write(json.dumps({"frame": fm.frame_num, "ts": round(fm.buf_pts / 1e9, 3), "objects": row}) + "\n")
        count["frames"] += 1
        count["people"] += len(row)
    return Gst.PadProbeReturn.OK


pose = pipe.get_by_name("pose")
pose.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, enlarge)
pose.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, record)
loop = GLib.MainLoop()
bus = pipe.get_bus(); bus.add_signal_watch()
bus.connect("message", lambda b, m: loop.quit() if m.type in (Gst.MessageType.EOS, Gst.MessageType.ERROR) else None)
pipe.set_state(Gst.State.PLAYING); loop.run(); pipe.set_state(Gst.State.NULL)
out.close()
print(f"recorded {count['frames']} frames, {count['people']} person-frames at ~{a.fps:g} fps -> {a.out}")
