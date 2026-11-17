"""Builds the DeepStream trunk from a node's settings (the parts that need no DeepStream are in
pipeline_config.py)."""
from vp.pipeline_config import _preprocess_config

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst


def _make(kind, name, **props):
    el = Gst.ElementFactory.make(kind, name)
    if el is None:
        raise RuntimeError(f"could not create {kind}")
    for key, value in props.items():
        el.set_property(key.replace("_", "-"), value)
    return el


def _link_video(src, pad, mux_pad):
    caps = pad.get_current_caps() or pad.query_caps(None)
    if caps.get_structure(0).get_name().startswith("video") and not mux_pad.is_linked():
        pad.link(mux_pad)


def build(settings, on_overrun=None):
    """One nvurisrcbin per camera -> nvstreammux -> person detector -> tracker
    -> RGBA conversion (so snapshots can read pixels) -> sink.
    Queues between stages hold at most 4 batches and drop the oldest when full, so video stays current under
    overload (a full queue would otherwise hold up every stage before it); on_overrun(queue name) counts each drop.
    Returns the pipeline and the pad where the probe attaches."""
    e, cams = settings.engine, settings.cameras
    pipe = Gst.Pipeline.new("vision-pipeline")
    mux = _make("nvstreammux", "mux", batch_size=len(cams), width=e["width"], height=e["height"],
                live_source=True, batched_push_timeout=40000)
    pgie = _make("nvinfer", "pgie", config_file_path=e["pgie_config"], batch_size=len(cams))
    tracker = _make("nvtracker", "tracker", ll_lib_file=e["tracker_lib"], ll_config_file=e["tracker_config"])
    extra, model_elements = [], {}
    index = {c.id: i for i, c in enumerate(cams)}
    for m in settings.models:
        infer = _make("nvinfer", f"model-{m['name']}", config_file_path=m["config"], batch_size=len(cams),
                      interval=int(m.get("interval", 0)))
        if m.get("cameras") and m.get("crop") != "pose":
            # whole-frame model on some cameras only: nvdspreprocess prepares input for those cameras alone
            src_ids = [index[c] for c in m["cameras"]]
            extra.append(_make("nvdspreprocess", f"pre-{m['name']}", config_file=_preprocess_config(m, src_ids, settings)))
            infer.set_property("input-tensor-meta", True)
        extra.append(infer)
        model_elements[m["name"]] = infer
    conv = _make("nvvideoconvert", "conv")
    rgba = _make("capsfilter", "rgba", caps=Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))
    sink = _make("fakesink", "sink", sync=False)
    # a queue after each heavy stage lets the next batch start while a model is still busy
    chain = []
    for n, el in enumerate([mux, pgie, tracker, *extra]):
        q = _make("queue", f"queue-{n}", max_size_buffers=4, max_size_bytes=0, max_size_time=0, leaky=2)
        if on_overrun:
            q.connect("overrun", lambda queue: on_overrun(queue.get_name()))
        chain += [el, q]
    chain = tuple(chain + [conv, rgba, sink])
    for el in chain:
        pipe.add(el)
    for i, cam in enumerate(cams):
        src = _make("nvurisrcbin", f"src-{cam.id}", uri=cam.uri,
                    drop_frame_interval=cam.drop_interval, rtsp_reconnect_interval=10)
        pipe.add(src)
        src.connect("pad-added", _link_video, mux.request_pad_simple(f"sink_{i}"))
    for a, b in zip(chain, chain[1:]):
        if not a.link(b):
            raise RuntimeError(f"could not link {a.get_name()} -> {b.get_name()}")
    return pipe, rgba.get_static_pad("src"), model_elements
