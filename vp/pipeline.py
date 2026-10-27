"""Builds the DeepStream trunk from a node's settings."""
import os

PREPROCESS_LIB = "/opt/nvidia/deepstream/deepstream/lib/gst-plugins/libcustom2d_preprocess.so"


def _preprocess_config(m, src_ids, settings):
    """nvdspreprocess settings that prepare model m's input only for the given camera indexes."""
    c, h, w = m["input"]
    path = os.path.join(os.path.abspath(settings.data_dir), f"preprocess-{m['name']}.txt")
    with open(path, "w") as f:
        f.write(f"""[property]
enable=1
target-unique-ids={m['gie_id']}
network-input-order=0
process-on-frame=1
unique-id={100 + m['gie_id']}
gpu-id=0
maintain-aspect-ratio=0
symmetric-padding=0
processing-width={w}
processing-height={h}
scaling-buf-pool-size=6
tensor-buf-pool-size=6
network-input-shape={len(src_ids)};{c};{h};{w}
network-color-format=0
tensor-data-type=0
tensor-name={m['tensor']}
scaling-pool-memory-type=0
scaling-pool-compute-hw=0
scaling-filter=1
custom-lib-path={PREPROCESS_LIB}
custom-tensor-preparation-function=CustomTensorPreparation

[user-configs]
pixel-normalization-factor=0.003921568
offsets=0;0;0

[group-0]
src-ids={';'.join(str(i) for i in src_ids)}
custom-input-transformation-function=CustomAsyncTransformation
process-on-roi=0
""")
    return path
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


def build(settings):
    """One nvurisrcbin per camera -> nvstreammux -> person detector -> tracker
    -> RGBA conversion (so snapshots can read pixels) -> sink.
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
        chain += [el, _make("queue", f"queue-{n}")]
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
