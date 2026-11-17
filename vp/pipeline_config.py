"""The video pipeline's settings that need no DeepStream to work out, so they can be tested anywhere:
nvdspreprocess settings for a model that runs on some cameras only, and a check that every engine file exists."""
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


def engine_file(config_path):
    """The engine file an nvinfer settings file names (model-engine-file), as a path; None if it names none.
    Relative paths count from the settings file's folder, as DeepStream reads them."""
    with open(config_path) as f:
        for line in f:
            key, _, value = line.strip().partition("=")
            if key.strip() == "model-engine-file" and value.strip():
                v = value.strip()
                return os.path.normpath(v if os.path.isabs(v) else os.path.join(os.path.dirname(config_path), v))
    return None


def missing_engines(settings):
    """[(model name, engine path)] for every model whose engine file is missing. DeepStream would quietly rebuild
    it from ONNX (~25 min) under another name, and so again at every start."""
    out = [("person detector", settings.engine.get("pgie_config"))] + \
          [(m.get("name"), m.get("config")) for m in settings.models]
    missing = []
    for name, cfg in out:
        if not cfg or not os.path.exists(cfg):
            continue  # the settings check reports missing settings files
        eng = engine_file(cfg)
        if eng and not os.path.exists(eng):
            missing.append((name, eng))
    return missing
