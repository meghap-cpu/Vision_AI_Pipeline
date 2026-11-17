"""Pipeline settings that need no DeepStream. Run: python3 -m tests.test_pipeline_config"""
import os, tempfile
from types import SimpleNamespace

from vp.pipeline_config import _preprocess_config, engine_file, missing_engines


def run():
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "config")); os.makedirs(os.path.join(d, "models"))
    open(os.path.join(d, "models", "yolo26s_b16_fp16.engine"), "w").write("x")
    def cfg(name, engine):
        path = os.path.join(d, "config", name)
        open(path, "w").write(f"[property]\ngpu-id=0\nmodel-engine-file={engine}\nbatch-size=1\n")
        return path
    pgie = cfg("pgie.txt", "../models/yolo26s_b16_fp16.engine")       # present, relative to the config
    spill = cfg("spill.txt", "../models/spill_yolo26n_b16_fp16.engine")  # missing
    assert engine_file(pgie) == os.path.join(d, "models", "yolo26s_b16_fp16.engine")
    s = SimpleNamespace(engine={"pgie_config": pgie}, models=[{"name": "spill", "config": spill},
                                                              {"name": "ghost", "config": "/nowhere.txt"}], data_dir=d)
    missing = missing_engines(s)
    assert missing == [("spill", os.path.join(d, "models", "spill_yolo26n_b16_fp16.engine"))], missing
    print("ok  engine paths read relative to their settings file; the missing spill engine is found before start-up")

    m = {"name": "ppe", "gie_id": 3, "input": [3, 560, 560], "tensor": "image"}
    text = open(_preprocess_config(m, [7, 8], s)).read()
    assert "network-input-shape=2;3;560;560" in text and "src-ids=7;8" in text and "target-unique-ids=3" in text
    print("ok  PPE pre-processing settings: batch of 2, cameras 7 and 8, model 3")
    print("PASS: engine check and pre-processing settings, without DeepStream")


if __name__ == "__main__":
    run()
