"""Moves RF-DETR's ImageNet normalisation inside the ONNX model, so DeepStream can feed plain RGB 0-1
(nvinfer can only divide all channels by one number). Checks the result matches the original.
Usage (conda env): python tools/rfdetr_normalise_inside.py in.onnx out.onnx"""
import sys
import numpy as np, onnx, onnxruntime as ort
from onnx import helper, numpy_helper

MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)
g = m.graph
old = g.input[0].name
dims = [d.dim_value or 1 for d in g.input[0].type.tensor_type.shape.dim]  # variable batch: check with 1
g.input[0].name = "image"
g.initializer.extend([numpy_helper.from_array(MEAN, "norm_mean"), numpy_helper.from_array(STD, "norm_std")])
g.node.insert(0, helper.make_node("Div", ["image_centered", "norm_std"], [old]))
g.node.insert(0, helper.make_node("Sub", ["image", "norm_mean"], ["image_centered"]))
onnx.checker.check_model(m)
onnx.save(m, dst)

x01 = np.random.default_rng(0).random(dims, dtype=np.float32)
a = ort.InferenceSession(src, providers=["CPUExecutionProvider"]).run(None, {old: (x01 - MEAN) / STD})
b = ort.InferenceSession(dst, providers=["CPUExecutionProvider"]).run(None, {"image": x01})
diff = max(float(np.abs(p - q).max()) for p, q in zip(a, b))
print(f"{dst}: max difference vs original {diff:.2e}", "OK" if diff < 1e-4 else "MISMATCH")
