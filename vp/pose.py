"""RTMPose helpers: the larger crop RTMPose expects, and decoding its output into 17 joints."""
import ctypes

import numpy as np
import pyds


def pose_box(left, top, width, height, W, H):
    """RTMPose's crop: centre, 1.25x larger, made 3:4 (w:h), kept inside the frame. Returns left, top, w, h."""
    cx, cy, w, h = left + width / 2, top + height / 2, width * 1.25, height * 1.25
    w, h = (w, w / 0.75) if w > h * 0.75 else (h * 0.75, h)
    x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
    x2, y2 = min(W - 1.0, cx + w / 2), min(H - 1.0, cy + h / 2)
    return x1, y1, x2 - x1, y2 - y1


def tensors(obj, gie_id):
    """The raw output layers a secondary model (gie_id) attached to this object, or None."""
    l = obj.obj_user_meta_list
    while l is not None:
        um = pyds.NvDsUserMeta.cast(l.data)
        if um.base_meta.meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
            tm = pyds.NvDsInferTensorMeta.cast(um.user_meta_data)
            if tm.unique_id == gie_id:
                out = {}
                for i in range(tm.num_output_layers):
                    layer = pyds.get_nvds_LayerInfo(tm, i)
                    ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
                    dims = [layer.inferDims.d[k] for k in range(layer.inferDims.numDims)]
                    out[layer.layerName] = np.ctypeslib.as_array(ptr, shape=dims).copy()
                return out
        try:
            l = l.next
        except StopIteration:
            break
    return None


def decode(layers, crop, W, H):
    """simcc strips -> 17 joints (x, y, score); x and y as fractions of the frame. crop = left, top, w, h."""
    sx, sy = layers["simcc_x"], layers["simcc_y"]
    kx, ky = sx.argmax(1) / 2.0, sy.argmax(1) / 2.0  # joint positions inside the 192x256 crop
    score = np.minimum(sx.max(1), sy.max(1))
    x1, y1, w, h = crop
    return tuple((round(float((x1 + kx[j] / 192 * w) / W), 4), round(float((y1 + ky[j] / 256 * h) / H), 4),
                  round(float(score[j]), 3)) for j in range(17))
