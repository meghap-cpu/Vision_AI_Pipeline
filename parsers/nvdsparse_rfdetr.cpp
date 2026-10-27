#include <algorithm>
#include <cmath>
#include <cstring>
#include "nvdsinfer_custom_impl.h"

// RF-DETR output (per image):
//   "dets"   [300, 4]           cx, cy, w, h as fractions of the model input
//   "labels" [300, num_classes] class logits; sigmoid gives the score
// No duplicate removal is needed: each of the 300 queries is one candidate object.
static const NvDsInferLayerInfo* find(std::vector<NvDsInferLayerInfo> const& layers, const char* name) {
    for (auto const& l : layers)
        if (l.layerName && std::strcmp(l.layerName, name) == 0) return &l;
    return nullptr;
}

extern "C" bool NvDsInferParseRfdetr(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    const NvDsInferLayerInfo* dets = find(outputLayersInfo, "dets");
    const NvDsInferLayerInfo* labels = find(outputLayersInfo, "labels");
    if (!dets || !labels) return false;
    const float* box = static_cast<const float*>(dets->buffer);
    const float* logit = static_cast<const float*>(labels->buffer);
    const int queries = dets->inferDims.d[0];
    const int classes = labels->inferDims.d[1];
    const int usable = std::min(classes, static_cast<int>(detectionParams.numClassesConfigured));
    const float W = networkInfo.width, H = networkInfo.height;
    for (int q = 0; q < queries; ++q) {
        const float* l = logit + q * classes;
        int best = 0;
        for (int c = 1; c < usable; ++c)
            if (l[c] > l[best]) best = c;
        const float score = 1.f / (1.f + std::exp(-l[best]));
        if (score < detectionParams.perClassPreclusterThreshold[best]) continue;
        const float* b = box + q * 4;
        const float x1 = std::clamp((b[0] - b[2] / 2) * W, 0.f, W), y1 = std::clamp((b[1] - b[3] / 2) * H, 0.f, H);
        const float x2 = std::clamp((b[0] + b[2] / 2) * W, 0.f, W), y2 = std::clamp((b[1] + b[3] / 2) * H, 0.f, H);
        if (x2 <= x1 || y2 <= y1) continue;
        NvDsInferParseObjectInfo obj;
        obj.classId = best;
        obj.detectionConfidence = score;
        obj.left = x1; obj.top = y1; obj.width = x2 - x1; obj.height = y2 - y1;
        objectList.push_back(obj);
    }
    return true;
}
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseRfdetr);
