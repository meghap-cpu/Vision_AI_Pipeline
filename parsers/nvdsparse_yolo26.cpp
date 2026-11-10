#include <algorithm>
#include "nvdsinfer_custom_impl.h"

// YOLO26 end-to-end output: [300, 6] per image = x1, y1, x2, y2, score, class
// in network input pixels. Duplicate removal is already done inside the model.
extern "C" bool NvDsInferParseYolo26(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    const NvDsInferLayerInfo& layer = outputLayersInfo[0];
    const float* data = static_cast<const float*>(layer.buffer);
    const int numDets = layer.inferDims.d[0];
    const float w = networkInfo.width, h = networkInfo.height;
    for (int i = 0; i < numDets; ++i) {
        const float* d = data + i * 6;
        const int cls = static_cast<int>(d[5]);
        if (cls < 0 || cls >= static_cast<int>(detectionParams.numClassesConfigured)) continue;
        if (d[4] < detectionParams.perClassPreclusterThreshold[cls]) continue;
        const float x1 = std::clamp(d[0], 0.f, w), y1 = std::clamp(d[1], 0.f, h);
        const float x2 = std::clamp(d[2], 0.f, w), y2 = std::clamp(d[3], 0.f, h);
        if (x2 <= x1 || y2 <= y1) continue;
        NvDsInferParseObjectInfo obj;
        obj.classId = cls;
        obj.detectionConfidence = d[4];
        obj.left = x1; obj.top = y1; obj.width = x2 - x1; obj.height = y2 - y1;
        objectList.push_back(obj);
    }
    return true;
}
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYolo26);
