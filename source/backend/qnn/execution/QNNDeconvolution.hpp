//
//  QNNDeconvolution.hpp
//  MNN
//
//  Created by MNN on 2025/07/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_QNNDECONVOLUTION_HPP
#define MNN_QNNDECONVOLUTION_HPP

#include "QNNCommonExecution.hpp"
#include "QnnTypes.h"

namespace MNN {
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE

class QNNDeconvolution : public QNNCommonExecution {
public:
    QNNDeconvolution(Backend *backend, const Op *op) : QNNCommonExecution(backend, op) {}
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    template <typename T>
    void convertWeight(const T * src, T * dst, int oc, int ic, int kernelH, int kernelW) {
        // MNN Deconv weight: [IC, OC, KH, KW] (from ONNX IOHW format)
        // QNN TransposeConv2d weight: [KH, KW, IC, OC] (HWIO format)
        for (int i = 0; i < ic; i++) {
            for (int o = 0; o < oc; o++) {
                for (int h = 0; h < kernelH; h++) {
                    for (int w = 0; w < kernelW; w++) {
                        uint32_t srcOffset = w + kernelW * (h + kernelH * (o + oc * i));
                        uint32_t dstOffset = o + oc * (i + ic * (w + kernelW * h));
                        dst[dstOffset] = src[srcOffset];
                    }
                }
            }
        }
    }
    void createWeightAndBias(Qnn_DataType_t dataType, int oc, int ic, int kernelH, int kernelW, int group);
    void createBias(Qnn_DataType_t dataType, int oc, int group);
};

#endif
} // end namespace QNN
} // end namespace MNN

#endif
