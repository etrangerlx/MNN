//
//  QNNDeconvolution.hpp
//  MNN
//
//  Created by MNN on 2025/07/10.
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
    void convertWeightDeconv(const T * src, T * dst, int oc, int ic, int kernelH, int kernelW) {
        // MNN deconv weight: [IC, OC, KH, KW] (NCHW format, but logically [IC, OC, KH, KW])
        // -> QNN TransposeConv2d filter: [KH, KW, IC/group, OC] (HWIO format)
        for (int o = 0; o < oc; o++) {
            for (int i = 0; i < ic; i++) {
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
    bool createWeightAndBias(Qnn_DataType_t dataType, const Tensor *input, int oc, int ic, int kernelH, int kernelW,
                             int group);
    ErrorCode onEncodeQuantDequantConv(Tensor *input, Tensor *output, const int n, const int ic, const int oc,
                                       const int group, const int outPadH, const int outPadW,
                                       const Convolution2DCommon *common);
};

#endif
} // namespace QNN
} // namespace MNN

#endif
