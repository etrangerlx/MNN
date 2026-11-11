#include <cfloat>
#include <cmath>
#include <cstring>

#include "CPUUtils.h"

namespace infer {
namespace {

class CPUPooling : public Execution {
public:
    CPUPooling(CPUBackend* backend, const OpDesc& op) : Execution(backend), mParam(op.attr.pool) {
        if (mParam.strideX <= 0) {
            mParam.strideX = 1;
        }
        if (mParam.strideY <= 0) {
            mParam.strideY = 1;
        }
        if (mParam.kernelX <= 0) {
            mParam.kernelX = 1;
        }
        if (mParam.kernelY <= 0) {
            mParam.kernelY = 1;
        }
    }

    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.empty() || outputs.empty() || inputs[0]->dimensions() != 4) {
            return NOT_SUPPORT;
        }
        const Tensor* input = inputs[0];
        mBatch = input->batch();
        mChannel = input->channel();
        mInH   = input->height();
        mInW   = input->width();

        int kernelX = mParam.kernelX;
        int kernelY = mParam.kernelY;
        int strideX = mParam.strideX;
        int strideY = mParam.strideY;
        if (mParam.isGlobal) {
            kernelX = mInW;
            kernelY = mInH;
            strideX = 1;
            strideY = 1;
        }

        int padLeft = 0, padRight = 0, padTop = 0, padBottom = 0;
        const bool ceilMode = mParam.ceilModel;
        switch (mParam.padType) {
            case PoolPadType_VALID:
                break;
            case PoolPadType_SAME: {
                const int outH = upDiv(mInH, strideY);
                const int outW = upDiv(mInW, strideX);
                const int totalY = std::max(0, (outH - 1) * strideY + kernelY - mInH);
                const int totalX = std::max(0, (outW - 1) * strideX + kernelX - mInW);
                padTop = totalY / 2;
                padLeft = totalX / 2;
                padBottom = totalY - padTop;
                padRight = totalX - padLeft;
                break;
            }
            case PoolPadType_CAFFE:
            default:
                padLeft = padRight = mParam.padX;
                padTop = padBottom = mParam.padY;
                break;
        }
        if (mParam.pads.size() >= 2) {
            if (mParam.pads.size() == 2) {
                padTop = padBottom = mParam.pads[0];
                padLeft = padRight = mParam.pads[1];
            } else if (mParam.pads.size() >= 4) {
                padTop    = mParam.pads[0];
                padLeft   = mParam.pads[1];
                padBottom = mParam.pads[2];
                padRight  = mParam.pads[3];
            }
        }

        mKernelX = kernelX;
        mKernelY = kernelY;
        mStrideX = strideX;
        mStrideY = strideY;
        mPadLeft = padLeft;
        mPadTop  = padTop;
        mPadRight = padRight;
        mPadBottom = padBottom;

        if (mParam.isGlobal) {
            mOutH = 1;
            mOutW = 1;
        } else if (mParam.padType == PoolPadType_SAME) {
            // SAME stride semantics: the output covers ceil(input / stride).
            mOutH = upDiv(mInH, strideY);
            mOutW = upDiv(mInW, strideX);
        } else {
            // MNN computes VALID extents as ceil((in - kernel + 1) / stride),
            // i.e. the plain floor formula without the ceil-mode rounding.
            const bool ceilMode =
                (mParam.padType == PoolPadType_VALID) ? false : mParam.ceilModel;
            mOutH = computeOutExtent(mInH, kernelY, strideY, 1, padTop, ceilMode);
            mOutW = computeOutExtent(mInW, kernelX, strideX, 1, padLeft, ceilMode);
        }
        if (mOutH <= 0 || mOutW <= 0) {
            return INVALID_INPUT;
        }
        outputs[0]->setShape({mBatch, mChannel, mOutH, mOutW});
        outputs[0]->setType(DT_FLOAT);
        return NO_ERROR;
    }

    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        const float* input = inputs[0]->host<float>();
        float* output      = outputs[0]->host<float>();
        if (!input || !output) {
            return INVALID_INPUT;
        }
        const bool isMax = mParam.type == PoolType_MAXPOOL;
        const int outPlane = mOutH * mOutW;
        const int inPlane  = mInH * mInW;

        int countType = mParam.countType;
        if (countType == AvgPoolCountType_DEFAULT) {
            countType = (mParam.padType == PoolPadType_CAFFE) ? AvgPoolCountType_INCLUDE_PADDING
                                                              : AvgPoolCountType_EXCLUDE_PADDING;
        }

        auto task = [&](int begin, int end) {
            for (int index = begin; index < end; ++index) {
                const int n = index / (mChannel * mOutH);
                const int r = index % (mChannel * mOutH);
                const int c = r / mOutH;
                const int oh = r % mOutH;
                const float* src = input + (static_cast<int64_t>(n) * mChannel + c) * inPlane;
                float* dst       = output + (static_cast<int64_t>(n) * mChannel + c) * outPlane + oh * mOutW;
                const int ih0 = oh * mStrideY - mPadTop;
                for (int ow = 0; ow < mOutW; ++ow) {
                    const int iw0 = ow * mStrideX - mPadLeft;
                    if (isMax) {
                        float best = -FLT_MAX;
                        for (int ky = 0; ky < mKernelY; ++ky) {
                            int ih = ih0 + ky;
                            ih     = clampInt(ih, 0, mInH - 1);
                            const float* row = src + ih * mInW;
                            for (int kx = 0; kx < mKernelX; ++kx) {
                                int iw = iw0 + kx;
                                iw     = clampInt(iw, 0, mInW - 1);
                                best   = std::max(best, row[iw]);
                            }
                        }
                        dst[ow] = best;
                    } else {
                        const int khs = std::max(0, -ih0);
                        const int khe = std::min(mKernelY, mInH - ih0);
                        const int kws = std::max(0, -iw0);
                        const int kwe = std::min(mKernelX, mInW - iw0);
                        float sum = 0.0f;
                        for (int ky = khs; ky < khe; ++ky) {
                            const float* row = src + (ih0 + ky) * mInW;
                            for (int kx = kws; kx < kwe; ++kx) {
                                sum += row[iw0 + kx];
                            }
                        }
                        int count = 1;
                        if (countType == AvgPoolCountType_INCLUDE_PADDING) {
                            const int endH = std::min(ih0 + mKernelY, mInH + mPadBottom);
                            const int endW = std::min(iw0 + mKernelX, mInW + mPadRight);
                            count = std::max(0, endH - ih0) * std::max(0, endW - iw0);
                        } else {
                            count = std::max(0, khe - khs) * std::max(0, kwe - kws);
                        }
                        dst[ow] = count > 0 ? sum / static_cast<float>(count) : 0.0f;
                    }
                }
            }
        };
        const int total = mBatch * mChannel * mOutH;
        cpuPool(backend())->parallelFor(total, task, 1);
        return NO_ERROR;
    }

private:
    PoolCommon mParam;
    int mBatch = 0, mChannel = 0, mInH = 0, mInW = 0;
    int mOutH = 0, mOutW = 0;
    int mKernelX = 1, mKernelY = 1, mStrideX = 1, mStrideY = 1;
    int mPadLeft = 0, mPadRight = 0, mPadTop = 0, mPadBottom = 0;
};

} // namespace

std::shared_ptr<Execution> createCPUPoolingOp(CPUBackend* backend, const OpDesc& op,
                                              const std::vector<Tensor*>&,
                                              const std::vector<Tensor*>&) {
    return std::shared_ptr<Execution>(new CPUPooling(backend, op));
}

} // namespace infer
