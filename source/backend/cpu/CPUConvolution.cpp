#include <cmath>
#include <cstring>

#include "CPUUtils.h"

namespace infer {
namespace {

class CPUConvolution : public Execution {
public:
    CPUConvolution(CPUBackend* backend, const OpDesc& op)
        : Execution(backend),
          mCommon(op.attr.conv),
          mWeight(op.attr.weight),
          mBias(op.attr.bias),
          mDepthwise(op.type == OpType_ConvolutionDepthwise) {
        if (mCommon.kernelX <= 0) {
            mCommon.kernelX = 1;
        }
        if (mCommon.kernelY <= 0) {
            mCommon.kernelY = 1;
        }
        if (mCommon.strideX <= 0) {
            mCommon.strideX = 1;
        }
        if (mCommon.strideY <= 0) {
            mCommon.strideY = 1;
        }
        if (mCommon.dilateX <= 0) {
            mCommon.dilateX = 1;
        }
        if (mCommon.dilateY <= 0) {
            mCommon.dilateY = 1;
        }
        if (mCommon.group <= 0) {
            mCommon.group = 1;
        }
    }

    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.empty() || outputs.empty()) {
            return INVALID_INPUT;
        }
        const Tensor* input = inputs[0];
        if (input->dimensions() != 4) {
            return NOT_SUPPORT;
        }
        mBatch = input->batch();
        mInC   = input->channel();
        mInH   = input->height();
        mInW   = input->width();

        const int group = mDepthwise ? mInC : mCommon.group;
        mGroup          = group;
        int outC        = mCommon.outputCount;
        if (outC <= 0) {
            outC = static_cast<int>(mBias.size());
        }
        if (outC <= 0 && !mWeight.empty()) {
            outC = static_cast<int>(mWeight.size()) / std::max(1, mCommon.kernelX * mCommon.kernelY);
        }
        if (outC <= 0) {
            return INVALID_INPUT;
        }
        mOutC = outC;
        if (mInC % mGroup != 0 || mOutC % mGroup != 0) {
            return INVALID_INPUT;
        }
        mInCPerGroup  = mInC / mGroup;
        mOutCPerGroup = mOutC / mGroup;

        const int expectedWeight =
            mOutC * mInCPerGroup * mCommon.kernelY * mCommon.kernelX;
        if (static_cast<int>(mWeight.size()) != expectedWeight) {
            return INVALID_INPUT;
        }

        computeConvPads(mInW, mCommon.kernelX, mCommon.strideX, mCommon.dilateX, mCommon.padMode, mCommon.padX,
                        mPadLeft, mPadRight);
        computeConvPads(mInH, mCommon.kernelY, mCommon.strideY, mCommon.dilateY, mCommon.padMode, mCommon.padY,
                        mPadTop, mPadBottom);
        const int effectiveH = mCommon.dilateY * (mCommon.kernelY - 1) + 1;
        const int effectiveW = mCommon.dilateX * (mCommon.kernelX - 1) + 1;
        mOutH = (mInH + mPadTop + mPadBottom - effectiveH) / mCommon.strideY + 1;
        mOutW = (mInW + mPadLeft + mPadRight - effectiveW) / mCommon.strideX + 1;
        if (mOutH <= 0 || mOutW <= 0) {
            return INVALID_INPUT;
        }

        // The MNN "pads" field, when present, overrides CAFFE padding.
        if (mCommon.pads.size() >= 2) {
            if (mCommon.pads.size() == 2) {
                mPadTop = mPadBottom = mCommon.pads[0];
                mPadLeft = mPadRight = mCommon.pads[1];
            } else if (mCommon.pads.size() >= 4) {
                mPadTop    = mCommon.pads[0];
                mPadLeft   = mCommon.pads[1];
                mPadBottom = mCommon.pads[2];
                mPadRight  = mCommon.pads[3];
            }
            mOutH = computeOutExtent(mInH, mCommon.kernelY, mCommon.strideY, mCommon.dilateY, mPadTop, false);
            mOutW = computeOutExtent(mInW, mCommon.kernelX, mCommon.strideX, mCommon.dilateX, mPadLeft, false);
        }

        outputs[0]->setShape({mBatch, mOutC, mOutH, mOutW});
        outputs[0]->setType(DT_FLOAT);
        return NO_ERROR;
    }

    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.empty() || outputs.empty()) {
            return INVALID_INPUT;
        }
        const float* input = inputs[0]->host<float>();
        float* output      = outputs[0]->host<float>();
        if (!input || !output) {
            return INVALID_INPUT;
        }
        const bool useDepthwisePath = mDepthwise || (mGroup == mInC && mOutC % mInC == 0);
        if (useDepthwisePath) {
            executeDepthwise(input, output);
        } else {
            executeGeneric(input, output);
        }
        return NO_ERROR;
    }

private:
    inline float activate(float value) const {
        if (mCommon.relu6) {
            return std::min(std::max(value, 0.0f), 6.0f);
        }
        if (mCommon.relu) {
            return std::max(value, 0.0f);
        }
        return value;
    }

    void executeDepthwise(const float* input, float* output) {
        const int kernelX = mCommon.kernelX;
        const int kernelY = mCommon.kernelY;
        const int outC    = mOutC;
        const int multiplier = mOutC / mInC;
        auto task = [&](int begin, int end) {
            for (int index = begin; index < end; ++index) {
                int n = index / (outC * mOutH);
                int r = index % (outC * mOutH);
                int oc = r / mOutH;
                int oh = r % mOutH;
                int ocIn = oc / multiplier;
                const float* src = input + (static_cast<int64_t>(n) * mInC + ocIn) * mInH * mInW;
                float bias       = oc < static_cast<int>(mBias.size()) ? mBias[oc] : 0.0f;
                const float* w   = mWeight.data() + static_cast<int64_t>(oc) * kernelY * kernelX;
                float* dst       = output + (static_cast<int64_t>(n) * outC + oc) * mOutH * mOutW + oh * mOutW;
                for (int ow = 0; ow < mOutW; ++ow) {
                    float sum = bias;
                    const int ih0 = oh * mCommon.strideY - mPadTop;
                    const int iw0 = ow * mCommon.strideX - mPadLeft;
                    for (int ky = 0; ky < kernelY; ++ky) {
                        const int ih = ih0 + ky * mCommon.dilateY;
                        if (ih < 0 || ih >= mInH) {
                            continue;
                        }
                        for (int kx = 0; kx < kernelX; ++kx) {
                            const int iw = iw0 + kx * mCommon.dilateX;
                            if (iw < 0 || iw >= mInW) {
                                continue;
                            }
                            sum += src[ih * mInW + iw] * w[ky * kernelX + kx];
                        }
                    }
                    dst[ow] = activate(sum);
                }
            }
        };
        const int total = mBatch * mOutC * mOutH;
        cpuPool(backend())->parallelFor(total, task, 1);
    }

    void executeGeneric(const float* input, float* output) {
        thread_local std::vector<float> colBuffer;
        const int kernelX = mCommon.kernelX;
        const int kernelY = mCommon.kernelY;
        const int colSize = mInCPerGroup * kernelY * kernelX * mOutH * mOutW;
        colBuffer.resize(colSize);

        const int planeIn  = mInH * mInW;
        const int planeOut = mOutH * mOutW;
        const int K        = mInCPerGroup * kernelY * kernelX;

        const float* colData = colBuffer.data();
        auto gemm = [&, colData](int begin, int end) {
            const float* col = colData;
            for (int oc = begin; oc < end; ++oc) {
                const float* weightRow = mWeight.data() + static_cast<int64_t>(oc) * K;
                float* dst = output + (static_cast<int64_t>(mTaskN) * mOutC + mTaskG * mOutCPerGroup + oc) * planeOut;
                const int outChannel = mTaskG * mOutCPerGroup + oc;
                const float bias = outChannel < static_cast<int>(mBias.size()) ? mBias[outChannel] : 0.0f;
                for (int j = 0; j < planeOut; ++j) {
                    dst[j] = bias;
                }
                for (int k = 0; k < K; ++k) {
                    const float wv = weightRow[k];
                    if (wv == 0.0f) {
                        continue;
                    }
                    const float* colRowPtr = col + static_cast<int64_t>(k) * planeOut;
                    for (int j = 0; j < planeOut; ++j) {
                        dst[j] += wv * colRowPtr[j];
                    }
                }
                if (mCommon.relu || mCommon.relu6) {
                    for (int j = 0; j < planeOut; ++j) {
                        dst[j] = activate(dst[j]);
                    }
                }
            }
        };

        for (int index = 0; index < mBatch * mGroup; ++index) {
            const int n = index / mGroup;
            const int g = index % mGroup;
            float* col  = colBuffer.data();
            // im2col for this (batch, group)
            int colRow = 0;
            for (int c = 0; c < mInCPerGroup; ++c) {
                const float* src = input + (static_cast<int64_t>(n) * mInC + g * mInCPerGroup + c) * planeIn;
                for (int ky = 0; ky < kernelY; ++ky) {
                    for (int kx = 0; kx < kernelX; ++kx) {
                        float* row = col + static_cast<int64_t>(colRow) * planeOut;
                        ++colRow;
                        for (int oh = 0; oh < mOutH; ++oh) {
                            const int ih = oh * mCommon.strideY - mPadTop + ky * mCommon.dilateY;
                            float* dstRow = row + oh * mOutW;
                            if (ih < 0 || ih >= mInH) {
                                std::memset(dstRow, 0, sizeof(float) * mOutW);
                                continue;
                            }
                            for (int ow = 0; ow < mOutW; ++ow) {
                                const int iw = ow * mCommon.strideX - mPadLeft + kx * mCommon.dilateX;
                                dstRow[ow]  = (iw >= 0 && iw < mInW) ? src[ih * mInW + iw] : 0.0f;
                            }
                        }
                    }
                }
            }
            mTaskN = n;
            mTaskG = g;
            cpuPool(backend())->parallelFor(mOutCPerGroup, gemm, 1);
        }
    }

    Conv2DCommon mCommon;
    const std::vector<float>& mWeight;
    const std::vector<float>& mBias;
    bool mDepthwise = false;

    int mBatch = 0, mInC = 0, mInH = 0, mInW = 0;
    int mOutC = 0, mOutH = 0, mOutW = 0;
    int mGroup = 1, mInCPerGroup = 1, mOutCPerGroup = 1;
    // Current (batch, group) slice being computed, shared with the GEMM tasks.
    int mTaskN = 0, mTaskG = 0;
    int mPadTop = 0, mPadBottom = 0, mPadLeft = 0, mPadRight = 0;
};

} // namespace

std::shared_ptr<Execution> createCPUConvolutionOp(CPUBackend* backend, const OpDesc& op,
                                                  const std::vector<Tensor*>&,
                                                  const std::vector<Tensor*>&) {
    return std::shared_ptr<Execution>(new CPUConvolution(backend, op));
}

} // namespace infer
