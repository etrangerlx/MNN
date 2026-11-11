#include <cmath>
#include <cstring>

#include "CPUUtils.h"

namespace infer {
namespace {

// --------------------------------------------------------------------------
// InnerProduct
// --------------------------------------------------------------------------
class CPUInnerProduct : public Execution {
public:
    CPUInnerProduct(CPUBackend* backend, const OpDesc& op)
        : Execution(backend), mWeight(op.attr.weight), mBias(op.attr.bias), mOutputCount(op.attr.outputCount),
          mTranspose(op.attr.transposeB) {
    }

    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.empty() || outputs.empty()) {
            return INVALID_INPUT;
        }
        const Tensor* input = inputs[0];
        const int rank      = input->dimensions();
        int batch = 1;
        for (int i = 0; i + 1 < rank; ++i) {
            batch *= input->length(i);
        }
        mBatch      = batch;
        mInputCount = rank > 0 ? input->length(rank - 1) : static_cast<int>(input->elementCount());
        if (!mBias.empty()) {
            mOutputCount = static_cast<int>(mBias.size());
        }
        if (mOutputCount <= 0) {
            return INVALID_INPUT;
        }
        const int expected = mInputCount * mOutputCount;
        if (static_cast<int>(mWeight.size()) < expected) {
            return INVALID_INPUT;
        }
        std::vector<int> shape = input->shape();
        if (shape.empty()) {
            shape.push_back(1);
        }
        shape.back() = mOutputCount;
        outputs[0]->setShape(shape);
        outputs[0]->setType(DT_FLOAT);
        return NO_ERROR;
    }

    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.empty() || outputs.empty()) {
            return INVALID_INPUT;
        }
        const float* src = inputs[0]->host<float>();
        float* dst       = outputs[0]->host<float>();
        if (!src || !dst) {
            return INVALID_INPUT;
        }
        const int batch = mBatch;
        const int K     = mInputCount;
        const int O     = mOutputCount;
        const bool transpose = mTranspose;
        cpuPool(backend())->parallelFor(batch, [&](int begin, int end) {
            for (int n = begin; n < end; ++n) {
                const float* row = src + static_cast<int64_t>(n) * K;
                float* outRow    = dst + static_cast<int64_t>(n) * O;
                for (int o = 0; o < O; ++o) {
                    float sum = mBias.empty() ? 0.0f : mBias[o];
                    if (!transpose) {
                        const float* w = mWeight.data() + static_cast<int64_t>(o) * K;
                        for (int k = 0; k < K; ++k) {
                            sum += row[k] * w[k];
                        }
                    } else {
                        for (int k = 0; k < K; ++k) {
                            sum += row[k] * mWeight[static_cast<int64_t>(k) * O + o];
                        }
                    }
                    outRow[o] = sum;
                }
            }
        }, 1);
        return NO_ERROR;
    }

private:
    std::vector<float> mWeight, mBias;
    int mOutputCount = 0;
    bool mTranspose   = false;
    int mBatch        = 0;
    int mInputCount   = 0;
};

// --------------------------------------------------------------------------
// MatMul
// --------------------------------------------------------------------------
class CPUMatMul : public Execution {
public:
    CPUMatMul(CPUBackend* backend, const OpDesc& op)
        : Execution(backend), mTransposeA(op.attr.transposeA), mTransposeB(op.attr.transposeB) {
    }

    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.size() < 2 || outputs.empty()) {
            return INVALID_INPUT;
        }
        const std::vector<int>& aShape = inputs[0]->shape();
        const std::vector<int>& bShape = inputs[1]->shape();
        if (aShape.size() < 2 || bShape.size() < 2) {
            return INVALID_INPUT;
        }
        const int aRank = static_cast<int>(aShape.size());
        const int bRank = static_cast<int>(bShape.size());
        mM = mTransposeA ? aShape[aRank - 1] : aShape[aRank - 2];
        mK = mTransposeA ? aShape[aRank - 2] : aShape[aRank - 1];
        const int bK = mTransposeB ? bShape[bRank - 1] : bShape[bRank - 2];
        mN = mTransposeB ? bShape[bRank - 2] : bShape[bRank - 1];
        if (mK != bK) {
            return INVALID_INPUT;
        }
        std::vector<int> aBatch(aShape.begin(), aShape.end() - 2);
        std::vector<int> bBatch(bShape.begin(), bShape.end() - 2);
        mBatchShape = broadcastShape(aBatch, bBatch);
        mBatchCount = static_cast<int>(shapeProduct(mBatchShape));
        std::vector<int> shape = mBatchShape;
        shape.push_back(mM);
        shape.push_back(mN);
        outputs[0]->setShape(shape);
        outputs[0]->setType(DT_FLOAT);

        mAStrides = broadcastStrides(aBatch, static_cast<int>(mBatchShape.size()));
        mBStrides = broadcastStrides(bBatch, static_cast<int>(mBatchShape.size()));
        return NO_ERROR;
    }

    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.size() < 2 || outputs.empty()) {
            return INVALID_INPUT;
        }
        const float* a = inputs[0]->host<float>();
        const float* b = inputs[1]->host<float>();
        float* out     = outputs[0]->host<float>();
        if (!a || !b || !out) {
            return INVALID_INPUT;
        }
        const int M = mM, N = mN, K = mK;
        const bool transposeA = mTransposeA;
        const bool transposeB = mTransposeB;
        const int batchRank   = static_cast<int>(mBatchShape.size());
        const std::vector<int> aStrides = mAStrides;
        const std::vector<int> bStrides = mBStrides;
        const std::vector<int> batchShape = mBatchShape;
        const int batchCount = mBatchCount;
        auto task = [&](int begin, int end) {
            std::vector<int> coord(batchRank, 0);
            for (int index = begin; index < end; ++index) {
                const int batchIndex = index / M;
                const int row        = index % M;
                int rest             = batchIndex;
                int64_t aOffset      = 0;
                int64_t bOffset      = 0;
                for (int d = batchRank - 1; d >= 0; --d) {
                    const int c = rest % batchShape[d];
                    rest /= batchShape[d];
                    aOffset += static_cast<int64_t>(c) * aStrides[d];
                    bOffset += static_cast<int64_t>(c) * bStrides[d];
                }
                const float* aRow = a + aOffset;
                const float* bMat = b + bOffset;
                float* outRow     = out + static_cast<int64_t>(batchIndex) * M * N + static_cast<int64_t>(row) * N;
                for (int n = 0; n < N; ++n) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; ++k) {
                        const float av = transposeA ? aRow[static_cast<int64_t>(k) * M + row] : aRow[static_cast<int64_t>(row) * K + k];
                        const float bv = transposeB ? bMat[static_cast<int64_t>(n) * K + k] : bMat[static_cast<int64_t>(k) * N + n];
                        sum += av * bv;
                    }
                    outRow[n] = sum;
                }
            }
        };
        cpuPool(backend())->parallelFor(batchCount * M, task, 1);
        return NO_ERROR;
    }

private:
    bool mTransposeA = false;
    bool mTransposeB = false;
    int mM = 0, mN = 0, mK = 0, mBatchCount = 1;
    std::vector<int> mBatchShape, mAStrides, mBStrides;
};

// --------------------------------------------------------------------------
// Interp
// --------------------------------------------------------------------------
class CPUInterp : public Execution {
public:
    CPUInterp(CPUBackend* backend, const OpDesc& op) : Execution(backend), mAttr(op.attr) {
    }

    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.empty() || outputs.empty() || inputs[0]->dimensions() != 4) {
            return NOT_SUPPORT;
        }
        const Tensor* input = inputs[0];
        mBatch   = input->batch();
        mChannel = input->channel();
        mInH     = input->height();
        mInW     = input->width();

        int outH = mAttr.outputHeight;
        int outW = mAttr.outputWidth;
        if (inputs.size() > 1 && inputs[1]->host()) {
            const Tensor* sizeTensor = inputs[1];
            const int count          = static_cast<int>(sizeTensor->elementCount());
            if (count >= 2) {
                if (sizeTensor->type() == DT_INT32) {
                    outH = sizeTensor->host<int32_t>()[0];
                    outW = sizeTensor->host<int32_t>()[1];
                }
            }
        }
        if (outH <= 0) {
            outH = mAttr.heightScale > 0.0f ? static_cast<int>(std::round(mInH * mAttr.heightScale)) : mInH;
        }
        if (outW <= 0) {
            outW = mAttr.widthScale > 0.0f ? static_cast<int>(std::round(mInW * mAttr.widthScale)) : mInW;
        }
        mOutH = outH;
        mOutW = outW;
        // MNN recomputes the sampling transform from the actual input/output
        // sizes (TensorFlow semantics) instead of using the stored scale, which
        // only drives the output size here.
        if (mAttr.halfPixelCenters && mAttr.resizeType != 1) {
            mScaleX  = static_cast<float>(mInW) / std::max(1, outW);
            mScaleY  = static_cast<float>(mInH) / std::max(1, outH);
            mOffsetX = 0.5f * mScaleX - 0.5f;
            mOffsetY = 0.5f * mScaleY - 0.5f;
        } else if (mAttr.alignCorners) {
            mScaleX  = outW > 1 ? static_cast<float>(mInW - 1) / static_cast<float>(outW - 1) : 0.0f;
            mScaleY  = outH > 1 ? static_cast<float>(mInH - 1) / static_cast<float>(outH - 1) : 0.0f;
            mOffsetX = 0.0f;
            mOffsetY = 0.0f;
        } else {
            mScaleX  = static_cast<float>(mInW) / std::max(1, outW);
            mScaleY  = static_cast<float>(mInH) / std::max(1, outH);
            mOffsetX = 0.0f;
            mOffsetY = 0.0f;
        }
        outputs[0]->setShape({mBatch, mChannel, outH, outW});
        outputs[0]->setType(DT_FLOAT);
        return NO_ERROR;
    }

    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.empty() || outputs.empty()) {
            return INVALID_INPUT;
        }
        const float* src = inputs[0]->host<float>();
        float* dst       = outputs[0]->host<float>();
        if (!src || !dst) {
            return INVALID_INPUT;
        }
        const int inPlane  = mInH * mInW;
        const int outPlane = mOutH * mOutW;
        const bool bilinear = mAttr.resizeType == 2;
        const float scaleX = mScaleX;
        const float scaleY = mScaleY;
        const float offsetX = mOffsetX;
        const float offsetY = mOffsetY;
        auto task = [&](int begin, int end) {
            for (int index = begin; index < end; ++index) {
                const int n  = index / mChannel;
                const int c  = index % mChannel;
                const float* plane = src + (static_cast<int64_t>(n) * mChannel + c) * inPlane;
                float* outPlanePtr = dst + (static_cast<int64_t>(n) * mChannel + c) * outPlane;
                for (int oy = 0; oy < mOutH; ++oy) {
                    const float sy = oy * scaleY + offsetY;
                    for (int ox = 0; ox < mOutW; ++ox) {
                        const float sx = ox * scaleX + offsetX;
                        if (!bilinear) {
                            int x = clampInt(static_cast<int>(std::floor(sx)), 0, mInW - 1);
                            int y = clampInt(static_cast<int>(std::floor(sy)), 0, mInH - 1);
                            outPlanePtr[oy * mOutW + ox] = plane[y * mInW + x];
                        } else {
                            const int x0 = clampInt(static_cast<int>(std::floor(sx)), 0, mInW - 1);
                            const int y0 = clampInt(static_cast<int>(std::floor(sy)), 0, mInH - 1);
                            const int x1 = clampInt(x0 + 1, 0, mInW - 1);
                            const int y1 = clampInt(y0 + 1, 0, mInH - 1);
                            const float fx = std::min(std::max(sx - std::floor(sx), 0.0f), 1.0f);
                            const float fy = std::min(std::max(sy - std::floor(sy), 0.0f), 1.0f);
                            const float v00 = plane[y0 * mInW + x0];
                            const float v01 = plane[y0 * mInW + x1];
                            const float v10 = plane[y1 * mInW + x0];
                            const float v11 = plane[y1 * mInW + x1];
                            outPlanePtr[oy * mOutW + ox] =
                                v00 * (1 - fx) * (1 - fy) + v01 * fx * (1 - fy) + v10 * (1 - fx) * fy +
                                v11 * fx * fy;
                        }
                    }
                }
            }
        };
        cpuPool(backend())->parallelFor(mBatch * mChannel, task, 1);
        return NO_ERROR;
    }

private:
    OpAttr mAttr;
    int mBatch = 0, mChannel = 0, mInH = 0, mInW = 0, mOutH = 0, mOutW = 0;
    float mScaleX = 1.0f, mScaleY = 1.0f;
    float mOffsetX = 0.0f, mOffsetY = 0.0f;
};

// --------------------------------------------------------------------------
// Padding
// --------------------------------------------------------------------------
class CPUPadding : public Execution {
public:
    CPUPadding(CPUBackend* backend, const OpDesc& op) : Execution(backend), mAttr(op.attr) {
    }

    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.size() < 2 || outputs.empty()) {
            return INVALID_INPUT;
        }
        const Tensor* input    = inputs[0];
        const Tensor* paddings = inputs[1];
        const int rank         = input->dimensions();
        if (static_cast<int>(paddings->elementCount()) < rank * 2) {
            return INVALID_INPUT;
        }
        mPads.resize(rank * 2);
        for (int i = 0; i < rank * 2; ++i) {
            mPads[i] = paddings->type() == DT_INT32 ? paddings->host<int32_t>()[i]
                                                    : static_cast<int>(paddings->host<int64_t>()[i]);
        }
        std::vector<int> shape(rank, 1);
        for (int i = 0; i < rank; ++i) {
            shape[i] = input->length(i) + mPads[2 * i] + mPads[2 * i + 1];
        }
        mRank = rank;
        outputs[0]->setShape(shape);
        outputs[0]->setType(input->type());
        return NO_ERROR;
    }

    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.empty() || outputs.empty()) {
            return INVALID_INPUT;
        }
        const Tensor* input = inputs[0];
        float* dst          = outputs[0]->host<float>();
        if (!dst) {
            return INVALID_INPUT;
        }
        float padValue = 0.0f;
        if (inputs.size() > 2 && inputs[2]->host()) {
            padValue = inputs[2]->type() == DT_FLOAT ? inputs[2]->host<float>()[0] : 0.0f;
        }
        if (mAttr.padMode == PadValueMode_CONSTANT && padValue == 0.0f) {
            std::memset(dst, 0, outputs[0]->bytes());
        }
        const float* src = input->host<float>();
        const int total  = static_cast<int>(outputs[0]->elementCount());
        const std::vector<int> outShape = outputs[0]->shape();
        const std::vector<int> outStrides = computeStrides(outShape);
        const std::vector<int> inStrides  = computeStrides(input->shape());
        const std::vector<int> pads = mPads;
        const int mode = mAttr.padMode;
        const int rank = mRank;
        cpuPool(backend())->parallelFor(total, [&](int begin, int end) {
            for (int index = begin; index < end; ++index) {
                int rest = index;
                int64_t offset = 0;
                bool inside    = true;
                for (int d = rank - 1; d >= 0; --d) {
                    const int c = rest % outShape[d];
                    rest /= outShape[d];
                    const int extent = input->length(d);
                    int srcIndex     = c - pads[2 * d];
                    if (srcIndex < 0 || srcIndex >= extent) {
                        inside = false;
                        if (mode == PadValueMode_CONSTANT) {
                            break;
                        }
                        if (srcIndex < 0) {
                            srcIndex = mode == PadValueMode_REFLECT ? -srcIndex : -srcIndex - 1;
                        } else {
                            const int over = srcIndex - extent;
                            srcIndex = mode == PadValueMode_REFLECT ? extent - 2 - over : extent - 1 - over;
                        }
                        srcIndex = clampInt(srcIndex, 0, extent - 1);
                    }
                    offset += static_cast<int64_t>(srcIndex) * inStrides[d];
                }
                if (!inside && mode == PadValueMode_CONSTANT) {
                    dst[index] = padValue;
                } else {
                    dst[index] = src[offset];
                }
            }
        }, 256);
        return NO_ERROR;
    }

private:
    OpAttr mAttr;
    std::vector<int> mPads;
    int mRank = 0;
};

} // namespace

std::shared_ptr<Execution> createCPUInnerProductOp(CPUBackend* backend, const OpDesc& op,
                                                   const std::vector<Tensor*>&,
                                                   const std::vector<Tensor*>&) {
    switch (op.type) {
        case OpType_InnerProduct:
            return std::shared_ptr<Execution>(new CPUInnerProduct(backend, op));
        case OpType_MatMul:
            return std::shared_ptr<Execution>(new CPUMatMul(backend, op));
        case OpType_Interp:
            return std::shared_ptr<Execution>(new CPUInterp(backend, op));
        case OpType_Padding:
            return std::shared_ptr<Execution>(new CPUPadding(backend, op));
        default:
            return nullptr;
    }
}

} // namespace infer
