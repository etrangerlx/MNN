#include <cfloat>
#include <cmath>
#include <cstring>
#include <limits>

#include "CPUUtils.h"

namespace infer {
namespace {

class CPUActivation : public Execution {
public:
    CPUActivation(CPUBackend* backend, const OpDesc& op)
        : Execution(backend), mType(op.type), mAttr(op.attr) {
    }

    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.empty() || outputs.empty()) {
            return INVALID_INPUT;
        }
        outputs[0]->setShape(inputs[0]->shape());
        outputs[0]->setType(inputs[0]->type());
        return NO_ERROR;
    }

    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        const float* input = inputs[0]->host<float>();
        float* output      = outputs[0]->host<float>();
        if (!input || !output) {
            return INVALID_INPUT;
        }
        const int total = static_cast<int>(outputs[0]->elementCount());
        switch (mType) {
            case OpType_ReLU: {
                const float slope = mAttr.slopeValue;
                cpuPool(backend())->parallelFor(total, [&](int begin, int end) {
                    for (int i = begin; i < end; ++i) {
                        const float v = input[i];
                        output[i]     = v > 0.0f ? v : v * slope;
                    }
                }, 1024);
                break;
            }
            case OpType_ReLU6: {
                const float minValue = mAttr.minValue;
                const float maxValue = mAttr.maxValue;
                cpuPool(backend())->parallelFor(total, [&](int begin, int end) {
                    for (int i = begin; i < end; ++i) {
                        output[i] = std::min(std::max(input[i], minValue), maxValue);
                    }
                }, 1024);
                break;
            }
            case OpType_PReLU: {
                const int channels = inputs[0]->channel();
                const int plane    = channels > 0 ? static_cast<int>(inputs[0]->elementCount()) / channels : total;
                const int slopes   = static_cast<int>(mAttr.slope.size());
                cpuPool(backend())->parallelFor(total, [&](int begin, int end) {
                    for (int i = begin; i < end; ++i) {
                        const float v = input[i];
                        if (v > 0.0f) {
                            output[i] = v;
                        } else {
                            const int c = slopes > 0 ? ((i / plane) % channels) : 0;
                            const float s = slopes > 0 ? mAttr.slope[std::min(c, slopes - 1)]
                                                       : mAttr.slopeValue;
                            output[i] = v * s;
                        }
                    }
                }, 1024);
                break;
            }
            default:
                return NOT_SUPPORT;
        }
        (void)total;
        return NO_ERROR;
    }

private:
    OpType mType;
    OpAttr mAttr;
};

class CPUSoftmax : public Execution {
public:
    CPUSoftmax(CPUBackend* backend, const OpDesc& op) : Execution(backend), mAxis(op.attr.axis) {
    }

    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.empty() || outputs.empty()) {
            return INVALID_INPUT;
        }
        outputs[0]->setShape(inputs[0]->shape());
        outputs[0]->setType(inputs[0]->type());
        return NO_ERROR;
    }

    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        const Tensor* input = inputs[0];
        const float* src    = input->host<float>();
        float* dst          = outputs[0]->host<float>();
        if (!src || !dst) {
            return INVALID_INPUT;
        }
        int axis = mAxis;
        if (axis < 0) {
            axis += input->dimensions();
        }
        if (axis < 0 || axis >= input->dimensions()) {
            return INVALID_INPUT;
        }
        const int outside = static_cast<int>(shapeProduct(input->shape(), 0, axis));
        const int dim     = input->length(axis);
        const int inside  = static_cast<int>(shapeProduct(input->shape(), axis + 1));
        if (dim <= 0) {
            return NO_ERROR;
        }
        cpuPool(backend())->parallelFor(outside * inside, [&](int begin, int end) {
            for (int index = begin; index < end; ++index) {
                const int o = index / inside;
                const int i = index % inside;
                const float* row = src + static_cast<int64_t>(o) * dim * inside + i;
                float maxValue   = row[0];
                for (int k = 1; k < dim; ++k) {
                    maxValue = std::max(maxValue, row[static_cast<int64_t>(k) * inside]);
                }
                float sum = 0.0f;
                for (int k = 0; k < dim; ++k) {
                    const float e = std::exp(row[static_cast<int64_t>(k) * inside] - maxValue);
                    dst[static_cast<int64_t>(o) * dim * inside + static_cast<int64_t>(k) * inside + i] = e;
                    sum += e;
                }
                const float inv = 1.0f / sum;
                for (int k = 0; k < dim; ++k) {
                    dst[static_cast<int64_t>(o) * dim * inside + static_cast<int64_t>(k) * inside + i] *= inv;
                }
            }
        }, 8);
        return NO_ERROR;
    }

private:
    int mAxis = 1;
};

inline float applyUnary(int type, float x) {
    switch (type) {
        case UnaryOpOperation_ABS:
            return std::fabs(x);
        case UnaryOpOperation_NEG:
            return -x;
        case UnaryOpOperation_FLOOR:
            return std::floor(x);
        case UnaryOpOperation_CEIL:
            return std::ceil(x);
        case UnaryOpOperation_SQUARE:
            return x * x;
        case UnaryOpOperation_SQRT:
            return std::sqrt(x);
        case UnaryOpOperation_RSQRT:
            return 1.0f / std::sqrt(x);
        case UnaryOpOperation_EXP:
            return std::exp(x);
        case UnaryOpOperation_LOG:
            return std::log(x);
        case UnaryOpOperation_SIN:
            return std::sin(x);
        case UnaryOpOperation_COS:
            return std::cos(x);
        case UnaryOpOperation_TAN:
            return std::tan(x);
        case UnaryOpOperation_ASIN:
            return std::asin(x);
        case UnaryOpOperation_ACOS:
            return std::acos(x);
        case UnaryOpOperation_ATAN:
            return std::atan(x);
        case UnaryOpOperation_RECIPROCAL:
            return 1.0f / x;
        case UnaryOpOperation_LOG1P:
            return std::log1p(x);
        case UnaryOpOperation_SIGN:
            return (x > 0.0f) - (x < 0.0f);
        case UnaryOpOperation_ROUND:
            return std::round(x);
        case UnaryOpOperation_EXPM1:
            return std::expm1(x);
        case UnaryOpOperation_SIGMOID:
            return 1.0f / (1.0f + std::exp(-x));
        case UnaryOpOperation_TANH:
            return std::tanh(x);
        case UnaryOpOperation_HARDSWISH: {
            const float y = x * std::min(std::max(x + 3.0f, 0.0f), 6.0f);
            return y / 6.0f;
        }
        case UnaryOpOperation_GELU:
        case UnaryOpOperation_GELU_STANDARD: {
            const float cdf = 0.5f * (1.0f + std::erf(x * 0.7071067811865475f));
            return x * cdf;
        }
        case UnaryOpOperation_SILU:
            return x / (1.0f + std::exp(-x));
        default:
            return x;
    }
}

class CPUUnaryOp : public Execution {
public:
    CPUUnaryOp(CPUBackend* backend, const OpDesc& op) : Execution(backend), mType(op.attr.unaryOp) {
    }

    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.empty() || outputs.empty()) {
            return INVALID_INPUT;
        }
        outputs[0]->setShape(inputs[0]->shape());
        outputs[0]->setType(inputs[0]->type());
        return NO_ERROR;
    }

    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        const float* src = inputs[0]->host<float>();
        float* dst       = outputs[0]->host<float>();
        if (!src || !dst) {
            return INVALID_INPUT;
        }
        const int total = static_cast<int>(outputs[0]->elementCount());
        const int type  = mType;
        cpuPool(backend())->parallelFor(total, [&](int begin, int end) {
            for (int i = begin; i < end; ++i) {
                dst[i] = applyUnary(type, src[i]);
            }
        }, 1024);
        return NO_ERROR;
    }

private:
    int mType;
};

inline float applyBinary(int type, float a, float b) {
    switch (type) {
        case BinaryOpOperation_ADD:
            return a + b;
        case BinaryOpOperation_SUB:
            return a - b;
        case BinaryOpOperation_MUL:
            return a * b;
        case BinaryOpOperation_DIV:
        case BinaryOpOperation_REALDIV:
            return a / b;
        case BinaryOpOperation_MAX_TEMP:
        case BinaryOpOperation_MAXIMUM:
            return std::max(a, b);
        case BinaryOpOperation_MIN_TEMP:
        case BinaryOpOperation_MINIMUM:
            return std::min(a, b);
        case BinaryOpOperation_POW:
            return std::pow(a, b);
        case BinaryOpOperation_FLOORDIV:
            return std::floor(a / b);
        case BinaryOpOperation_SquaredDifference: {
            const float d = a - b;
            return d * d;
        }
        case BinaryOpOperation_FLOORMOD:
            return a - std::floor(a / b) * b;
        case BinaryOpOperation_MOD:
            return std::fmod(a, b);
        case BinaryOpOperation_ATAN2:
            return std::atan2(a, b);
        case BinaryOpOperation_MUL_SILU:
            return a * b / (1.0f + std::exp(-(a * b)));
        default:
            return a + b;
    }
}

inline float applyCompare(int type, float a, float b) {
    return applyBinary(type, a, b);
}

class CPUBinaryOp : public Execution {
public:
    CPUBinaryOp(CPUBackend* backend, const OpDesc& op) : Execution(backend), mType(op.attr.binaryOp) {
    }

    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.size() < 2 || outputs.empty()) {
            return INVALID_INPUT;
        }
        outputs[0]->setShape(broadcastShape(inputs[0]->shape(), inputs[1]->shape()));
        outputs[0]->setType(inputs[0]->type());
        return NO_ERROR;
    }

    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        const Tensor* inputA = inputs[0];
        const Tensor* inputB = inputs[1];
        const float* a       = inputA->host<float>();
        const float* b       = inputB->host<float>();
        float* out           = outputs[0]->host<float>();
        if (!a || !b || !out) {
            return INVALID_INPUT;
        }
        const std::vector<int>& shape = outputs[0]->shape();
        const int rank                = static_cast<int>(shape.size());
        const int total               = static_cast<int>(outputs[0]->elementCount());
        const std::vector<int> stridesA = broadcastStrides(inputA->shape(), rank);
        const std::vector<int> stridesB = broadcastStrides(inputB->shape(), rank);
        const int type = mType;

        if (sameShape(inputA->shape(), inputB->shape())) {
            cpuPool(backend())->parallelFor(total, [&](int begin, int end) {
                for (int i = begin; i < end; ++i) {
                    out[i] = applyBinary(type, a[i], b[i]);
                }
            }, 1024);
            return NO_ERROR;
        }

        cpuPool(backend())->parallelFor(total, [&](int begin, int end) {
            std::vector<int> coord(rank, 0);
            for (int index = begin; index < end; ++index) {
                int64_t offsetA = 0;
                int64_t offsetB = 0;
                int rest        = index;
                for (int d = rank - 1; d >= 0; --d) {
                    const int c = rest % shape[d];
                    rest /= shape[d];
                    offsetA += static_cast<int64_t>(c) * stridesA[d];
                    offsetB += static_cast<int64_t>(c) * stridesB[d];
                }
                out[index] = applyBinary(type, a[offsetA], b[offsetB]);
            }
        }, 256);
        return NO_ERROR;
    }

private:
    int mType;
};

class CPUEltwise : public Execution {
public:
    CPUEltwise(CPUBackend* backend, const OpDesc& op) : Execution(backend), mType(op.attr.eltwiseType) {
        if (op.attr.coeff.empty()) {
            mCoeff.assign(op.inputs.size(), 1.0f);
        } else {
            mCoeff = op.attr.coeff;
        }
    }

    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.empty() || outputs.empty()) {
            return INVALID_INPUT;
        }
        outputs[0]->setShape(inputs[0]->shape());
        outputs[0]->setType(inputs[0]->type());
        return NO_ERROR;
    }

    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.empty()) {
            return INVALID_INPUT;
        }
        const int total = static_cast<int>(outputs[0]->elementCount());
        float* dst      = outputs[0]->host<float>();
        const int count = static_cast<int>(inputs.size());
        const int type  = mType;
        std::vector<const float*> src(count);
        for (int i = 0; i < count; ++i) {
            src[i] = inputs[i]->host<float>();
        }
        cpuPool(backend())->parallelFor(total, [&](int begin, int end) {
            for (int i = begin; i < end; ++i) {
                float value = 0.0f;
                switch (type) {
                    case EltwiseType_PROD:
                        value = 1.0f;
                        for (int k = 0; k < count; ++k) {
                            value *= src[k][i];
                        }
                        break;
                    case EltwiseType_SUM:
                        for (int k = 0; k < count; ++k) {
                            value += src[k][i] * mCoeff[std::min(k, static_cast<int>(mCoeff.size()) - 1)];
                        }
                        break;
                    case EltwiseType_MAXIMUM:
                        value = src[0][i];
                        for (int k = 1; k < count; ++k) {
                            value = std::max(value, src[k][i]);
                        }
                        break;
                    case EltwiseType_SUB:
                        value = src[0][i];
                        for (int k = 1; k < count; ++k) {
                            value -= src[k][i];
                        }
                        break;
                    default:
                        break;
                }
                dst[i] = value;
            }
        }, 1024);
        return NO_ERROR;
    }

private:
    int mType;
    std::vector<float> mCoeff;
};

class CPUReduction : public Execution {
public:
    CPUReduction(CPUBackend* backend, const OpDesc& op)
        : Execution(backend), mType(op.attr.reductionOp), mDims(op.attr.reductionDims), mKeepDims(op.attr.keepDims) {
    }

    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.empty() || outputs.empty()) {
            return INVALID_INPUT;
        }
        const Tensor* input = inputs[0];
        const int rank      = input->dimensions();
        std::vector<int> axes = normalizeAxes(rank, effectiveDims(inputs));
        std::vector<bool> reduced(rank, false);
        for (int axis : axes) {
            reduced[axis] = true;
        }
        std::vector<int> shape;
        for (int i = 0; i < rank; ++i) {
            if (mKeepDims) {
                shape.push_back(reduced[i] ? 1 : input->length(i));
            } else if (!reduced[i]) {
                shape.push_back(input->length(i));
            }
        }
        if (shape.empty()) {
            shape.push_back(1);
        }
        outputs[0]->setShape(shape);
        outputs[0]->setType(input->type());
        return NO_ERROR;
    }

    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        const Tensor* input = inputs[0];
        const float* src    = input->host<float>();
        float* dst          = outputs[0]->host<float>();
        if (!src || !dst) {
            return INVALID_INPUT;
        }
        const int rank         = input->dimensions();
        const std::vector<int> axes = normalizeAxes(rank, effectiveDims(inputs));
        std::vector<bool> reduced(rank, false);
        for (int axis : axes) {
            reduced[axis] = true;
        }
        std::vector<int> keptDims;
        std::vector<int> keptExtents;
        std::vector<int> reducedExtents;
        for (int i = 0; i < rank; ++i) {
            if (reduced[i]) {
                reducedExtents.push_back(input->length(i));
            } else {
                keptDims.push_back(i);
                keptExtents.push_back(input->length(i));
            }
        }
        const std::vector<int> inStrides = computeStrides(input->shape());
        const int outCount               = static_cast<int>(outputs[0]->elementCount());
        const int reducedCount           = static_cast<int>(shapeProduct(reducedExtents));
        const int keptRank               = static_cast<int>(keptDims.size());
        const int type                   = mType;

        cpuPool(backend())->parallelFor(outCount, [&](int begin, int end) {
            std::vector<int> coord(keptRank, 0);
            std::vector<int> rcoord(reducedExtents.size(), 0);
            for (int index = begin; index < end; ++index) {
                int rest = index;
                for (int d = keptRank - 1; d >= 0; --d) {
                    coord[d] = rest % keptExtents[d];
                    rest /= keptExtents[d];
                }
                int64_t base = 0;
                for (int d = 0; d < keptRank; ++d) {
                    base += static_cast<int64_t>(coord[d]) * inStrides[keptDims[d]];
                }
                float acc;
                switch (type) {
                    case ReductionType_MAXIMUM:
                        acc = -std::numeric_limits<float>::max();
                        break;
                    case ReductionType_MINIMUM:
                        acc = std::numeric_limits<float>::max();
                        break;
                    case ReductionType_PROD:
                        acc = 1.0f;
                        break;
                    default:
                        acc = 0.0f;
                        break;
                }
                for (int r = 0; r < reducedCount; ++r) {
                    int restR = r;
                    int64_t offset = base;
                    for (int d = static_cast<int>(reducedExtents.size()) - 1; d >= 0; --d) {
                        const int c = restR % reducedExtents[d];
                        restR /= reducedExtents[d];
                        offset += static_cast<int64_t>(c) * inStrides[axes[d]];
                    }
                    const float v = src[offset];
                    switch (type) {
                        case ReductionType_SUM:
                        case ReductionType_MEAN:
                        case ReductionType_ASUM:
                        case ReductionType_SUMSQ:
                            acc += (type == ReductionType_SUMSQ) ? v * v : (type == ReductionType_ASUM ? std::fabs(v) : v);
                            break;
                        case ReductionType_MAXIMUM:
                            acc = std::max(acc, v);
                            break;
                        case ReductionType_MINIMUM:
                            acc = std::min(acc, v);
                            break;
                        case ReductionType_PROD:
                            acc *= v;
                            break;
                        default:
                            acc += v;
                            break;
                    }
                }
                if (type == ReductionType_MEAN && reducedCount > 0) {
                    acc /= static_cast<float>(reducedCount);
                }
                dst[index] = acc;
            }
        }, 64);
        return NO_ERROR;
    }

private:
    // Reduction axes may come from the attribute or from a second (constant)
    // input tensor, as in the TFLite-derived models.
    std::vector<int> effectiveDims(const std::vector<Tensor*>& inputs) const {
        if (!mDims.empty() || inputs.size() < 2 || !inputs[1]->host()) {
            return mDims;
        }
        const int count = static_cast<int>(inputs[1]->elementCount());
        std::vector<int> dims;
        if (inputs[1]->type() == DT_INT64) {
            const int64_t* src = inputs[1]->host<int64_t>();
            for (int i = 0; i < count; ++i) {
                dims.push_back(static_cast<int>(src[i]));
            }
        } else {
            const int32_t* src = inputs[1]->host<int32_t>();
            for (int i = 0; i < count; ++i) {
                dims.push_back(src[i]);
            }
        }
        return dims;
    }

    std::vector<int> normalizeAxes(int rank, const std::vector<int>& dims) const {
        std::vector<int> axes;
        if (dims.empty()) {
            for (int i = 0; i < rank; ++i) {
                axes.push_back(i);
            }
            return axes;
        }
        for (int dim : dims) {
            int axis = dim < 0 ? dim + rank : dim;
            if (axis >= 0 && axis < rank) {
                axes.push_back(axis);
            }
        }
        if (axes.empty()) {
            for (int i = 0; i < rank; ++i) {
                axes.push_back(i);
            }
        }
        return axes;
    }

    int mType;
    std::vector<int> mDims;
    bool mKeepDims = false;
};

class CPUScale : public Execution {
public:
    CPUScale(CPUBackend* backend, const OpDesc& op)
        : Execution(backend), mScale(op.attr.scale), mOffset(op.attr.offset), mChannels(op.attr.channels) {
    }

    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.empty() || outputs.empty()) {
            return INVALID_INPUT;
        }
        outputs[0]->setShape(inputs[0]->shape());
        outputs[0]->setType(inputs[0]->type());
        return NO_ERROR;
    }

    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        const Tensor* input = inputs[0];
        const float* src    = input->host<float>();
        float* dst          = outputs[0]->host<float>();
        if (!src || !dst) {
            return INVALID_INPUT;
        }
        const int total = static_cast<int>(input->elementCount());
        if (mScale.empty()) {
            std::memcpy(dst, src, sizeof(float) * total);
            return NO_ERROR;
        }
        if (mScale.size() == 1) {
            const float s = mScale[0];
            const float b = mOffset.empty() ? 0.0f : mOffset[0];
            cpuPool(backend())->parallelFor(total, [&](int begin, int end) {
                for (int i = begin; i < end; ++i) {
                    dst[i] = src[i] * s + b;
                }
            }, 1024);
            return NO_ERROR;
        }
        if (static_cast<int>(mScale.size()) == total) {
            cpuPool(backend())->parallelFor(total, [&](int begin, int end) {
                for (int i = begin; i < end; ++i) {
                    dst[i] = src[i] * mScale[i] + (mOffset.empty() ? 0.0f : mOffset[i]);
                }
            }, 1024);
            return NO_ERROR;
        }
        const int channels = static_cast<int>(mScale.size());
        const int plane    = total / std::max(1, mChannels > 0 ? mChannels : channels);
        cpuPool(backend())->parallelFor(total, [&](int begin, int end) {
            for (int i = begin; i < end; ++i) {
                const int c = (i / plane) % channels;
                dst[i]      = src[i] * mScale[c] + (mOffset.empty() ? 0.0f : mOffset[std::min(c, static_cast<int>(mOffset.size()) - 1)]);
            }
        }, 1024);
        return NO_ERROR;
    }

private:
    std::vector<float> mScale, mOffset;
    int mChannels = 0;
};

} // namespace

std::shared_ptr<Execution> createCPUUnaryOp(CPUBackend* backend, const OpDesc& op,
                                            const std::vector<Tensor*>&,
                                            const std::vector<Tensor*>&) {
    switch (op.type) {
        case OpType_ReLU:
        case OpType_ReLU6:
        case OpType_PReLU:
            return std::shared_ptr<Execution>(new CPUActivation(backend, op));
        case OpType_Softmax:
            return std::shared_ptr<Execution>(new CPUSoftmax(backend, op));
        case OpType_UnaryOp:
            return std::shared_ptr<Execution>(new CPUUnaryOp(backend, op));
        case OpType_BinaryOp:
            return std::shared_ptr<Execution>(new CPUBinaryOp(backend, op));
        case OpType_Eltwise:
            return std::shared_ptr<Execution>(new CPUEltwise(backend, op));
        case OpType_Reduction:
            return std::shared_ptr<Execution>(new CPUReduction(backend, op));
        case OpType_Scale:
            return std::shared_ptr<Execution>(new CPUScale(backend, op));
        default:
            return nullptr;
    }
}

} // namespace infer
