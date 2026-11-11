#include <algorithm>
#include <cmath>
#include <cstring>
#include <set>

#include "CPUUtils.h"

namespace infer {
namespace {

// --------------------------------------------------------------------------
// Element-wise copy style operators: Const / Input / Identity / ConvertTensor
// --------------------------------------------------------------------------
class CPUNoop : public Execution {
public:
    CPUNoop(CPUBackend* backend, const OpDesc& op)
        : Execution(backend), mType(op.type), mSrcFormat(op.attr.srcFormat), mDstFormat(op.attr.dstFormat) {
    }

    // MNN dimension formats: 0 = NCHW, 1 = NHWC, 2 = NC4HW4. NC4HW4 is only a
    // physical blocking of NCHW, so it shares the logical order with NCHW.
    static bool isNHWC(int format) {
        return format == 1;
    }

    bool needTranspose() const {
        return mType == OpType_ConvertTensor && isNHWC(mSrcFormat) != isNHWC(mDstFormat);
    }

    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (mType != OpType_Identity && mType != OpType_ConvertTensor) {
            return NO_ERROR;
        }
        if (inputs.empty() || outputs.empty()) {
            return INVALID_INPUT;
        }
        std::vector<int> shape = inputs[0]->shape();
        if (needTranspose() && shape.size() == 4) {
            if (isNHWC(mSrcFormat)) {
                // [N, H, W, C] -> [N, C, H, W]
                shape = {shape[0], shape[3], shape[1], shape[2]};
            } else {
                // [N, C, H, W] -> [N, H, W, C]
                shape = {shape[0], shape[2], shape[3], shape[1]};
            }
        }
        outputs[0]->setShape(shape);
        outputs[0]->setType(inputs[0]->type());
        return NO_ERROR;
    }

    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (mType != OpType_Identity && mType != OpType_ConvertTensor) {
            return NO_ERROR;
        }
        if (inputs.empty() || outputs.empty()) {
            return INVALID_INPUT;
        }
        if (!inputs[0]->host() || !outputs[0]->host()) {
            return NO_ERROR;
        }
        if (!needTranspose() || inputs[0]->dimensions() != 4) {
            const size_t bytes = std::min(inputs[0]->bytes(), outputs[0]->bytes());
            if (bytes > 0) {
                std::memcpy(outputs[0]->host(), inputs[0]->host(), bytes);
            }
            return NO_ERROR;
        }
        if (isNHWC(mSrcFormat)) {
            // [N, H, W, C] -> [N, C, H, W]
            const Tensor* input = inputs[0];
            const int batch = input->length(0), height = input->length(1), width = input->length(2),
                      channel = input->length(3);
            const float* src = input->host<float>();
            float* dst       = outputs[0]->host<float>();
            cpuPool(backend())->parallelFor(batch * channel, [&](int begin, int end) {
                for (int index = begin; index < end; ++index) {
                    const int n = index / channel;
                    const int c = index % channel;
                    for (int h = 0; h < height; ++h) {
                        for (int w = 0; w < width; ++w) {
                            dst[((static_cast<int64_t>(n) * channel + c) * height + h) * width + w] =
                                src[((static_cast<int64_t>(n) * height + h) * width + w) * channel + c];
                        }
                    }
                }
            }, 1);
        } else {
            // [N, C, H, W] -> [N, H, W, C]
            const Tensor* input = inputs[0];
            const int batch = input->length(0), channel = input->length(1), height = input->length(2),
                      width = input->length(3);
            const float* src = input->host<float>();
            float* dst       = outputs[0]->host<float>();
            cpuPool(backend())->parallelFor(batch * channel, [&](int begin, int end) {
                for (int index = begin; index < end; ++index) {
                    const int n = index / channel;
                    const int c = index % channel;
                    for (int h = 0; h < height; ++h) {
                        for (int w = 0; w < width; ++w) {
                            dst[((static_cast<int64_t>(n) * height + h) * width + w) * channel + c] =
                                src[((static_cast<int64_t>(n) * channel + c) * height + h) * width + w];
                        }
                    }
                }
            }, 1);
        }
        return NO_ERROR;
    }

private:
    OpType mType;
    int mSrcFormat = 0;
    int mDstFormat = 0;
};

// --------------------------------------------------------------------------
// Concat
// --------------------------------------------------------------------------
class CPUConcat : public Execution {
public:
    CPUConcat(CPUBackend* backend, const OpDesc& op) : Execution(backend), mAxis(op.attr.axis) {
    }

    int resolveAxis(int rank) const {
        int axis = mAxis;
        if (axis < 0) {
            axis += rank;
        }
        return axis;
    }

    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.empty() || outputs.empty()) {
            return INVALID_INPUT;
        }
        const int rank = inputs[0]->dimensions();
        const int axis = resolveAxis(rank);
        if (axis < 0 || axis >= rank) {
            return INVALID_INPUT;
        }
        std::vector<int> shape = inputs[0]->shape();
        int total = 0;
        for (const Tensor* input : inputs) {
            total += input->length(axis);
        }
        shape[axis] = total;
        outputs[0]->setShape(shape);
        outputs[0]->setType(inputs[0]->type());
        return NO_ERROR;
    }

    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.empty() || outputs.empty()) {
            return INVALID_INPUT;
        }
        const int rank = outputs[0]->dimensions();
        const int axis = resolveAxis(rank);
        if (axis < 0 || axis >= rank) {
            return INVALID_INPUT;
        }
        const int outer = static_cast<int>(shapeProduct(outputs[0]->shape(), 0, axis));
        const int inner = static_cast<int>(shapeProduct(outputs[0]->shape(), axis + 1));
        const int outAxis = outputs[0]->length(axis);
        float* dst        = outputs[0]->host<float>();
        if (!dst) {
            return INVALID_INPUT;
        }
        cpuPool(backend())->parallelFor(outer, [&](int begin, int end) {
            for (int o = begin; o < end; ++o) {
                int axisOffset = 0;
                for (const Tensor* input : inputs) {
                    const int len = input->length(axis);
                    const float* src = input->host<float>() + static_cast<int64_t>(o) * len * inner;
                    std::memcpy(dst + (static_cast<int64_t>(o) * outAxis + axisOffset) * inner, src,
                                sizeof(float) * len * inner);
                    axisOffset += len;
                }
            }
        }, 1);
        return NO_ERROR;
    }

private:
    int mAxis;
};

// --------------------------------------------------------------------------
// Reshape / Squeeze / Flatten
// --------------------------------------------------------------------------
class CPUReshape : public Execution {
public:
    CPUReshape(CPUBackend* backend, const OpDesc& op) : Execution(backend), mAttr(op.attr) {
    }

    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.empty() || outputs.empty()) {
            return INVALID_INPUT;
        }
        std::vector<int> target;
        if (inputs.size() >= 2 && inputs[1]->host()) {
            const int count = static_cast<int>(inputs[1]->elementCount());
            if (inputs[1]->type() == DT_INT64) {
                const int64_t* src = inputs[1]->host<int64_t>();
                target.assign(src, src + count);
            } else {
                const int32_t* src = inputs[1]->host<int32_t>();
                target.assign(src, src + count);
            }
        } else {
            target = mAttr.dims;
        }
        if (target.empty()) {
            // No target given: fall back to the shape from the model description.
            if (!outputs[0]->shape().empty()) {
                return NO_ERROR;
            }
            return INVALID_INPUT;
        }
        std::vector<int> shape(target.size(), 0);
        int64_t known   = 1;
        int inferIndex  = -1;
        const std::vector<int>& inputShape = inputs[0]->shape();
        for (size_t i = 0; i < target.size(); ++i) {
            if (target[i] == -1) {
                inferIndex = static_cast<int>(i);
                shape[i]   = 1;
                continue;
            }
            if (target[i] == 0) {
                shape[i] = i < inputShape.size() ? inputShape[i] : 1;
            } else {
                shape[i] = target[i];
            }
            known *= shape[i];
        }
        const int64_t total = static_cast<int64_t>(inputs[0]->elementCount());
        if (inferIndex >= 0) {
            if (known == 0) {
                return INVALID_INPUT;
            }
            shape[inferIndex] = static_cast<int>(total / known);
        }
        outputs[0]->setShape(shape);
        outputs[0]->setType(inputs[0]->type());
        return NO_ERROR;
    }

    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.empty() || outputs.empty()) {
            return INVALID_INPUT;
        }
        const size_t bytes = std::min(inputs[0]->bytes(), outputs[0]->bytes());
        if (bytes > 0 && inputs[0]->host() && outputs[0]->host()) {
            std::memcpy(outputs[0]->host(), inputs[0]->host(), bytes);
        }
        return NO_ERROR;
    }

private:
    OpAttr mAttr;
};

class CPUSqueeze : public Execution {
public:
    CPUSqueeze(CPUBackend* backend, const OpDesc& op) : Execution(backend), mDims(op.attr.squeezeDims) {
    }

    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.empty() || outputs.empty()) {
            return INVALID_INPUT;
        }
        const Tensor* input = inputs[0];
        const int rank      = input->dimensions();
        std::vector<bool> mask(rank, false);
        // A dimension can only be removed when its extent is 1.
        auto buildMask = [&](const std::vector<int>& dims, std::vector<bool>& out) -> bool {
            out.assign(rank, false);
            for (int dim : dims) {
                int axis = dim < 0 ? dim + rank : dim;
                if (axis < 0 || axis >= rank || input->length(axis) != 1) {
                    return false;
                }
                out[axis] = true;
            }
            return true;
        };
        bool resolved = false;
        if (!mDims.empty()) {
            resolved = buildMask(mDims, mask);
            if (!resolved && rank == 4) {
                // Models converted from NHWC keep the squeeze axes in NHWC
                // order: H -> 2, W -> 3, C -> 1.
                std::vector<int> mapped;
                mapped.reserve(mDims.size());
                for (int dim : mDims) {
                    switch (dim) {
                        case 1:
                            mapped.push_back(2);
                            break;
                        case 2:
                            mapped.push_back(3);
                            break;
                        case 3:
                            mapped.push_back(1);
                            break;
                        default:
                            mapped.push_back(dim);
                            break;
                    }
                }
                resolved = buildMask(mapped, mask);
            }
        }
        if (!resolved) {
            mask.assign(rank, false);
            for (int i = 0; i < rank; ++i) {
                if (input->length(i) == 1) {
                    mask[i] = true;
                }
            }
        }
        std::vector<int> shape;
        for (int i = 0; i < rank; ++i) {
            if (!mask[i]) {
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
        if (inputs.empty() || outputs.empty()) {
            return INVALID_INPUT;
        }
        const size_t bytes = std::min(inputs[0]->bytes(), outputs[0]->bytes());
        if (bytes > 0 && inputs[0]->host() && outputs[0]->host()) {
            std::memcpy(outputs[0]->host(), inputs[0]->host(), bytes);
        }
        return NO_ERROR;
    }

private:
    std::vector<int> mDims;
};

class CPUFlatten : public Execution {
public:
    CPUFlatten(CPUBackend* backend, const OpDesc& op) : Execution(backend), mAxis(op.attr.axis), mEndAxis(op.attr.axis2) {
    }

    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.empty() || outputs.empty()) {
            return INVALID_INPUT;
        }
        const Tensor* input = inputs[0];
        const int rank      = input->dimensions();
        int axis = mAxis;
        int endAxis = mEndAxis;
        if (axis < 0) {
            axis += rank;
        }
        if (endAxis < 0) {
            endAxis += rank;
        }
        axis    = clampInt(axis, 0, rank - 1);
        endAxis = clampInt(endAxis, 0, rank - 1);
        if (endAxis < axis) {
            std::swap(axis, endAxis);
        }
        std::vector<int> shape;
        if (endAxis == 0) {
            shape = {static_cast<int>(shapeProduct(input->shape(), 0, axis)),
                     static_cast<int>(shapeProduct(input->shape(), axis, rank))};
        } else {
            for (int i = 0; i < axis; ++i) {
                shape.push_back(input->length(i));
            }
            shape.push_back(static_cast<int>(shapeProduct(input->shape(), axis, endAxis + 1)));
            for (int i = endAxis + 1; i < rank; ++i) {
                shape.push_back(input->length(i));
            }
        }
        outputs[0]->setShape(shape);
        outputs[0]->setType(input->type());
        return NO_ERROR;
    }

    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.empty() || outputs.empty()) {
            return INVALID_INPUT;
        }
        const size_t bytes = std::min(inputs[0]->bytes(), outputs[0]->bytes());
        if (bytes > 0 && inputs[0]->host() && outputs[0]->host()) {
            std::memcpy(outputs[0]->host(), inputs[0]->host(), bytes);
        }
        return NO_ERROR;
    }

private:
    int mAxis    = 0;
    int mEndAxis = -1;
};

// --------------------------------------------------------------------------
// Permute
// --------------------------------------------------------------------------
class CPUPermute : public Execution {
public:
    CPUPermute(CPUBackend* backend, const OpDesc& op) : Execution(backend), mDims(op.attr.permuteDims) {
    }

    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.empty() || outputs.empty()) {
            return INVALID_INPUT;
        }
        const Tensor* input = inputs[0];
        const int rank      = input->dimensions();
        std::vector<int> order = mDims;
        if (static_cast<int>(order.size()) != rank) {
            order.resize(rank);
            for (int i = 0; i < rank; ++i) {
                order[i] = rank - 1 - i;
            }
        }
        std::vector<int> shape(rank, 1);
        for (int i = 0; i < rank; ++i) {
            int src = order[i];
            if (src < 0) {
                src += rank;
            }
            if (src < 0 || src >= rank) {
                return INVALID_INPUT;
            }
            shape[i] = input->length(src);
        }
        mOrder = order;
        outputs[0]->setShape(shape);
        outputs[0]->setType(input->type());
        return NO_ERROR;
    }

    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.empty() || outputs.empty()) {
            return INVALID_INPUT;
        }
        const Tensor* input = inputs[0];
        const std::vector<int>& outShape = outputs[0]->shape();
        const int rank = static_cast<int>(outShape.size());
        const int total = static_cast<int>(outputs[0]->elementCount());
        const std::vector<int> inStrides = computeStrides(input->shape());
        const float* src = input->host<float>();
        float* dst       = outputs[0]->host<float>();
        if (!src || !dst) {
            return INVALID_INPUT;
        }
        const std::vector<int> order = mOrder;
        cpuPool(backend())->parallelFor(total, [&](int begin, int end) {
            for (int index = begin; index < end; ++index) {
                int rest = index;
                int64_t offset = 0;
                for (int d = rank - 1; d >= 0; --d) {
                    const int c = rest % outShape[d];
                    rest /= outShape[d];
                    int srcAxis = order[d];
                    if (srcAxis < 0) {
                        srcAxis += rank;
                    }
                    offset += static_cast<int64_t>(c) * inStrides[srcAxis];
                }
                dst[index] = src[offset];
            }
        }, 256);
        return NO_ERROR;
    }

private:
    std::vector<int> mDims;
    std::vector<int> mOrder;
};

// --------------------------------------------------------------------------
// Shape
// --------------------------------------------------------------------------
class CPUShape : public Execution {
public:
    CPUShape(CPUBackend* backend, const OpDesc& op)
        : Execution(backend), mHasStart(op.attr.hasStart), mHasEnd(op.attr.hasEnd), mStart(op.attr.start), mEnd(op.attr.end) {
    }

    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.empty() || outputs.empty()) {
            return INVALID_INPUT;
        }
        const int rank = inputs[0]->dimensions();
        int start = mHasStart ? mStart : 0;
        int end   = mHasEnd ? mEnd : rank;
        if (start < 0) {
            start += rank;
        }
        if (end < 0) {
            end += rank;
        }
        start = clampInt(start, 0, rank);
        end   = clampInt(end, start, rank);
        mStart = start;
        mEnd   = end;
        outputs[0]->setShape({end - start});
        outputs[0]->setType(DT_INT32);
        return NO_ERROR;
    }

    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        int32_t* dst = outputs[0]->host<int32_t>();
        if (!dst) {
            return INVALID_INPUT;
        }
        for (int i = mStart; i < mEnd; ++i) {
            dst[i - mStart] = inputs[0]->length(i);
        }
        return NO_ERROR;
    }

private:
    bool mHasStart;
    bool mHasEnd;
    int mStart;
    int mEnd;
};

// --------------------------------------------------------------------------
// Cast
// --------------------------------------------------------------------------
class CPUCast : public Execution {
public:
    CPUCast(CPUBackend* backend, const OpDesc& op) : Execution(backend), mDstType(op.attr.dataType) {
    }

    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.empty() || outputs.empty()) {
            return INVALID_INPUT;
        }
        outputs[0]->setShape(inputs[0]->shape());
        outputs[0]->setType(mDstType);
        return NO_ERROR;
    }

    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.empty() || outputs.empty()) {
            return INVALID_INPUT;
        }
        const size_t total = inputs[0]->elementCount();
        const DataType srcType = inputs[0]->type();
        const DataType dstType = mDstType;
        auto read = [&](size_t i) -> float {
            switch (srcType) {
                case DT_FLOAT:
                    return inputs[0]->host<float>()[i];
                case DT_INT32:
                    return static_cast<float>(inputs[0]->host<int32_t>()[i]);
                case DT_INT64:
                    return static_cast<float>(inputs[0]->host<int64_t>()[i]);
                case DT_INT8:
                    return static_cast<float>(inputs[0]->host<int8_t>()[i]);
                case DT_UINT8:
                    return static_cast<float>(inputs[0]->host<uint8_t>()[i]);
                default:
                    return 0.0f;
            }
        };
        for (size_t i = 0; i < total; ++i) {
            const float v = read(i);
            switch (dstType) {
                case DT_FLOAT:
                    outputs[0]->host<float>()[i] = v;
                    break;
                case DT_INT32:
                    outputs[0]->host<int32_t>()[i] = static_cast<int32_t>(v);
                    break;
                case DT_INT64:
                    outputs[0]->host<int64_t>()[i] = static_cast<int64_t>(v);
                    break;
                case DT_INT8:
                    outputs[0]->host<int8_t>()[i] = static_cast<int8_t>(clampInt(static_cast<int>(v), -128, 127));
                    break;
                case DT_UINT8:
                    outputs[0]->host<uint8_t>()[i] = static_cast<uint8_t>(clampInt(static_cast<int>(v), 0, 255));
                    break;
                default:
                    return NOT_SUPPORT;
            }
        }
        return NO_ERROR;
    }

private:
    DataType mDstType;
};

// --------------------------------------------------------------------------
// StridedSlice
// --------------------------------------------------------------------------
struct SliceSpec {
    int begin      = 0;
    int end        = 0;
    int stride     = 1;
    bool beginMask = false;
    bool endMask   = false;
    bool newAxis   = false;
    bool shrink    = false;
    bool ellipsis  = false;
};

class CPUStridedSlice : public Execution {
public:
    CPUStridedSlice(CPUBackend* backend, const OpDesc& op) : Execution(backend), mAttr(op.attr) {
    }

    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.size() < 3 || outputs.empty()) {
            return INVALID_INPUT;
        }
        const Tensor* input = inputs[0];
        const int rank      = input->dimensions();
        std::vector<SliceSpec> specs = readSpecs(inputs);
        if (specs.empty()) {
            return INVALID_INPUT;
        }
        // Expand ellipsis / trailing full slices.
        {
            int consumed = 0;
            int ellipsisIndex = -1;
            for (size_t i = 0; i < specs.size(); ++i) {
                if (specs[i].ellipsis) {
                    ellipsisIndex = static_cast<int>(i);
                } else if (!specs[i].newAxis) {
                    ++consumed;
                }
            }
            if (ellipsisIndex >= 0) {
                const int missing = rank - consumed;
                std::vector<SliceSpec> expanded;
                expanded.reserve(specs.size() + std::max(0, missing));
                for (size_t i = 0; i < specs.size(); ++i) {
                    if (static_cast<int>(i) == ellipsisIndex) {
                        for (int k = 0; k < missing; ++k) {
                            SliceSpec full;
                            full.beginMask = true;
                            full.endMask   = true;
                            expanded.push_back(full);
                        }
                    } else {
                        expanded.push_back(specs[i]);
                    }
                }
                specs.swap(expanded);
            }
            int nonNewAxis = 0;
            for (const SliceSpec& spec : specs) {
                if (!spec.newAxis) {
                    ++nonNewAxis;
                }
            }
            while (nonNewAxis < rank) {
                SliceSpec full;
                full.beginMask = true;
                full.endMask   = true;
                specs.push_back(full);
                ++nonNewAxis;
            }
        }

        mOutAxis.clear();
        mOutStart.clear();
        mOutStep.clear();
        mShrinkAxis.clear();
        mShrinkStart.clear();
        std::vector<int> shape;
        int inputAxis = 0;
        for (const SliceSpec& spec : specs) {
            if (spec.newAxis) {
                shape.push_back(1);
                mOutAxis.push_back(-1);
                mOutStart.push_back(0);
                mOutStep.push_back(1);
                continue;
            }
            if (inputAxis >= rank) {
                break;
            }
            const int extent = input->length(inputAxis);
            int begin = spec.begin;
            int end   = spec.end;
            int step  = spec.stride == 0 ? 1 : spec.stride;
            if (spec.beginMask) {
                begin = step > 0 ? 0 : extent - 1;
            }
            if (spec.endMask) {
                end = step > 0 ? extent : -extent - 1;
            }
            if (spec.shrink) {
                if (begin < 0) {
                    begin += extent;
                }
                begin = clampInt(begin, 0, extent - 1);
                mShrinkAxis.push_back(inputAxis);
                mShrinkStart.push_back(begin);
                ++inputAxis;
                continue;
            }
            if (step > 0) {
                if (begin < 0) {
                    begin += extent;
                }
                if (end < 0) {
                    end += extent;
                }
                begin = clampInt(begin, 0, extent);
                end   = clampInt(end, 0, extent);
                const int count = begin < end ? (end - begin + step - 1) / step : 0;
                shape.push_back(count);
            } else {
                if (begin < 0) {
                    begin += extent;
                }
                if (end < 0 && end != -extent - 1) {
                    end += extent;
                }
                begin = clampInt(begin, -1, extent - 1);
                end   = clampInt(end, -1, extent - 1);
                const int count = begin > end ? (begin - end + (-step) - 1) / (-step) : 0;
                shape.push_back(count);
            }
            mOutAxis.push_back(inputAxis);
            mOutStart.push_back(begin);
            mOutStep.push_back(step);
            ++inputAxis;
        }
        outputs[0]->setShape(shape);
        outputs[0]->setType(input->type());
        return NO_ERROR;
    }

    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.empty() || outputs.empty()) {
            return INVALID_INPUT;
        }
        const Tensor* input = inputs[0];
        const float* src    = input->host<float>();
        float* dst          = outputs[0]->host<float>();
        if (!src || !dst) {
            return INVALID_INPUT;
        }
        const std::vector<int>& outShape = outputs[0]->shape();
        const int outRank = static_cast<int>(outShape.size());
        const int total   = static_cast<int>(outputs[0]->elementCount());
        const std::vector<int> inStrides = computeStrides(input->shape());
        const std::vector<int> outAxis  = mOutAxis;
        const std::vector<int> outStart = mOutStart;
        const std::vector<int> outStep  = mOutStep;
        int64_t shrinkOffset = 0;
        for (size_t i = 0; i < mShrinkAxis.size(); ++i) {
            shrinkOffset += static_cast<int64_t>(mShrinkStart[i]) * inStrides[mShrinkAxis[i]];
        }
        cpuPool(backend())->parallelFor(total, [&](int begin, int end) {
            for (int index = begin; index < end; ++index) {
                int rest = index;
                int64_t offset = shrinkOffset;
                for (int d = outRank - 1; d >= 0; --d) {
                    const int c = rest % outShape[d];
                    rest /= outShape[d];
                    if (outAxis[d] >= 0) {
                        offset += static_cast<int64_t>(outStart[d] + c * outStep[d]) * inStrides[outAxis[d]];
                    }
                }
                dst[index] = src[offset];
            }
        }, 256);
        return NO_ERROR;
    }

private:
    std::vector<SliceSpec> readSpecs(const std::vector<Tensor*>& inputs) const {
        std::vector<SliceSpec> specs;
        const Tensor* beginTensor = inputs[1];
        const Tensor* endTensor   = inputs[2];
        const Tensor* strideTensor = inputs.size() > 3 ? inputs[3] : nullptr;
        if (!beginTensor || !endTensor) {
            return specs;
        }
        const int count = static_cast<int>(beginTensor->elementCount());
        for (int i = 0; i < count; ++i) {
            SliceSpec spec;
            spec.begin = readInt(beginTensor, i);
            spec.end   = readInt(endTensor, i);
            if (strideTensor) {
                spec.stride = readInt(strideTensor, i);
            }
            spec.beginMask = (mAttr.beginMask >> i) & 1;
            spec.endMask   = (mAttr.endMask >> i) & 1;
            spec.ellipsis  = (mAttr.ellipsisMask >> i) & 1;
            spec.newAxis   = (mAttr.newAxisMask >> i) & 1;
            spec.shrink    = (mAttr.shrinkAxisMask >> i) & 1;
            specs.push_back(spec);
        }
        return specs;
    }

    static int readInt(const Tensor* tensor, int index) {
        if (tensor->type() == DT_INT64) {
            return static_cast<int>(tensor->host<int64_t>()[index]);
        }
        if (tensor->type() == DT_INT32) {
            return tensor->host<int32_t>()[index];
        }
        return 0;
    }

    OpAttr mAttr;
    std::vector<int> mOutAxis, mOutStart, mOutStep;
    std::vector<int> mShrinkAxis, mShrinkStart;
};

// --------------------------------------------------------------------------
// GatherV2
// --------------------------------------------------------------------------
class CPUGatherV2 : public Execution {
public:
    CPUGatherV2(CPUBackend* backend, const OpDesc& op) : Execution(backend), mAttr(op.attr) {
    }

    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.size() < 2 || outputs.empty()) {
            return INVALID_INPUT;
        }
        mAxis = inputs.size() > 2 && inputs[2]->host() ? readInt(inputs[2], 0) : mAttr.axis;
        const int rank = inputs[0]->dimensions();
        if (mAxis < 0) {
            mAxis += rank;
        }
        if (mAxis < 0 || mAxis >= rank) {
            return INVALID_INPUT;
        }
        std::vector<int> shape;
        for (int i = 0; i < mAxis; ++i) {
            shape.push_back(inputs[0]->length(i));
        }
        for (int dim : inputs[1]->shape()) {
            shape.push_back(dim);
        }
        for (int i = mAxis + 1; i < rank; ++i) {
            shape.push_back(inputs[0]->length(i));
        }
        outputs[0]->setShape(shape);
        outputs[0]->setType(inputs[0]->type());
        return NO_ERROR;
    }

    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.size() < 2 || outputs.empty()) {
            return INVALID_INPUT;
        }
        const Tensor* params  = inputs[0];
        const Tensor* indices = inputs[1];
        const float* src      = params->host<float>();
        float* dst            = outputs[0]->host<float>();
        if (!src || !dst) {
            return INVALID_INPUT;
        }
        const int outer       = static_cast<int>(shapeProduct(params->shape(), 0, mAxis));
        const int inner       = static_cast<int>(shapeProduct(params->shape(), mAxis + 1));
        const int axisExtent  = params->length(mAxis);
        const int indexCount  = static_cast<int>(indices->elementCount());
        auto task = [&](int begin, int end) {
            for (int i = begin; i < end; ++i) {
                const int o = i / indexCount;
                const int k = i % indexCount;
                int index   = readInt(indices, k);
                if (index < 0) {
                    index += axisExtent;
                }
                index = clampInt(index, 0, axisExtent - 1);
                std::memcpy(dst + (static_cast<int64_t>(o) * indexCount + k) * inner,
                            src + (static_cast<int64_t>(o) * axisExtent + index) * inner,
                            sizeof(float) * inner);
            }
        };
        cpuPool(backend())->parallelFor(outer * indexCount, task, 1);
        return NO_ERROR;
    }

private:
    static int readInt(const Tensor* tensor, int index) {
        if (tensor->type() == DT_INT64) {
            return static_cast<int>(tensor->host<int64_t>()[index]);
        }
        return tensor->host<int32_t>()[index];
    }

    OpAttr mAttr;
    int mAxis = 0;
};

// --------------------------------------------------------------------------
// Pack
// --------------------------------------------------------------------------
class CPUPack : public Execution {
public:
    CPUPack(CPUBackend* backend, const OpDesc& op) : Execution(backend), mAxis(op.attr.axis) {
    }

    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.empty() || outputs.empty()) {
            return INVALID_INPUT;
        }
        const int rank = inputs[0]->dimensions() + 1;
        int axis       = mAxis;
        mValueCount    = static_cast<int>(inputs.size());
        if (axis < 0) {
            axis += rank;
        }
        if (axis < 0 || axis >= rank) {
            return INVALID_INPUT;
        }
        mAxis = axis;
        std::vector<int> shape;
        for (int i = 0; i < axis; ++i) {
            shape.push_back(inputs[0]->length(i));
        }
        shape.push_back(mValueCount);
        for (int i = axis; i < inputs[0]->dimensions(); ++i) {
            shape.push_back(inputs[0]->length(i));
        }
        outputs[0]->setShape(shape);
        outputs[0]->setType(inputs[0]->type());
        return NO_ERROR;
    }

    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.empty() || outputs.empty()) {
            return INVALID_INPUT;
        }
        void* dst = outputs[0]->host();
        if (!dst) {
            return INVALID_INPUT;
        }
        const int elementSize = static_cast<int>(outputs[0]->bytes()) / std::max(1, static_cast<int>(outputs[0]->elementCount()));
        const int outer = static_cast<int>(shapeProduct(outputs[0]->shape(), 0, mAxis));
        const int inner = static_cast<int>(shapeProduct(outputs[0]->shape(), mAxis + 1));
        const int count = mValueCount;
        cpuPool(backend())->parallelFor(outer, [&](int begin, int end) {
            for (int o = begin; o < end; ++o) {
                for (int k = 0; k < count; ++k) {
                    const char* src = static_cast<const char*>(inputs[k]->host());
                    if (!src) {
                        continue;
                    }
                    std::memcpy(static_cast<char*>(dst) + (static_cast<int64_t>(o) * count + k) * inner * elementSize,
                                src + static_cast<int64_t>(o) * inner * elementSize,
                                static_cast<size_t>(inner) * elementSize);
                }
            }
        }, 1);
        return NO_ERROR;
    }

private:
    int mAxis       = 0;
    int mValueCount = 1;
};

// --------------------------------------------------------------------------
// ArgMax / ArgMin
// --------------------------------------------------------------------------
class CPUArgMax : public Execution {
public:
    CPUArgMax(CPUBackend* backend, const OpDesc& op) : Execution(backend), mAxis(op.attr.axis), mIsMax(op.type == OpType_ArgMax) {
    }

    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.empty() || outputs.empty()) {
            return INVALID_INPUT;
        }
        const int rank = inputs[0]->dimensions();
        int axis = mAxis;
        if (axis < 0) {
            axis += rank;
        }
        if (axis < 0 || axis >= rank) {
            return INVALID_INPUT;
        }
        mAxis = axis;
        std::vector<int> shape;
        for (int i = 0; i < rank; ++i) {
            if (i != axis) {
                shape.push_back(inputs[0]->length(i));
            }
        }
        if (shape.empty()) {
            shape.push_back(1);
        }
        outputs[0]->setShape(shape);
        outputs[0]->setType(DT_INT32);
        return NO_ERROR;
    }

    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.empty() || outputs.empty()) {
            return INVALID_INPUT;
        }
        const Tensor* input = inputs[0];
        const float* src    = input->host<float>();
        int32_t* dst        = outputs[0]->host<int32_t>();
        if (!src || !dst) {
            return INVALID_INPUT;
        }
        const int outer = static_cast<int>(shapeProduct(input->shape(), 0, mAxis));
        const int dim   = input->length(mAxis);
        const int inner = static_cast<int>(shapeProduct(input->shape(), mAxis + 1));
        const bool isMax = mIsMax;
        cpuPool(backend())->parallelFor(outer * inner, [&](int begin, int end) {
            for (int index = begin; index < end; ++index) {
                const int o = index / inner;
                const int i = index % inner;
                const float* row = src + static_cast<int64_t>(o) * dim * inner + i;
                int best         = 0;
                float bestValue  = row[0];
                for (int k = 1; k < dim; ++k) {
                    const float v = row[static_cast<int64_t>(k) * inner];
                    if (isMax ? (v > bestValue) : (v < bestValue)) {
                        bestValue = v;
                        best      = k;
                    }
                }
                dst[index] = best;
            }
        }, 64);
        return NO_ERROR;
    }

private:
    int mAxis = 0;
    bool mIsMax = true;
};

// --------------------------------------------------------------------------
// Transpose (permutation is supplied by the second input tensor)
// --------------------------------------------------------------------------
class CPUTranspose : public Execution {
public:
    CPUTranspose(CPUBackend* backend, const OpDesc& op) : Execution(backend) {
    }

    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.size() < 2 || outputs.empty()) {
            return INVALID_INPUT;
        }
        const Tensor* input = inputs[0];
        const int rank      = input->dimensions();
        if (!inputs[1]->host() || static_cast<int>(inputs[1]->elementCount()) != rank) {
            return INVALID_INPUT;
        }
        const int32_t* perm = inputs[1]->host<int32_t>();
        std::vector<char> used(rank, 0);
        std::vector<int> shape(rank);
        mPerm.resize(rank);
        for (int i = 0; i < rank; ++i) {
            const int d = perm[i];
            if (d < 0 || d >= rank || used[d]) {
                return INVALID_INPUT;
            }
            used[d] = 1;
            mPerm[i] = d;
            shape[i] = input->length(d);
        }
        outputs[0]->setShape(shape);
        outputs[0]->setType(input->type());
        return NO_ERROR;
    }

    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.empty() || outputs.empty() || mPerm.empty()) {
            return NO_ERROR;
        }
        const Tensor* input  = inputs[0];
        const Tensor* output = outputs[0];
        const uint32_t* src  = input->host<uint32_t>();
        uint32_t* dst        = output->host<uint32_t>();
        if (!src || !dst) {
            return INVALID_INPUT;
        }
        const int rank = input->dimensions();
        std::vector<int64_t> inStrides(rank, 1);
        for (int d = rank - 2; d >= 0; --d) {
            inStrides[d] = inStrides[d + 1] * input->length(d + 1);
        }
        const std::vector<int>& outShape = output->shape();
        const int total = static_cast<int>(output->elementCount());
        auto task = [&](int begin, int end) {
            for (int index = begin; index < end; ++index) {
                int rest          = index;
                int64_t srcOffset = 0;
                for (int d = rank - 1; d >= 0; --d) {
                    const int coord = rest % outShape[d];
                    rest /= outShape[d];
                    srcOffset += static_cast<int64_t>(coord) * inStrides[mPerm[d]];
                }
                dst[index] = src[srcOffset];
            }
        };
        cpuPool(backend())->parallelFor(total, task, 256);
        return NO_ERROR;
    }

private:
    std::vector<int> mPerm;
};

// --------------------------------------------------------------------------
// Unpack: split the input along one axis into several outputs
// --------------------------------------------------------------------------
class CPUUnpack : public Execution {
public:
    CPUUnpack(CPUBackend* backend, const OpDesc& op) : Execution(backend), mAxis(op.attr.axis) {
    }

    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.empty() || outputs.empty()) {
            return INVALID_INPUT;
        }
        const Tensor* input = inputs[0];
        const int rank      = input->dimensions();
        int axis            = mAxis < 0 ? mAxis + rank : mAxis;
        if (axis < 0 || axis >= rank) {
            return INVALID_INPUT;
        }
        mAxis = axis;
        std::vector<int> shape;
        shape.reserve(rank - 1);
        for (int d = 0; d < rank; ++d) {
            if (d != axis) {
                shape.push_back(input->length(d));
            }
        }
        for (Tensor* output : outputs) {
            output->setShape(shape);
            output->setType(input->type());
        }
        return NO_ERROR;
    }

    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.empty() || outputs.empty()) {
            return NO_ERROR;
        }
        const Tensor* input = inputs[0];
        const uint32_t* src = input->host<uint32_t>();
        if (!src) {
            return INVALID_INPUT;
        }
        const int rank = input->dimensions();
        int64_t outer  = 1;
        int64_t inner  = 1;
        for (int d = 0; d < mAxis; ++d) {
            outer *= input->length(d);
        }
        for (int d = mAxis + 1; d < rank; ++d) {
            inner *= input->length(d);
        }
        const int axisLen = input->length(mAxis);
        auto task = [&](int begin, int end) {
            for (int o = begin; o < end; ++o) {
                uint32_t* dst = outputs[o]->host<uint32_t>();
                if (!dst) {
                    continue;
                }
                for (int64_t i = 0; i < outer; ++i) {
                    std::memcpy(dst + i * inner, src + (i * axisLen + o) * inner,
                                sizeof(uint32_t) * inner);
                }
            }
        };
        cpuPool(backend())->parallelFor(static_cast<int>(outputs.size()), task, 1);
        return NO_ERROR;
    }

private:
    int mAxis = 0;
};

// --------------------------------------------------------------------------
// SliceTf: begin / size come from the second and third input tensors
// --------------------------------------------------------------------------
class CPUSliceTf : public Execution {
public:
    CPUSliceTf(CPUBackend* backend, const OpDesc& op) : Execution(backend) {
    }

    static bool readInts(const Tensor* tensor, std::vector<int>& out) {
        if (!tensor->host()) {
            return false;
        }
        const int count = static_cast<int>(tensor->elementCount());
        out.resize(count);
        if (tensor->type() == DT_INT64) {
            const int64_t* src = tensor->host<int64_t>();
            for (int i = 0; i < count; ++i) {
                out[i] = static_cast<int>(src[i]);
            }
        } else {
            const int32_t* src = tensor->host<int32_t>();
            for (int i = 0; i < count; ++i) {
                out[i] = src[i];
            }
        }
        return true;
    }

    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.size() < 3 || outputs.empty()) {
            return INVALID_INPUT;
        }
        const Tensor* input = inputs[0];
        const int rank      = input->dimensions();
        std::vector<int> begin;
        std::vector<int> size;
        if (!readInts(inputs[1], begin) || !readInts(inputs[2], size)) {
            return INVALID_INPUT;
        }
        if (begin.size() != size.size() || static_cast<int>(begin.size()) > rank) {
            return INVALID_INPUT;
        }
        mBegin.assign(rank, 0);
        std::vector<int> shape(rank);
        for (int d = 0; d < rank; ++d) {
            shape[d] = input->length(d);
        }
        for (size_t i = 0; i < begin.size(); ++i) {
            int start = begin[i];
            if (start < 0) {
                start += shape[i];
            }
            int count = size[i];
            if (count == -1) {
                count = shape[i] - start;
            }
            if (start < 0 || count < 0 || start + count > shape[i]) {
                return INVALID_INPUT;
            }
            mBegin[i] = start;
            shape[i]  = count;
        }
        outputs[0]->setShape(shape);
        outputs[0]->setType(input->type());
        return NO_ERROR;
    }

    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.empty() || outputs.empty()) {
            return NO_ERROR;
        }
        const Tensor* input  = inputs[0];
        const Tensor* output = outputs[0];
        const uint32_t* src  = input->host<uint32_t>();
        uint32_t* dst        = output->host<uint32_t>();
        if (!src || !dst) {
            return INVALID_INPUT;
        }
        const int rank = input->dimensions();
        std::vector<int64_t> inStrides(rank, 1);
        for (int d = rank - 2; d >= 0; --d) {
            inStrides[d] = inStrides[d + 1] * input->length(d + 1);
        }
        const std::vector<int>& outShape = output->shape();
        const int total                  = static_cast<int>(output->elementCount());
        const std::vector<int> begin     = mBegin;
        auto task = [&](int beginIndex, int end) {
            for (int index = beginIndex; index < end; ++index) {
                int rest          = index;
                int64_t srcOffset = 0;
                for (int d = rank - 1; d >= 0; --d) {
                    const int coord = rest % outShape[d];
                    rest /= outShape[d];
                    srcOffset += static_cast<int64_t>(begin[d] + coord) * inStrides[d];
                }
                dst[index] = src[srcOffset];
            }
        };
        cpuPool(backend())->parallelFor(total, task, 256);
        return NO_ERROR;
    }

private:
    std::vector<int> mBegin;
};

} // namespace

std::shared_ptr<Execution> createCPUShapeOp(CPUBackend* backend, const OpDesc& op,
                                            const std::vector<Tensor*>&,
                                            const std::vector<Tensor*>&) {
    switch (op.type) {
        case OpType_Const:
        case OpType_Input:
        case OpType_Identity:
        case OpType_ConvertTensor:
            return std::shared_ptr<Execution>(new CPUNoop(backend, op));
        case OpType_Concat:
            return std::shared_ptr<Execution>(new CPUConcat(backend, op));
        case OpType_Reshape:
            return std::shared_ptr<Execution>(new CPUReshape(backend, op));
        case OpType_Squeeze:
            return std::shared_ptr<Execution>(new CPUSqueeze(backend, op));
        case OpType_Flatten:
            return std::shared_ptr<Execution>(new CPUFlatten(backend, op));
        case OpType_Permute:
            return std::shared_ptr<Execution>(new CPUPermute(backend, op));
        case OpType_Shape:
            return std::shared_ptr<Execution>(new CPUShape(backend, op));
        case OpType_Cast:
            return std::shared_ptr<Execution>(new CPUCast(backend, op));
        case OpType_StridedSlice:
            return std::shared_ptr<Execution>(new CPUStridedSlice(backend, op));
        case OpType_GatherV2:
            return std::shared_ptr<Execution>(new CPUGatherV2(backend, op));
        case OpType_Pack:
            return std::shared_ptr<Execution>(new CPUPack(backend, op));
        case OpType_ArgMax:
        case OpType_ArgMin:
            return std::shared_ptr<Execution>(new CPUArgMax(backend, op));
        case OpType_Transpose:
            return std::shared_ptr<Execution>(new CPUTranspose(backend, op));
        case OpType_Unpack:
            return std::shared_ptr<Execution>(new CPUUnpack(backend, op));
        case OpType_SliceTf:
            return std::shared_ptr<Execution>(new CPUSliceTf(backend, op));
        default:
            return nullptr;
    }
}

} // namespace infer
