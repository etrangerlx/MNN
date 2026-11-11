#pragma once

#include <algorithm>
#include <cstring>
#include <numeric>
#include <vector>

#include "CPUBackend.h"
#include "infer/Tensor.h"

namespace infer {

inline std::vector<int> computeStrides(const std::vector<int>& shape) {
    std::vector<int> strides(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

inline int64_t shapeProduct(const std::vector<int>& shape, int begin = 0, int end = -1) {
    if (end < 0) {
        end = static_cast<int>(shape.size());
    }
    int64_t product = 1;
    for (int i = begin; i < end && i < static_cast<int>(shape.size()); ++i) {
        product *= shape[i];
    }
    return product;
}

// Strides aligned to the right (numpy broadcasting), zero for broadcast dims.
inline std::vector<int> broadcastStrides(const std::vector<int>& shape, int outRank) {
    std::vector<int> strides(outRank, 0);
    const std::vector<int> own = computeStrides(shape);
    const int offset           = outRank - static_cast<int>(shape.size());
    for (int i = 0; i < static_cast<int>(shape.size()); ++i) {
        const int dim = offset + i;
        if (dim >= 0 && shape[i] != 1) {
            strides[dim] = own[i];
        }
    }
    return strides;
}

inline std::vector<int> broadcastShape(const std::vector<int>& a, const std::vector<int>& b) {
    const int rank = std::max(static_cast<int>(a.size()), static_cast<int>(b.size()));
    std::vector<int> out(rank, 1);
    for (int i = 0; i < rank; ++i) {
        const int dimA = i < rank - static_cast<int>(a.size()) ? 1 : a[i - (rank - static_cast<int>(a.size()))];
        const int dimB = i < rank - static_cast<int>(b.size()) ? 1 : b[i - (rank - static_cast<int>(b.size()))];
        out[i]         = std::max(dimA, dimB);
    }
    return out;
}

inline bool sameShape(const std::vector<int>& a, const std::vector<int>& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

// Convolution/pooling output extent, ceilMode mirrors MNN's pooling behaviour.
inline int computeOutExtent(int input, int kernel, int stride, int dilate, int pad, bool ceilMode) {
    const int effective = dilate * (kernel - 1) + 1;
    if (ceilMode) {
        return (input + 2 * pad - effective + stride - 1) / stride + 1;
    }
    return (input + 2 * pad - effective) / stride + 1;
}

inline void computeConvPads(int input, int kernel, int stride, int dilate, int padMode, int explicitPad,
                            int& padBefore, int& padAfter) {
    padBefore = explicitPad;
    padAfter  = explicitPad;
    if (padMode == PadMode_SAME) {
        const int out    = (input + stride - 1) / stride;
        const int total  = std::max(0, (out - 1) * stride + dilate * (kernel - 1) + 1 - input);
        padBefore        = total / 2;
        padAfter         = total - padBefore;
    } else if (padMode == PadMode_VALID) {
        padBefore = 0;
        padAfter  = 0;
    }
}

} // namespace infer
