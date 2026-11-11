#pragma once

#include <string>
#include <vector>

#include "infer/Types.h"

namespace infer {

// A tensor is a NCHW (row major) view over host memory. The memory itself is
// owned either by the session arena or by the model (constant tensors), so
// Tensor never frees the buffer it points to.
class Tensor {
public:
    Tensor() = default;
    Tensor(const std::string& name, DataType type, const std::vector<int>& shape);

    const std::string& name() const {
        return mName;
    }
    void setName(const std::string& name) {
        mName = name;
    }

    DataType type() const {
        return mType;
    }
    void setType(DataType type) {
        mType = type;
    }

    const std::vector<int>& shape() const {
        return mShape;
    }
    void setShape(const std::vector<int>& shape) {
        mShape = shape;
    }

    int dimensions() const {
        return static_cast<int>(mShape.size());
    }
    int length(int axis) const;
    int batch() const {
        return mShape.empty() ? 1 : mShape[0];
    }
    int channel() const {
        return mShape.size() < 2 ? 1 : mShape[1];
    }
    int height() const {
        return mShape.size() < 3 ? 1 : mShape[2];
    }
    int width() const {
        return mShape.size() < 4 ? 1 : mShape[3];
    }

    size_t elementCount() const;
    size_t bytes() const;
    bool empty() const {
        return mShape.empty() || elementCount() == 0;
    }

    void* host() const {
        return mHost;
    }
    template <typename T>
    T* host() const {
        return reinterpret_cast<T*>(mHost);
    }
    void setHost(void* host) {
        mHost = host;
    }

    bool isConstant() const {
        return mConstant;
    }
    void setConstant(bool c) {
        mConstant = c;
    }

    // Memory planning metadata: [liveBegin, liveEnd] in operator index units.
    int liveBegin() const {
        return mLiveBegin;
    }
    int liveEnd() const {
        return mLiveEnd;
    }
    void setLiveRange(int begin, int end) {
        mLiveBegin = begin;
        mLiveEnd   = end;
    }

    std::string describe() const;

private:
    std::string mName;
    DataType mType     = DT_FLOAT;
    std::vector<int> mShape;
    void* mHost        = nullptr;
    bool mConstant     = false;
    int mLiveBegin     = 0;
    int mLiveEnd       = 0;
};

} // namespace infer
