#include "infer/Tensor.h"

#include <sstream>

namespace infer {

Tensor::Tensor(const std::string& name, DataType type, const std::vector<int>& shape)
    : mName(name), mType(type), mShape(shape) {
}

int Tensor::length(int axis) const {
    const int n = static_cast<int>(mShape.size());
    if (axis < 0) {
        axis += n;
    }
    if (axis < 0 || axis >= n) {
        return 1;
    }
    return mShape[axis];
}

size_t Tensor::elementCount() const {
    size_t count = 1;
    for (int d : mShape) {
        count *= static_cast<size_t>(d < 0 ? 0 : d);
    }
    return count;
}

size_t Tensor::bytes() const {
    return elementCount() * dataTypeBytes(mType);
}

std::string Tensor::describe() const {
    std::ostringstream os;
    os << mName << "(" << dataTypeName(mType) << ")[";
    for (size_t i = 0; i < mShape.size(); ++i) {
        if (i) {
            os << ",";
        }
        os << mShape[i];
    }
    os << "]";
    return os.str();
}

} // namespace infer
