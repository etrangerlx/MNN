#pragma once

#include <memory>
#include <string>
#include <vector>

#include "infer/Op.h"
#include "infer/Tensor.h"
#include "infer/Types.h"

namespace infer {

struct TensorDesc {
    std::string name;
    DataType type = DT_FLOAT;
    std::vector<int> dims;
    bool hasData = false;
    std::shared_ptr<std::vector<uint8_t>> data;
};

// A loaded, backend independent graph. The model keeps the raw file buffer
// alive because constant data may point into it.
struct Model {
    std::vector<TensorDesc> tensors;
    std::vector<OpDesc> ops; // topologically sorted
    std::vector<int> inputIndices;
    std::vector<int> outputIndices;
    std::shared_ptr<std::vector<uint8_t>> buffer;
};

} // namespace infer
