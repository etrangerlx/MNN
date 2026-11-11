#pragma once

#include <memory>

#include "infer/Model.h"
#include "infer/Types.h"

namespace infer {

// Parses an MNN 3.x flatbuffer model into the backend independent Model.
// The raw buffer is copied and kept alive by the returned model.
ErrorCode loadModelFromBuffer(const void* buffer, size_t size, std::shared_ptr<Model>* model);

} // namespace infer
