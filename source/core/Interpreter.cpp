#include "infer/Interpreter.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <unordered_map>

#include "infer/ModelLoader.h"

namespace infer {

int Interpreter::defaultThreadNumber() const {
    const unsigned hw = std::thread::hardware_concurrency();
    return hw == 0 ? 1 : static_cast<int>(hw);
}

std::shared_ptr<Interpreter> Interpreter::createFromBuffer(const void* buffer, size_t size, ErrorCode* code) {
    std::shared_ptr<Model> model;
    const ErrorCode result = loadModelFromBuffer(buffer, size, &model);
    if (code) {
        *code = result;
    }
    if (result != NO_ERROR) {
        return nullptr;
    }
    auto interpreter  = std::shared_ptr<Interpreter>(new Interpreter);
    interpreter->mModel = model;
    for (int index : model->inputIndices) {
        if (index >= 0 && index < static_cast<int>(model->tensors.size())) {
            interpreter->mInputNames.push_back(model->tensors[index].name);
        }
    }
    for (int index : model->outputIndices) {
        if (index >= 0 && index < static_cast<int>(model->tensors.size())) {
            interpreter->mOutputNames.push_back(model->tensors[index].name);
        }
    }
    return interpreter;
}

std::shared_ptr<Interpreter> Interpreter::createFromFile(const std::string& path, ErrorCode* code) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        if (code) {
            *code = INVALID_INPUT;
        }
        return nullptr;
    }
    const std::streamsize size = file.tellg();
    if (size <= 0) {
        if (code) {
            *code = INVALID_INPUT;
        }
        return nullptr;
    }
    std::vector<uint8_t> buffer(static_cast<size_t>(size));
    file.seekg(0, std::ios::beg);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        if (code) {
            *code = INVALID_INPUT;
        }
        return nullptr;
    }
    return createFromBuffer(buffer.data(), buffer.size(), code);
}

std::shared_ptr<Interpreter> Interpreter::createFromModel(std::shared_ptr<Model> model) {
    if (!model) {
        return nullptr;
    }
    auto interpreter    = std::shared_ptr<Interpreter>(new Interpreter);
    interpreter->mModel = std::move(model);
    for (int index : interpreter->mModel->inputIndices) {
        interpreter->mInputNames.push_back(interpreter->mModel->tensors[index].name);
    }
    for (int index : interpreter->mModel->outputIndices) {
        interpreter->mOutputNames.push_back(interpreter->mModel->tensors[index].name);
    }
    return interpreter;
}

std::shared_ptr<Session> Interpreter::createSession(int threads) const {
    if (threads <= 0) {
        threads = defaultThreadNumber();
    }
    auto session = std::shared_ptr<Session>(new Session(std::const_pointer_cast<Interpreter>(shared_from_this()), threads));
    const ErrorCode code = session->init();
    if (code != NO_ERROR) {
        fprintf(stderr, "infer: failed to create session (error %d)\n", static_cast<int>(code));
        return nullptr;
    }
    return session;
}

Session::Session(std::shared_ptr<Interpreter> interpreter, int threads)
    : mInterpreter(std::move(interpreter)), mThreads(std::max(1, threads)) {
}

Session::~Session() {
    if (mArena) {
        std::free(mArena);
        mArena = nullptr;
    }
}

int Session::threadNumber() const {
    return mPool ? mPool->threadNumber() : 1;
}

ErrorCode Session::init() {
    const Model* model = mInterpreter->model();
    mPool.reset(new ThreadPool(mThreads - 1));
    mBackend = createCPUBackend(mPool.get());
    auto* backend = mBackend.get();

    mTensors.resize(model->tensors.size());
    for (size_t i = 0; i < model->tensors.size(); ++i) {
        const TensorDesc& desc = model->tensors[i];
        auto tensor          = std::unique_ptr<Tensor>(new Tensor(desc.name, desc.type, desc.dims));
        if (desc.hasData && desc.data) {
            tensor->setHost(desc.data->data());
            tensor->setConstant(true);
        }
        mTensors[i] = std::move(tensor);
    }

    mExecutions.resize(model->ops.size());
    mProfile.resize(model->ops.size());
    for (size_t i = 0; i < model->ops.size(); ++i) {
        const OpDesc& op = model->ops[i];
        std::vector<Tensor*> inputs;
        std::vector<Tensor*> outputs;
        inputs.reserve(op.inputs.size());
        outputs.reserve(op.outputs.size());
        for (int t : op.inputs) {
            if (t < 0 || t >= static_cast<int>(mTensors.size())) {
                return INVALID_INPUT;
            }
            inputs.push_back(mTensors[t].get());
        }
        for (int t : op.outputs) {
            if (t < 0 || t >= static_cast<int>(mTensors.size())) {
                return INVALID_INPUT;
            }
            outputs.push_back(mTensors[t].get());
        }
        auto execution = backend->onCreate(op, inputs, outputs);
        if (!execution) {
            fprintf(stderr, "infer: unsupported op %s (%s)\n", opTypeName(op.type), op.name.c_str());
            return NOT_SUPPORT;
        }
        mExecutions[i] = execution;
        // Input tensors take their final shape from the graph description.
        if (op.type == OpType_Input) {
            for (Tensor* output : outputs) {
                if (output->shape().empty()) {
                    continue;
                }
            }
        }
        mProfile[i].name = op.name.empty() ? opTypeName(op.type) : op.name;
        mProfile[i].type = opTypeName(op.type);
    }

    mResized = false;
    // Allocate the runtime buffers up front so callers can fill the input
    // tensors right after the session is created.
    return resize();
}

Tensor* Session::input(int index) const {
    const Model* model = mInterpreter->model();
    if (index < 0 || index >= static_cast<int>(model->inputIndices.size())) {
        return nullptr;
    }
    return mTensors[model->inputIndices[index]].get();
}

Tensor* Session::output(int index) const {
    const Model* model = mInterpreter->model();
    if (index < 0 || index >= static_cast<int>(model->outputIndices.size())) {
        return nullptr;
    }
    return mTensors[model->outputIndices[index]].get();
}

Tensor* Session::input(const std::string& name) const {
    const Model* model = mInterpreter->model();
    for (size_t i = 0; i < model->inputIndices.size(); ++i) {
        if (model->tensors[model->inputIndices[i]].name == name) {
            return input(static_cast<int>(i));
        }
    }
    return nullptr;
}

Tensor* Session::output(const std::string& name) const {
    const Model* model = mInterpreter->model();
    for (size_t i = 0; i < model->outputIndices.size(); ++i) {
        if (model->tensors[model->outputIndices[i]].name == name) {
            return output(static_cast<int>(i));
        }
    }
    return nullptr;
}

Tensor* Session::tensor(int index) const {
    if (index < 0 || index >= static_cast<int>(mTensors.size())) {
        return nullptr;
    }
    return mTensors[index].get();
}

Tensor* Session::tensor(const std::string& name) const {
    for (const auto& tensor : mTensors) {
        if (tensor->name() == name) {
            return tensor.get();
        }
    }
    return nullptr;
}

int Session::inputCount() const {
    return static_cast<int>(mInterpreter->model()->inputIndices.size());
}

int Session::outputCount() const {
    return static_cast<int>(mInterpreter->model()->outputIndices.size());
}

ErrorCode Session::resizeInput(int index, const std::vector<int>& dims) {
    Tensor* tensor = input(index);
    if (!tensor) {
        return INVALID_INPUT;
    }
    tensor->setShape(dims);
    mResized = false;
    return NO_ERROR;
}

ErrorCode Session::resizeInput(const std::string& name, const std::vector<int>& dims) {
    const Model* model = mInterpreter->model();
    for (size_t i = 0; i < model->inputIndices.size(); ++i) {
        if (model->tensors[model->inputIndices[i]].name == name) {
            return resizeInput(static_cast<int>(i), dims);
        }
    }
    return INVALID_INPUT;
}

namespace {

struct ArenaBlock {
    size_t offset = 0;
    size_t size   = 0;
    bool free     = true;
};

constexpr size_t kAlign = 64;

size_t alignBytes(size_t value) {
    return (value + kAlign - 1) & ~(kAlign - 1);
}

bool isShapeFoldingType(OpType type) {
    switch (type) {
        case OpType_StridedSlice:
        case OpType_SliceTf:
        case OpType_Pack:
        case OpType_Concat:
        case OpType_Squeeze:
        case OpType_Flatten:
        case OpType_Reshape:
        case OpType_GatherV2:
            return true;
        default:
            return false;
    }
}

} // namespace

ErrorCode Session::resize() {
    const Model* model = mInterpreter->model();
    mShapeBuffers.clear();
    for (int index : mMaterialized) {
        Tensor* tensor = mTensors[index].get();
        tensor->setHost(nullptr);
        tensor->setConstant(false);
    }
    mMaterialized.clear();
    for (size_t i = 0; i < model->ops.size(); ++i) {
        const OpDesc& op = model->ops[i];
        std::vector<Tensor*> inputs;
        std::vector<Tensor*> outputs;
        for (int t : op.inputs) {
            inputs.push_back(mTensors[t].get());
        }
        for (int t : op.outputs) {
            outputs.push_back(mTensors[t].get());
        }
        const ErrorCode code = mExecutions[i]->onResize(inputs, outputs);
        if (code != NO_ERROR) {
            fprintf(stderr, "infer: resize failed at op %d %s (%s), error %d\n", static_cast<int>(i), opTypeName(op.type),
                    op.name.c_str(), static_cast<int>(code));
            for (const Tensor* input : inputs) {
                fprintf(stderr, "  input: %s\n", input->describe().c_str());
            }
            return code;
        }
        if (std::getenv("INFER_RESIZE_TRACE")) {
            fprintf(stderr, "[%d] %s", static_cast<int>(i), opTypeName(op.type));
            for (const Tensor* output : outputs) {
                fprintf(stderr, " -> %s", output->describe().c_str());
            }
            fprintf(stderr, "\n");
        }
        if (op.type == OpType_Shape && !op.inputs.empty() && !op.outputs.empty()) {
            // Shape values depend only on the current shapes, so they can be
            // produced immediately and consumed by later resize passes.
            const Tensor* source = mTensors[op.inputs[0]].get();
            Tensor* target       = mTensors[op.outputs[0]].get();
            const int rank       = std::max(1, source->dimensions());
            auto buffer          = std::unique_ptr<int[]>(new int[rank]);
            for (int d = 0; d < source->dimensions(); ++d) {
                buffer[d] = source->length(d);
            }
            target->setShape({source->dimensions()});
            target->setType(DT_INT32);
            target->setHost(buffer.get());
            target->setConstant(true);
            mShapeBuffers.push_back(std::move(buffer));
            mMaterialized.push_back(op.outputs[0]);
        } else if (isShapeFoldingType(op.type) && !op.outputs.empty()) {
            // Evaluate shape-only sub-graphs at resize time so that later
            // operators (Reshape, Padding, ...) can consume their values.
            bool foldable = true;
            for (const Tensor* input : inputs) {
                if (!input->host()) {
                    foldable = false;
                    break;
                }
            }
            if (foldable) {
                for (const Tensor* output : outputs) {
                    if (output->type() != DT_INT32 || output->bytes() == 0 || output->bytes() > 16384) {
                        foldable = false;
                        break;
                    }
                }
            }
            if (foldable) {
                for (size_t j = 0; j < outputs.size(); ++j) {
                    Tensor* output     = outputs[j];
                    const size_t words = output->bytes() / sizeof(int) + 2;
                    auto buffer        = std::unique_ptr<int[]>(new int[words]);
                    output->setHost(buffer.get());
                    output->setConstant(true);
                    mShapeBuffers.push_back(std::move(buffer));
                    mMaterialized.push_back(op.outputs[j]);
                }
                const ErrorCode foldCode = mExecutions[i]->onExecute(inputs, outputs);
                if (foldCode != NO_ERROR) {
                    return foldCode;
                }
            }
        }
    }
    return allocate();
}

ErrorCode Session::allocate() {
    const Model* model = mInterpreter->model();
    const int opCount  = static_cast<int>(model->ops.size());
    const int tensorCount = static_cast<int>(mTensors.size());

    std::vector<int> producer(tensorCount, -1);
    std::vector<int> lastUse(tensorCount, -1);
    for (int i = 0; i < opCount; ++i) {
        const OpDesc& op = model->ops[i];
        for (int t : op.outputs) {
            producer[t] = i;
        }
        for (int t : op.inputs) {
            lastUse[t] = std::max(lastUse[t], i);
        }
    }

    std::vector<bool> isOutput(tensorCount, false);
    for (int t : model->outputIndices) {
        isOutput[t] = true;
    }

    for (int i = 0; i < opCount; ++i) {
        for (int t : model->ops[i].inputs) {
            if (lastUse[t] < 0) {
                lastUse[t] = i;
            }
        }
    }

    std::vector<int> begin(tensorCount, -1);
    std::vector<int> end(tensorCount, -1);
    for (int t = 0; t < tensorCount; ++t) {
        Tensor* tensor = mTensors[t].get();
        if (tensor->isConstant()) {
            continue;
        }
        const int b = producer[t] >= 0 ? producer[t] : 0;
        int e       = lastUse[t];
        if (isOutput[t]) {
            e = opCount;
        }
        if (e < b) {
            e = b;
        }
        begin[t] = b;
        end[t]   = e;
        if (tensor->bytes() == 0) {
            begin[t] = end[t] = -1;
        }
    }

    // Drop the previous allocation.
    if (mArena) {
        std::free(mArena);
        mArena = nullptr;
    }

    std::vector<ArenaBlock> blocks;
    std::vector<size_t> offsets(tensorCount, 0);
    std::vector<int> blockOf(tensorCount, -1);
    size_t arenaSize = 0;

    auto releaseTensor = [&](int t) {
        const int index = blockOf[t];
        if (index >= 0) {
            blocks[index].free = true;
            blockOf[t]         = -1;
        }
    };
    auto acquireTensor = [&](int t) -> ErrorCode {
        const size_t size = alignBytes(mTensors[t]->bytes());
        if (size == 0) {
            return NO_ERROR;
        }
        size_t best      = static_cast<size_t>(-1);
        size_t bestIndex = blocks.size();
        for (size_t i = 0; i < blocks.size(); ++i) {
            if (blocks[i].free && blocks[i].size >= size && blocks[i].size < best) {
                best      = blocks[i].size;
                bestIndex = i;
            }
        }
        if (bestIndex != blocks.size()) {
            // Split the block when the remainder is still usable.
            if (blocks[bestIndex].size > size) {
                ArenaBlock rest;
                rest.offset = blocks[bestIndex].offset + size;
                rest.size   = blocks[bestIndex].size - size;
                rest.free   = true;
                blocks[bestIndex].size = size;
                blocks.push_back(rest);
            }
            blocks[bestIndex].free = false;
            offsets[t]             = blocks[bestIndex].offset;
            blockOf[t]             = static_cast<int>(bestIndex);
            return NO_ERROR;
        }
        ArenaBlock block;
        block.offset = arenaSize;
        block.size   = size;
        block.free   = false;
        offsets[t]   = block.offset;
        arenaSize += size;
        blockOf[t] = static_cast<int>(blocks.size());
        blocks.push_back(block);
        return NO_ERROR;
    };

    for (int i = 0; i <= opCount; ++i) {
        // Free tensors whose live range ended before this operator.
        for (int t = 0; t < tensorCount; ++t) {
            if (begin[t] >= 0 && end[t] < i) {
                releaseTensor(t);
                begin[t] = -1;
            }
        }
        if (i == opCount) {
            break;
        }
        for (int t = 0; t < tensorCount; ++t) {
            if (begin[t] == i) {
                const ErrorCode code = acquireTensor(t);
                if (code != NO_ERROR) {
                    return code;
                }
            }
        }
    }

    if (arenaSize > 0) {
        void* arena = nullptr;
        if (posix_memalign(&arena, kAlign, arenaSize) != 0) {
            return OUT_OF_MEMORY;
        }
        std::memset(arena, 0, arenaSize);
        mArena = arena;
    }
    mArenaSize = arenaSize;
    for (int t = 0; t < tensorCount; ++t) {
        Tensor* tensor = mTensors[t].get();
        if (tensor->isConstant()) {
            continue;
        }
        if (tensor->bytes() > 0 && offsets[t] < arenaSize) {
            tensor->setHost(static_cast<uint8_t*>(mArena) + offsets[t]);
        }
    }
    mResized = true;
    return NO_ERROR;
}

ErrorCode Session::run() {
    if (!mResized) {
        const ErrorCode code = resize();
        if (code != NO_ERROR) {
            return code;
        }
    }
    return execute();
}

ErrorCode Session::execute() {
    const Model* model = mInterpreter->model();
    double total       = 0.0;
    for (size_t i = 0; i < model->ops.size(); ++i) {
        const OpDesc& op = model->ops[i];
        std::vector<Tensor*> inputs;
        std::vector<Tensor*> outputs;
        inputs.reserve(op.inputs.size());
        outputs.reserve(op.outputs.size());
        for (int t : op.inputs) {
            inputs.push_back(mTensors[t].get());
        }
        for (int t : op.outputs) {
            outputs.push_back(mTensors[t].get());
        }
        const auto start = std::chrono::steady_clock::now();
        const ErrorCode code = mExecutions[i]->onExecute(inputs, outputs);
        const auto stop  = std::chrono::steady_clock::now();
        const double ms  = std::chrono::duration<double, std::milli>(stop - start).count();
        mProfile[i].ms   = ms;
        total += ms;
        if (code != NO_ERROR) {
            fprintf(stderr, "infer: execute failed at op %d %s (%s), error %d\n", static_cast<int>(i),
                    opTypeName(op.type), op.name.c_str(), static_cast<int>(code));
            return code;
        }
    }
    mTotalMs = total;
    return NO_ERROR;
}

} // namespace infer
