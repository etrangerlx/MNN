#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "infer/Model.h"
#include "infer/Op.h"
#include "infer/ThreadPool.h"
#include "infer/Types.h"

namespace infer {

class Interpreter;

struct OpProfile {
    std::string name;
    std::string type;
    double ms = 0.0;
};

// A Session owns the runtime tensors of one model instance: it performs static
// shape inference, plans a single reusable arena for all intermediate tensors
// and executes the operators in topological order.
class Session {
public:
    ~Session();

    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    int inputCount() const;
    int outputCount() const;

    Tensor* input(int index) const;
    Tensor* output(int index) const;
    Tensor* input(const std::string& name) const;
    Tensor* output(const std::string& name) const;
    Tensor* tensor(const std::string& name) const;
    Tensor* tensor(int index) const;

    ErrorCode resizeInput(int index, const std::vector<int>& dims);
    ErrorCode resizeInput(const std::string& name, const std::vector<int>& dims);

    ErrorCode run();

    const std::vector<OpProfile>& profile() const {
        return mProfile;
    }
    double totalMs() const {
        return mTotalMs;
    }
    size_t arenaSize() const {
        return mArenaSize;
    }
    int threadNumber() const;

private:
    friend class Interpreter;
    Session(std::shared_ptr<Interpreter> interpreter, int threads);
    ErrorCode init();
    ErrorCode resize();
    ErrorCode allocate();
    ErrorCode execute();

    std::shared_ptr<Interpreter> mInterpreter;
    std::vector<std::unique_ptr<Tensor>> mTensors;
    std::vector<std::shared_ptr<Execution>> mExecutions;
    std::shared_ptr<Backend> mBackend;
    std::unique_ptr<ThreadPool> mPool;
    // Values produced by Shape ops, materialized before allocation so that
    // downstream operators (e.g. Reshape) can consume them during resize.
    std::vector<std::unique_ptr<int[]>> mShapeBuffers;
    // Tensors whose memory is provided by mShapeBuffers instead of the arena.
    std::vector<int> mMaterialized;
    std::vector<OpProfile> mProfile;
    double mTotalMs = 0.0;
    size_t mArenaSize = 0;
    void* mArena     = nullptr;
    bool mResized    = false;
    int mThreads     = 1;
};

class Interpreter : public std::enable_shared_from_this<Interpreter> {
public:
    static std::shared_ptr<Interpreter> createFromFile(const std::string& path, ErrorCode* code = nullptr);
    static std::shared_ptr<Interpreter> createFromBuffer(const void* buffer, size_t size, ErrorCode* code = nullptr);
    // Builds an interpreter from a programmatically constructed model (used by
    // the unit tests and for embedding graphs without the flatbuffer format).
    static std::shared_ptr<Interpreter> createFromModel(std::shared_ptr<Model> model);

    const Model* model() const {
        return mModel.get();
    }
    const std::vector<std::string>& inputNames() const {
        return mInputNames;
    }
    const std::vector<std::string>& outputNames() const {
        return mOutputNames;
    }

    std::shared_ptr<Session> createSession(int threads = 0) const;
    int defaultThreadNumber() const;

private:
    Interpreter() = default;

    std::shared_ptr<Model> mModel;
    std::vector<std::string> mInputNames;
    std::vector<std::string> mOutputNames;
};

} // namespace infer
