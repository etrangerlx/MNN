#pragma once

#include <memory>
#include <vector>

#include "infer/Op.h"
#include "infer/Tensor.h"
#include "infer/ThreadPool.h"

namespace infer {

class CPUBackend : public Backend {
public:
    explicit CPUBackend(ThreadPool* pool) : mPool(pool) {
    }

    const char* name() const override {
        return "CPU";
    }

    std::shared_ptr<Execution> onCreate(const OpDesc& op, const std::vector<Tensor*>& inputs,
                                        const std::vector<Tensor*>& outputs) override;

    ThreadPool* pool() const {
        return mPool;
    }

private:
    ThreadPool* mPool = nullptr;
};

// CPU operators store the generic Backend pointer in Execution, which is all the
// base class needs to know about. This helper reaches the shared thread pool.
inline ThreadPool* cpuPool(Backend* backend) {
    return static_cast<CPUBackend*>(backend)->pool();
}

std::shared_ptr<Execution> createCPUConvolutionOp(CPUBackend* backend, const OpDesc& op,
                                                  const std::vector<Tensor*>& inputs,
                                                  const std::vector<Tensor*>& outputs);
std::shared_ptr<Execution> createCPUPoolingOp(CPUBackend* backend, const OpDesc& op,
                                              const std::vector<Tensor*>& inputs,
                                              const std::vector<Tensor*>& outputs);
std::shared_ptr<Execution> createCPUUnaryOp(CPUBackend* backend, const OpDesc& op,
                                            const std::vector<Tensor*>& inputs,
                                            const std::vector<Tensor*>& outputs);
std::shared_ptr<Execution> createCPUShapeOp(CPUBackend* backend, const OpDesc& op,
                                            const std::vector<Tensor*>& inputs,
                                            const std::vector<Tensor*>& outputs);
std::shared_ptr<Execution> createCPUInnerProductOp(CPUBackend* backend, const OpDesc& op,
                                                   const std::vector<Tensor*>& inputs,
                                                   const std::vector<Tensor*>& outputs);

} // namespace infer
