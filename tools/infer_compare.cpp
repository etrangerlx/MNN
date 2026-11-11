// Compares the infer engine against MNN's own CPU interpreter on the same
// model file, using identical deterministic inputs.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

#include "infer/Interpreter.h"

namespace {

float patternValue(size_t index) {
    const uint32_t x = static_cast<uint32_t>(index) * 1664525u + 1013904223u;
    const int32_t v  = static_cast<int32_t>(x);
    return static_cast<float>(v) / 2147483648.0f * 0.5f;
}

void fillHostTensor(MNN::Tensor* tensor) {
    float* data = tensor->host<float>();
    const size_t count = tensor->elementSize();
    for (size_t i = 0; i < count; ++i) {
        data[i] = patternValue(i);
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s model.mnn [--threads N] [--runs N]\n", argv[0]);
        return 1;
    }
    const std::string path = argv[1];
    int threads            = 4;
    int runs               = 3;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--threads" && i + 1 < argc) {
            threads = std::atoi(argv[++i]);
        } else if (arg == "--runs" && i + 1 < argc) {
            runs = std::atoi(argv[++i]);
        }
    }

    infer::ErrorCode code = infer::NO_ERROR;
    auto interpreter      = infer::Interpreter::createFromFile(path, &code);
    if (!interpreter) {
        fprintf(stderr, "infer: failed to load %s: %s\n", path.c_str(), infer::errorCodeName(code));
        return 1;
    }
    auto session = interpreter->createSession(threads);
    if (!session) {
        fprintf(stderr, "infer: failed to create session\n");
        return 1;
    }

    MNN::Interpreter* mnnInterpreter = MNN::Interpreter::createFromFile(path.c_str());
    if (!mnnInterpreter) {
        fprintf(stderr, "mnn: failed to load %s\n", path.c_str());
        return 1;
    }
    MNN::ScheduleConfig config;
    config.type      = MNN_FORWARD_CPU;
    config.numThread = threads;
    MNN::BackendConfig backendConfig;
    backendConfig.precision = MNN::BackendConfig::Precision_High;
    config.backendConfig    = &backendConfig;
    MNN::Session* mnnSession = mnnInterpreter->createSession(config);
    if (!mnnSession) {
        fprintf(stderr, "mnn: failed to create session\n");
        return 1;
    }

    printf("model: %s\n", path.c_str());
    printf("infer inputs=%d outputs=%d threads=%d\n", session->inputCount(), session->outputCount(),
           session->threadNumber());

    // Feed identical inputs into both engines.
    std::vector<MNN::Tensor*> mnnHostInputs(session->inputCount(), nullptr);
    for (int i = 0; i < session->inputCount(); ++i) {
        infer::Tensor* input     = session->input(i);
        const std::string& name  = input->name();
        MNN::Tensor* mnnInput    = mnnInterpreter->getSessionInput(mnnSession, name.empty() ? nullptr : name.c_str());
        if (!mnnInput) {
            fprintf(stderr, "mnn: missing input %s\n", name.c_str());
            return 1;
        }
        // Keep the shape MNN derived for the graph input. Resizing a legacy model
        // forces a different layout and changes the whole graph.
        std::vector<int> mnnShape(mnnInput->dimensions());
        for (int d = 0; d < mnnInput->dimensions(); ++d) {
            mnnShape[d] = mnnInput->length(d);
        }
        if (mnnShape != input->shape()) {
            fprintf(stderr, "input shape mismatch: infer=[");
            for (size_t d = 0; d < input->shape().size(); ++d) {
                fprintf(stderr, "%s%d", d ? "," : "", input->shape()[d]);
            }
            fprintf(stderr, "] mnn=[");
            for (size_t d = 0; d < mnnShape.size(); ++d) {
                fprintf(stderr, "%s%d", d ? "," : "", mnnShape[d]);
            }
            fprintf(stderr, "]\n");
            return 1;
        }
        MNN::Tensor* hostInput = MNN::Tensor::create<float>(mnnShape, nullptr, MNN::Tensor::CAFFE);
        fillHostTensor(hostInput);
        mnnInput->copyFromHostTensor(hostInput);
        mnnHostInputs[i] = hostInput;
        printf("  input[%d] %s\n", i, input->describe().c_str());
        float* dst = input->host<float>();
        for (size_t k = 0; k < input->elementCount(); ++k) {
            dst[k] = patternValue(k);
        }
    }

    // Run MNN first and snapshot its outputs.
    mnnInterpreter->runSession(mnnSession);
    std::vector<std::vector<float>> mnnOutputs(session->outputCount());
    for (int i = 0; i < session->outputCount(); ++i) {
        auto* output            = session->output(i);
        const std::string& name = output->name();
        MNN::Tensor* mnnOutput  = mnnInterpreter->getSessionOutput(mnnSession, name.empty() ? nullptr : name.c_str());
        if (!mnnOutput) {
            fprintf(stderr, "mnn: missing output %s\n", name.c_str());
            return 1;
        }
        MNN::Tensor hostOutput(mnnOutput, MNN::Tensor::CAFFE);
        mnnOutput->copyToHostTensor(&hostOutput);
        mnnOutputs[i].assign(hostOutput.host<float>(), hostOutput.host<float>() + hostOutput.elementSize());
    }

    if (session->run() != infer::NO_ERROR) {
        fprintf(stderr, "infer: run failed\n");
        return 1;
    }
    const double inferFirstMs = session->totalMs();

    bool allPass = true;
    for (int i = 0; i < session->outputCount(); ++i) {
        infer::Tensor* output = session->output(i);
        const std::vector<float>& mnnVec = mnnOutputs[i];
        const float* mnnData  = mnnVec.data();
        const float* ourData  = output->host<float>();
        const size_t mnnCount = mnnVec.size();
        const size_t ourCount = output->elementCount();
        if (mnnCount != ourCount) {
            printf("  output[%d] SHAPE MISMATCH ours=%zu mnn=%zu\n", i, ourCount, mnnCount);
            allPass = false;
            continue;
        }
        double maxAbs = 0.0;
        double sumAbs = 0.0;
        size_t worstIndex = 0;
        for (size_t k = 0; k < ourCount; ++k) {
            const double diff = std::fabs(static_cast<double>(ourData[k]) - static_cast<double>(mnnData[k]));
            sumAbs += diff;
            if (diff > maxAbs) {
                maxAbs     = diff;
                worstIndex = k;
            }
        }
        const double meanAbs = ourCount > 0 ? sumAbs / ourCount : 0.0;
        const bool pass      = maxAbs < 1e-3;
        allPass              = allPass && pass;
        printf("  output[%d] %s maxAbs=%.3e meanAbs=%.3e at[%zu] ours=%.6f mnn=%.6f %s\n", i,
               output->describe().c_str(), maxAbs, meanAbs, worstIndex,
               ourCount > 0 ? ourData[worstIndex] : 0.0f, ourCount > 0 ? mnnData[worstIndex] : 0.0f,
               pass ? "PASS" : "FAIL");
    }

    // Latency comparison.
    const int timedRuns = std::max(1, runs);
    double ourTotal     = 0.0;
    for (int r = 0; r < timedRuns; ++r) {
        session->run();
        ourTotal += session->totalMs();
    }
    const auto start = std::chrono::steady_clock::now();
    for (int r = 0; r < timedRuns; ++r) {
        mnnInterpreter->runSession(mnnSession);
    }
    const auto stop = std::chrono::steady_clock::now();
    const double mnnMs = std::chrono::duration<double, std::milli>(stop - start).count() / timedRuns;
    printf("latency: infer %.3f ms, mnn %.3f ms (first run infer %.3f ms)\n", ourTotal / timedRuns, mnnMs, inferFirstMs);

    for (auto* tensor : mnnHostInputs) {
        MNN::Tensor::destroy(tensor);
    }
    mnnInterpreter->releaseSession(mnnSession);
    MNN::Interpreter::destroy(mnnInterpreter);
    printf("%s\n", allPass ? "RESULT: PASS" : "RESULT: FAIL");
    return allPass ? 0 : 2;
}
