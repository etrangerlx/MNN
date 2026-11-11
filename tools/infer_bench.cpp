// Runs an MNN model with the infer engine and reports outputs / latency.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "infer/Interpreter.h"

namespace {

float patternValue(size_t index) {
    const uint32_t x = static_cast<uint32_t>(index) * 1664525u + 1013904223u;
    const int32_t v  = static_cast<int32_t>(x);
    return static_cast<float>(v) / 2147483648.0f * 0.5f;
}

void fillPattern(infer::Tensor* tensor) {
    float* data = tensor->host<float>();
    if (!data) {
        return;
    }
    const size_t count = tensor->elementCount();
    for (size_t i = 0; i < count; ++i) {
        data[i] = patternValue(i);
    }
}

void printTensorSummary(const infer::Tensor* tensor) {
    const float* data = tensor->host<float>();
    if (!data || tensor->elementCount() == 0) {
        printf("    <empty>\n");
        return;
    }
    double sum = 0.0;
    float minValue = data[0], maxValue = data[0];
    for (size_t i = 0; i < tensor->elementCount(); ++i) {
        sum += data[i];
        minValue = std::min(minValue, data[i]);
        maxValue = std::max(maxValue, data[i]);
    }
    printf("    sum=%.6f mean=%.6f min=%.6f max=%.6f first=[", sum, sum / tensor->elementCount(), minValue, maxValue);
    for (size_t i = 0; i < std::min<size_t>(tensor->elementCount(), 8); ++i) {
        printf("%s%.6f", i ? "," : "", data[i]);
    }
    printf("]\n");
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s model.mnn [--threads N] [--runs N] [--save-output file]\n", argv[0]);
        return 1;
    }
    const std::string path = argv[1];
    int threads            = 4;
    int runs               = 10;
    std::string savePath;
    bool dumpTensors = false;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--threads" && i + 1 < argc) {
            threads = std::atoi(argv[++i]);
        } else if (arg == "--runs" && i + 1 < argc) {
            runs = std::atoi(argv[++i]);
        } else if (arg == "--dump-tensors") {
            dumpTensors = true;
        } else if (arg == "--save-output" && i + 1 < argc) {
            savePath = argv[++i];
        }
    }

    infer::ErrorCode code = infer::NO_ERROR;
    auto interpreter      = infer::Interpreter::createFromFile(path, &code);
    if (!interpreter) {
        fprintf(stderr, "failed to load %s: %s\n", path.c_str(), infer::errorCodeName(code));
        return 1;
    }
    auto session = interpreter->createSession(threads);
    if (!session) {
        fprintf(stderr, "failed to create session\n");
        return 1;
    }

    printf("model: %s\n", path.c_str());
    printf("threads: %d\n", session->threadNumber());
    printf("inputs: %d\n", session->inputCount());
    for (int i = 0; i < session->inputCount(); ++i) {
        infer::Tensor* tensor = session->input(i);
        printf("  input[%d] %s\n", i, tensor->describe().c_str());
        if (tensor->shape().empty()) {
            fprintf(stderr, "input has no shape in the model, cannot run\n");
            return 1;
        }
    }
    printf("outputs: %d\n", session->outputCount());
    for (int i = 0; i < session->outputCount(); ++i) {
        printf("  output[%d] %s\n", i, session->output(i)->describe().c_str());
    }

    for (int i = 0; i < session->inputCount(); ++i) {
        fillPattern(session->input(i));
    }

    if (dumpTensors) {
        printf("tensors:\n");
        for (int i = 0;; ++i) {
            infer::Tensor* tensor = session->tensor(i);
            if (!tensor) {
                break;
            }
            printf("  [%d] %s\n", i, tensor->describe().c_str());
        }
    }

    const infer::ErrorCode runCode = session->run();
    if (runCode != infer::NO_ERROR) {
        fprintf(stderr, "run failed: %s\n", infer::errorCodeName(runCode));
        return 1;
    }

    printf("output values:\n");
    for (int i = 0; i < session->outputCount(); ++i) {
        printf("  output[%d] %s\n", i, session->output(i)->describe().c_str());
        printTensorSummary(session->output(i));
    }

    if (!savePath.empty()) {
        std::ofstream out(savePath, std::ios::binary);
        for (int i = 0; i < session->outputCount(); ++i) {
            infer::Tensor* tensor = session->output(i);
            out.write(reinterpret_cast<const char*>(tensor->host()), tensor->bytes());
        }
        printf("saved outputs to %s\n", savePath.c_str());
    }

    const int timedRuns = std::max(1, runs);
    for (int r = 0; r < timedRuns; ++r) {
        if (session->run() != infer::NO_ERROR) {
            return 1;
        }
    }
    printf("latency: %.3f ms/run (%d runs, arena %.2f MB)\n", session->totalMs(), timedRuns,
           session->arenaSize() / 1024.0 / 1024.0);

    std::vector<infer::OpProfile> profile = session->profile();
    std::sort(profile.begin(), profile.end(), [](const infer::OpProfile& a, const infer::OpProfile& b) {
        return a.ms > b.ms;
    });
    printf("top operators (last run):\n");
    for (size_t i = 0; i < std::min<size_t>(profile.size(), 10); ++i) {
        printf("  %-40s %-22s %8.3f ms\n", profile[i].name.c_str(), profile[i].type.c_str(), profile[i].ms);
    }
    return 0;
}
