// Unit tests for the CPU backend operators. Every operator is compared against
// a straightforward reference implementation written in this file.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "infer/Interpreter.h"

namespace {

int gFailures = 0;
int gChecks   = 0;

#define CHECK(condition, message)                                                        \
    do {                                                                                 \
        ++gChecks;                                                                       \
        if (!(condition)) {                                                              \
            ++gFailures;                                                                 \
            printf("  FAIL: %s (%s:%d)\n", message, __FILE__, __LINE__);                 \
        }                                                                                \
    } while (0)

struct Graph {
    std::shared_ptr<infer::Model> model = std::make_shared<infer::Model>();

    int addTensor(const std::string& name, const std::vector<int>& dims, const std::vector<float>& data = {}) {
        infer::TensorDesc desc;
        desc.name = name;
        desc.dims = dims;
        desc.type = infer::DT_FLOAT;
        if (!data.empty()) {
            desc.data    = std::make_shared<std::vector<uint8_t>>(data.size() * sizeof(float));
            std::memcpy(desc.data->data(), data.data(), desc.data->size());
            desc.hasData = true;
        }
        model->tensors.push_back(desc);
        return static_cast<int>(model->tensors.size()) - 1;
    }

    int addIntTensor(const std::string& name, const std::vector<int>& dims, const std::vector<int>& data) {
        infer::TensorDesc desc;
        desc.name = name;
        desc.dims = dims;
        desc.type = infer::DT_INT32;
        if (!data.empty()) {
            desc.data    = std::make_shared<std::vector<uint8_t>>(data.size() * sizeof(int32_t));
            std::memcpy(desc.data->data(), data.data(), desc.data->size());
            desc.hasData = true;
        }
        model->tensors.push_back(desc);
        return static_cast<int>(model->tensors.size()) - 1;
    }

    int addOp(infer::OpType type, const std::vector<int>& inputs, const std::vector<int>& outputs,
              const std::function<void(infer::OpAttr&)>& setup = nullptr) {
        infer::OpDesc op;
        op.type    = type;
        op.inputs  = inputs;
        op.outputs = outputs;
        op.name    = infer::opTypeName(type);
        if (setup) {
            setup(op.attr);
        }
        model->ops.push_back(op);
        return static_cast<int>(model->ops.size()) - 1;
    }

    void setInputs(const std::vector<int>& indices) {
        model->inputIndices = indices;
    }
    void setOutputs(const std::vector<int>& indices) {
        model->outputIndices = indices;
    }
};

struct Runner {
    std::shared_ptr<infer::Session> session;

    explicit Runner(Graph& graph, int threads = 2) {
        auto interpreter = infer::Interpreter::createFromModel(graph.model);
        session          = interpreter ? interpreter->createSession(threads) : nullptr;
    }

    std::vector<float> run(int inputIndex, const std::vector<float>& input, int outputIndex = 0) {
        if (!session) {
            return {};
        }
        infer::Tensor* tensor = session->input(inputIndex);
        if (tensor && !input.empty()) {
            std::memcpy(tensor->host(), input.data(), std::min(input.size(), tensor->elementCount()) * sizeof(float));
        }
        if (session->run() != infer::NO_ERROR) {
            return {};
        }
        infer::Tensor* out = session->output(outputIndex);
        if (!out || !out->host()) {
            return {};
        }
        std::vector<float> result(out->elementCount());
        std::memcpy(result.data(), out->host(), result.size() * sizeof(float));
        return result;
    }
};

double maxDiff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) {
        return 1e9;
    }
    double worst = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        worst = std::max(worst, std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
    }
    return worst;
}

void report(const char* name, const std::vector<float>& actual, const std::vector<float>& expected) {
    const double diff = maxDiff(actual, expected);
    CHECK(actual.size() == expected.size(), name);
    CHECK(diff < 1e-4, name);
    printf("  %-28s maxDiff=%.3e size=%zu\n", name, diff, actual.size());
}

std::vector<float> makeInput(size_t count) {
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i) {
        values[i] = std::sin(static_cast<float>(i) * 0.37f) * 0.5f;
    }
    return values;
}

// --------------------------------------------------------------------------
void testConvolution() {
    printf("Convolution\n");
    const int N = 1, C = 2, H = 5, W = 5, O = 3, K = 3;
    Graph graph;
    const int input  = graph.addTensor("input", {N, C, H, W});
    const int output = graph.addTensor("output", {});
    std::vector<float> weight(O * C * K * K), bias(O);
    for (size_t i = 0; i < weight.size(); ++i) {
        weight[i] = std::sin(static_cast<float>(i) * 0.11f) * 0.2f;
    }
    for (int i = 0; i < O; ++i) {
        bias[i] = static_cast<float>(i) * 0.01f;
    }
    graph.addOp(infer::OpType_Convolution, {input}, {output}, [&](infer::OpAttr& attr) {
        attr.conv.kernelX = K;
        attr.conv.kernelY = K;
        attr.conv.strideX = 1;
        attr.conv.strideY = 1;
        attr.conv.padX    = 1;
        attr.conv.padY    = 1;
        attr.conv.padMode = infer::PadMode_CAFFE;
        attr.conv.group   = 1;
        attr.conv.outputCount = O;
        attr.conv.inputCount  = C;
        attr.weight = weight;
        attr.bias   = bias;
    });
    graph.setInputs({input});
    graph.setOutputs({output});

    const std::vector<float> inputData = makeInput(N * C * H * W);
    Runner runner(graph);
    const std::vector<float> actual = runner.run(0, inputData);

    std::vector<float> expected(N * O * H * W, 0.0f);
    for (int n = 0; n < N; ++n) {
        for (int o = 0; o < O; ++o) {
            for (int y = 0; y < H; ++y) {
                for (int x = 0; x < W; ++x) {
                    float sum = bias[o];
                    for (int c = 0; c < C; ++c) {
                        for (int ky = -1; ky <= 1; ++ky) {
                            for (int kx = -1; kx <= 1; ++kx) {
                                const int iy = y + ky;
                                const int ix = x + kx;
                                if (iy < 0 || iy >= H || ix < 0 || ix >= W) {
                                    continue;
                                }
                                const float v = inputData[((n * C + c) * H + iy) * W + ix];
                                sum += v * weight[((o * C + c) * K + (ky + 1)) * K + (kx + 1)];
                            }
                        }
                    }
                    expected[((n * O + o) * H + y) * W + x] = sum;
                }
            }
        }
    }
    report("conv 3x3 pad1", actual, expected);
}

void testDepthwiseConvolution() {
    printf("DepthwiseConvolution\n");
    const int N = 1, C = 3, H = 4, W = 4, K = 3;
    Graph graph;
    const int input  = graph.addTensor("input", {N, C, H, W});
    const int output = graph.addTensor("output", {});
    std::vector<float> weight(C * K * K), bias(C);
    for (size_t i = 0; i < weight.size(); ++i) {
        weight[i] = std::cos(static_cast<float>(i) * 0.23f) * 0.3f;
    }
    for (int i = 0; i < C; ++i) {
        bias[i] = 0.05f;
    }
    graph.addOp(infer::OpType_ConvolutionDepthwise, {input}, {output}, [&](infer::OpAttr& attr) {
        attr.conv.kernelX = K;
        attr.conv.kernelY = K;
        attr.conv.strideX = 2;
        attr.conv.strideY = 2;
        attr.conv.padX    = 1;
        attr.conv.padY    = 1;
        attr.conv.padMode = infer::PadMode_CAFFE;
        attr.conv.group   = C;
        attr.conv.outputCount = C;
        attr.conv.inputCount  = C;
        attr.weight = weight;
        attr.bias   = bias;
    });
    graph.setInputs({input});
    graph.setOutputs({output});

    const std::vector<float> inputData = makeInput(N * C * H * W);
    Runner runner(graph);
    const std::vector<float> actual = runner.run(0, inputData);
    const int outH = 2, outW = 2;
    std::vector<float> expected(N * C * outH * outW, 0.0f);
    for (int c = 0; c < C; ++c) {
        for (int y = 0; y < outH; ++y) {
            for (int x = 0; x < outW; ++x) {
                float sum = bias[c];
                for (int ky = 0; ky < K; ++ky) {
                    for (int kx = 0; kx < K; ++kx) {
                        const int iy = y * 2 - 1 + ky;
                        const int ix = x * 2 - 1 + kx;
                        if (iy < 0 || iy >= H || ix < 0 || ix >= W) {
                            continue;
                        }
                        sum += inputData[(c * H + iy) * W + ix] * weight[(c * K + ky) * K + kx];
                    }
                }
                expected[(c * outH + y) * outW + x] = sum;
            }
        }
    }
    report("depthwise 3x3 s2 pad1", actual, expected);
}

void testPooling() {
    printf("Pooling\n");
    const int N = 1, C = 2, H = 4, W = 4;
    const std::vector<float> inputData = makeInput(N * C * H * W);
    {
        Graph graph;
        const int input  = graph.addTensor("input", {N, C, H, W});
        const int output = graph.addTensor("output", {});
        graph.addOp(infer::OpType_Pooling, {input}, {output}, [](infer::OpAttr& attr) {
            attr.pool.type    = infer::PoolType_MAXPOOL;
            attr.pool.kernelX = 2;
            attr.pool.kernelY = 2;
            attr.pool.strideX = 2;
            attr.pool.strideY = 2;
            attr.pool.padType = infer::PoolPadType_VALID;
        });
        graph.setInputs({input});
        graph.setOutputs({output});
        Runner runner(graph);
        const std::vector<float> actual = runner.run(0, inputData);
        std::vector<float> expected(N * C * 2 * 2, 0.0f);
        for (int c = 0; c < C; ++c) {
            for (int y = 0; y < 2; ++y) {
                for (int x = 0; x < 2; ++x) {
                    float best = -1e30f;
                    for (int ky = 0; ky < 2; ++ky) {
                        for (int kx = 0; kx < 2; ++kx) {
                            best = std::max(best, inputData[(c * H + y * 2 + ky) * W + x * 2 + kx]);
                        }
                    }
                    expected[(c * 2 + y) * 2 + x] = best;
                }
            }
        }
        report("maxpool 2x2 s2", actual, expected);
    }
    {
        Graph graph;
        const int input  = graph.addTensor("input", {N, C, H, W});
        const int output = graph.addTensor("output", {});
        graph.addOp(infer::OpType_Pooling, {input}, {output}, [](infer::OpAttr& attr) {
            attr.pool.type     = infer::PoolType_AVEPOOL;
            attr.pool.isGlobal = true;
            attr.pool.kernelX  = W;
            attr.pool.kernelY  = H;
            attr.pool.padType  = infer::PoolPadType_VALID;
        });
        graph.setInputs({input});
        graph.setOutputs({output});
        Runner runner(graph);
        const std::vector<float> actual = runner.run(0, inputData);
        std::vector<float> expected(N * C, 0.0f);
        for (int c = 0; c < C; ++c) {
            float sum = 0.0f;
            for (int i = 0; i < H * W; ++i) {
                sum += inputData[c * H * W + i];
            }
            expected[c] = sum / (H * W);
        }
        report("global avgpool", actual, expected);
    }
}

void testSoftmax() {
    printf("Softmax\n");
    const int N = 1, C = 4, H = 1, W = 3;
    const std::vector<float> inputData = makeInput(N * C * H * W);
    Graph graph;
    const int input  = graph.addTensor("input", {N, C, H, W});
    const int output = graph.addTensor("output", {});
    graph.addOp(infer::OpType_Softmax, {input}, {output}, [](infer::OpAttr& attr) { attr.axis = 1; });
    graph.setInputs({input});
    graph.setOutputs({output});
    Runner runner(graph);
    const std::vector<float> actual = runner.run(0, inputData);
    std::vector<float> expected(inputData.size(), 0.0f);
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            float maxValue = -1e30f;
            for (int c = 0; c < C; ++c) {
                maxValue = std::max(maxValue, inputData[(c * H + h) * W + w]);
            }
            float sum = 0.0f;
            for (int c = 0; c < C; ++c) {
                const float e = std::exp(inputData[(c * H + h) * W + w] - maxValue);
                expected[(c * H + h) * W + w] = e;
                sum += e;
            }
            for (int c = 0; c < C; ++c) {
                expected[(c * H + h) * W + w] /= sum;
            }
        }
    }
    report("softmax axis=1", actual, expected);
}

void testBinaryBroadcast() {
    printf("BinaryOp broadcast\n");
    const int aCount = 3 * 1 * 1;
    const int bCount = 3 * 2 * 2;
    const std::vector<float> aData = makeInput(aCount);
    const std::vector<float> bData = makeInput(bCount);
    Graph graph;
    const int a      = graph.addTensor("a", {1, 3, 1, 1}, aData);
    const int b      = graph.addTensor("b", {1, 3, 2, 2}, bData);
    const int output = graph.addTensor("output", {});
    graph.addOp(infer::OpType_BinaryOp, {a, b}, {output}, [](infer::OpAttr& attr) {
        attr.binaryOp = infer::BinaryOpOperation_MUL;
    });
    graph.setInputs({a});
    graph.setOutputs({output});
    Runner runner(graph);
    const std::vector<float> actual = runner.run(0, aData);
    std::vector<float> expected(bCount, 0.0f);
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < 4; ++i) {
            expected[c * 4 + i] = aData[c] * bData[c * 4 + i];
        }
    }
    report("mul broadcast", actual, expected);
}

void testReduction() {
    printf("Reduction\n");
    const int N = 1, C = 3, H = 2, W = 2;
    const std::vector<float> inputData = makeInput(N * C * H * W);
    Graph graph;
    const int input  = graph.addTensor("input", {N, C, H, W});
    const int output = graph.addTensor("output", {});
    graph.addOp(infer::OpType_Reduction, {input}, {output}, [](infer::OpAttr& attr) {
        attr.reductionOp   = infer::ReductionType_MEAN;
        attr.reductionDims = {2, 3};
        attr.keepDims      = true;
    });
    graph.setInputs({input});
    graph.setOutputs({output});
    Runner runner(graph);
    const std::vector<float> actual = runner.run(0, inputData);
    std::vector<float> expected(C, 0.0f);
    for (int c = 0; c < C; ++c) {
        float sum = 0.0f;
        for (int i = 0; i < H * W; ++i) {
            sum += inputData[c * H * W + i];
        }
        expected[c] = sum / (H * W);
    }
    report("mean dims=[2,3] keepdims", actual, expected);
}

void testMatMulAndInnerProduct() {
    printf("MatMul / InnerProduct\n");
    const int M = 2, K = 3, N = 2;
    std::vector<float> a = makeInput(M * K);
    std::vector<float> b = makeInput(K * N);
    {
        Graph graph;
        const int ta     = graph.addTensor("a", {M, K}, a);
        const int tb     = graph.addTensor("b", {K, N}, b);
        const int output = graph.addTensor("output", {});
        graph.addOp(infer::OpType_MatMul, {ta, tb}, {output});
        graph.setInputs({ta});
        graph.setOutputs({output});
        Runner runner(graph);
        const std::vector<float> actual = runner.run(0, a);
        std::vector<float> expected(M * N, 0.0f);
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    sum += a[i * K + k] * b[k * N + j];
                }
                expected[i * N + j] = sum;
            }
        }
        report("matmul", actual, expected);
    }
    {
        const int O = 4;
        std::vector<float> weight(O * K);
        std::vector<float> bias(O);
        for (size_t i = 0; i < weight.size(); ++i) {
            weight[i] = std::cos(static_cast<float>(i) * 0.19f) * 0.3f;
        }
        for (int i = 0; i < O; ++i) {
            bias[i] = static_cast<float>(i) * 0.1f;
        }
        Graph graph;
        const int ta     = graph.addTensor("a", {M, K}, a);
        const int output = graph.addTensor("output", {});
        graph.addOp(infer::OpType_InnerProduct, {ta}, {output}, [&](infer::OpAttr& attr) {
            attr.outputCount = O;
            attr.weight      = weight;
            attr.bias        = bias;
        });
        graph.setInputs({ta});
        graph.setOutputs({output});
        Runner runner(graph);
        const std::vector<float> actual = runner.run(0, a);
        std::vector<float> expected(M * O, 0.0f);
        for (int i = 0; i < M; ++i) {
            for (int o = 0; o < O; ++o) {
                float sum = bias[o];
                for (int k = 0; k < K; ++k) {
                    sum += a[i * K + k] * weight[o * K + k];
                }
                expected[i * O + o] = sum;
            }
        }
        report("innerproduct", actual, expected);
    }
}

void testShapeOps() {
    printf("Shape ops\n");
    // Concat
    {
        const int A = 2, B = 3;
        std::vector<float> a = makeInput(1 * A * 2 * 2);
        std::vector<float> b = makeInput(1 * B * 2 * 2);
        Graph graph;
        const int ta     = graph.addTensor("a", {1, A, 2, 2}, a);
        const int tb     = graph.addTensor("b", {1, B, 2, 2}, b);
        const int output = graph.addTensor("output", {});
        graph.addOp(infer::OpType_Concat, {ta, tb}, {output}, [](infer::OpAttr& attr) { attr.axis = 1; });
        graph.setInputs({ta});
        graph.setOutputs({output});
        Runner runner(graph);
        const std::vector<float> actual = runner.run(0, a);
        std::vector<float> expected;
        expected.insert(expected.end(), a.begin(), a.end());
        expected.insert(expected.end(), b.begin(), b.end());
        report("concat axis=1", actual, expected);
    }
    // Permute
    {
        const int C = 2, H = 2, W = 3;
        const std::vector<float> data = makeInput(C * H * W);
        Graph graph;
        const int input  = graph.addTensor("input", {1, C, H, W}, data);
        const int output = graph.addTensor("output", {});
        graph.addOp(infer::OpType_Permute, {input}, {output}, [](infer::OpAttr& attr) {
            attr.permuteDims = {0, 2, 3, 1};
        });
        graph.setInputs({input});
        graph.setOutputs({output});
        Runner runner(graph);
        const std::vector<float> actual = runner.run(0, data);
        std::vector<float> expected(data.size(), 0.0f);
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    expected[((h * W + w) * C) + c] = data[(c * H + h) * W + w];
                }
            }
        }
        report("permute 0,2,3,1", actual, expected);
    }
    // StridedSlice
    {
        const int H = 4, W = 5;
        const std::vector<float> data = makeInput(H * W);
        std::vector<int> beginData = {0, 1, 1};
        std::vector<int> endData   = {1, 3, 4};
        std::vector<int> strideData = {1, 1, 1};
        Graph graph;
        const int input  = graph.addTensor("input", {1, H, W}, data);
        const int begin  = graph.addIntTensor("begin", {3}, beginData);
        const int end    = graph.addIntTensor("end", {3}, endData);
        const int stride = graph.addIntTensor("stride", {3}, strideData);
        const int output = graph.addTensor("output", {});
        graph.addOp(infer::OpType_StridedSlice, {input, begin, end, stride}, {output});
        graph.setInputs({input});
        graph.setOutputs({output});
        Runner runner(graph);
        const std::vector<float> actual = runner.run(0, data);
        std::vector<float> expected;
        for (int h = 1; h < 3; ++h) {
            for (int w = 1; w < 4; ++w) {
                expected.push_back(data[h * W + w]);
            }
        }
        report("stridedslice", actual, expected);
    }
    // GatherV2
    {
        const int H = 4, W = 3;
        const std::vector<float> data = makeInput(H * W);
        std::vector<int> indexData = {1, 3};
        Graph graph;
        const int input  = graph.addTensor("input", {1, H, W}, data);
        const int index  = graph.addIntTensor("index", {2}, indexData);
        const int output = graph.addTensor("output", {});
        graph.addOp(infer::OpType_GatherV2, {input, index}, {output}, [](infer::OpAttr& attr) { attr.axis = 1; });
        graph.setInputs({input});
        graph.setOutputs({output});
        Runner runner(graph);
        const std::vector<float> actual = runner.run(0, data);
        std::vector<float> expected;
        for (int w = 0; w < W; ++w) {
            expected.push_back(data[1 * W + w]);
        }
        for (int w = 0; w < W; ++w) {
            expected.push_back(data[3 * W + w]);
        }
        report("gatherv2", actual, expected);
    }
    // Pack
    {
        const std::vector<float> a = makeInput(2 * 3);
        const std::vector<float> b = makeInput(2 * 3);
        Graph graph;
        const int ta     = graph.addTensor("a", {2, 3}, a);
        const int tb     = graph.addTensor("b", {2, 3}, b);
        const int output = graph.addTensor("output", {});
        graph.addOp(infer::OpType_Pack, {ta, tb}, {output}, [](infer::OpAttr& attr) { attr.axis = 0; });
        graph.setInputs({ta});
        graph.setOutputs({output});
        Runner runner(graph);
        const std::vector<float> actual = runner.run(0, a);
        std::vector<float> expected;
        expected.insert(expected.end(), a.begin(), a.end());
        expected.insert(expected.end(), b.begin(), b.end());
        report("pack axis=0", actual, expected);
    }
    // Reshape + Flatten
    {
        const std::vector<float> data = makeInput(2 * 3 * 4);
        Graph graph;
        const int input  = graph.addTensor("input", {2, 3, 4}, data);
        const int output = graph.addTensor("output", {});
        graph.addOp(infer::OpType_Reshape, {input}, {output}, [](infer::OpAttr& attr) { attr.dims = {2, -1}; });
        graph.setInputs({input});
        graph.setOutputs({output});
        Runner runner(graph);
        const std::vector<float> actual = runner.run(0, data);
        report("reshape 2x-1", actual, data);
        CHECK(runner.session->output(0)->length(1) == 12, "reshape dim1==12");
    }
}

void testArgMaxAndInterpPadding() {
    printf("ArgMax / Interp / Padding\n");
    {
        const int C = 3, H = 2;
        const std::vector<float> data = {0.1f, 0.2f, 0.9f, 0.3f, 0.5f, 0.4f};
        Graph graph;
        const int input  = graph.addTensor("input", {1, C, H}, data);
        const int output = graph.addTensor("output", {});
        graph.addOp(infer::OpType_ArgMax, {input}, {output}, [](infer::OpAttr& attr) { attr.axis = 1; });
        graph.setInputs({input});
        graph.setOutputs({output});
        Runner runner(graph);
        if (runner.session && runner.session->run() == infer::NO_ERROR) {
            const int32_t* values = runner.session->output(0)->host<int32_t>();
            CHECK(values && values[0] == 1 && values[1] == 2, "argmax values");
            printf("  %-28s values=[%d,%d]\n", "argmax", values ? values[0] : -1, values ? values[1] : -1);
        }
    }
    {
        const std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
        Graph graph;
        const int input  = graph.addTensor("input", {1, 1, 2, 2}, data);
        const int output = graph.addTensor("output", {});
        graph.addOp(infer::OpType_Interp, {input}, {output}, [](infer::OpAttr& attr) {
            attr.resizeType  = 1;
            // widthScale only drives the output size; sampling is derived
            // from the actual input/output sizes.
            attr.widthScale  = 2.0f;
            attr.heightScale = 2.0f;
        });
        graph.setInputs({input});
        graph.setOutputs({output});
        Runner runner(graph);
        const std::vector<float> actual = runner.run(0, data);
        std::vector<float> expected = {1.0f, 1.0f, 2.0f, 2.0f, 1.0f, 1.0f, 2.0f, 2.0f,
                                       3.0f, 3.0f, 4.0f, 4.0f, 3.0f, 3.0f, 4.0f, 4.0f};
        report("interp nearest x2", actual, expected);
    }
    {
        const std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<int> pads = {0, 0, 0, 0, 1, 1, 1, 1};
        Graph graph;
        const int input  = graph.addTensor("input", {1, 1, 2, 2}, data);
        const int pad    = graph.addIntTensor("pads", {4, 2}, pads);
        const int output = graph.addTensor("output", {});
        graph.addOp(infer::OpType_Padding, {input, pad}, {output});
        graph.setInputs({input});
        graph.setOutputs({output});
        Runner runner(graph);
        const std::vector<float> actual = runner.run(0, data);
        std::vector<float> expected(4 * 4, 0.0f);
        expected[1 * 4 + 1] = 1.0f;
        expected[1 * 4 + 2] = 2.0f;
        expected[2 * 4 + 1] = 3.0f;
        expected[2 * 4 + 2] = 4.0f;
        report("padding constant", actual, expected);
    }
}

void testActivationAndScale() {
    printf("Activation / Scale\n");
    {
        const std::vector<float> data = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 7.0f};
        Graph graph;
        const int input  = graph.addTensor("input", {1, 2, 1, 3}, data);
        const int output = graph.addTensor("output", {});
        graph.addOp(infer::OpType_ReLU6, {input}, {output}, [](infer::OpAttr& attr) {
            attr.minValue = 0.0f;
            attr.maxValue = 6.0f;
        });
        graph.setInputs({input});
        graph.setOutputs({output});
        Runner runner(graph);
        const std::vector<float> actual = runner.run(0, data);
        const std::vector<float> expected = {0.0f, 0.0f, 0.0f, 0.5f, 1.0f, 6.0f};
        report("relu6", actual, expected);
    }
    {
        const std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        std::vector<float> scale      = {2.0f, 3.0f};
        std::vector<float> bias       = {0.5f, -0.5f};
        Graph graph;
        const int input  = graph.addTensor("input", {1, 2, 1, 3}, data);
        const int output = graph.addTensor("output", {});
        graph.addOp(infer::OpType_Scale, {input}, {output}, [&](infer::OpAttr& attr) {
            attr.scale    = scale;
            attr.offset   = bias;
            attr.channels = 2;
        });
        graph.setInputs({input});
        graph.setOutputs({output});
        Runner runner(graph);
        const std::vector<float> actual = runner.run(0, data);
        std::vector<float> expected(data.size());
        for (int c = 0; c < 2; ++c) {
            for (int i = 0; i < 3; ++i) {
                expected[c * 3 + i] = data[c * 3 + i] * scale[c] + bias[c];
            }
        }
        report("scale per channel", actual, expected);
    }
    {
        const std::vector<float> data = {-2.0f, -1.0f, 1.0f, 2.0f};
        std::vector<float> slope      = {0.1f, 0.2f};
        Graph graph;
        const int input  = graph.addTensor("input", {1, 2, 1, 2}, data);
        const int output = graph.addTensor("output", {});
        graph.addOp(infer::OpType_PReLU, {input}, {output}, [&](infer::OpAttr& attr) { attr.slope = slope; });
        graph.setInputs({input});
        graph.setOutputs({output});
        Runner runner(graph);
        const std::vector<float> actual = runner.run(0, data);
        const std::vector<float> expected = {-0.2f, -0.1f, 1.0f, 2.0f};
        report("prelu", actual, expected);
    }
}

} // namespace

int main() {
    testConvolution();
    testDepthwiseConvolution();
    testPooling();
    testSoftmax();
    testBinaryBroadcast();
    testReduction();
    testMatMulAndInnerProduct();
    testShapeOps();
    testArgMaxAndInterpPadding();
    testActivationAndScale();
    printf("\n%d checks, %d failures\n", gChecks, gFailures);
    return gFailures == 0 ? 0 : 1;
}
