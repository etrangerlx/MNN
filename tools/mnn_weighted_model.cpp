// Utility: inspect a MNN model and optionally write a copy with random weights.
//
// The benchmark models shipped with MNN only contain the graph structure (their
// convolution weights are empty). This tool fills every weight-bearing operator
// with deterministic pseudo random values so two engines can be compared on the
// exact same model.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "MNN_generated.h"
#include "flatbuffers/flatbuffers.h"

namespace {

struct Rng {
    uint32_t state;
    explicit Rng(uint32_t seed) : state(seed ? seed : 1u) {
    }
    float next() {
        state = state * 1664525u + 1013904223u;
        const int32_t v = static_cast<int32_t>(state);
        return static_cast<float>(v) / 2147483648.0f;
    }
    void fill(std::vector<float>& values, float scale) {
        for (auto& value : values) {
            value = next() * scale;
        }
    }
};

const char* opTypeName(MNN::OpType type) {
    return MNN::EnumNameOpType(type);
}

void printTensorDescribes(const flatbuffers::Vector<flatbuffers::Offset<MNN::TensorDescribe>>* describes, int limit) {
    if (!describes) {
        printf("  (no extraTensorDescribe)\n");
        return;
    }
    printf("  tensor describes: %d\n", static_cast<int>(describes->size()));
    for (size_t i = 0; i < describes->size() && static_cast<int>(i) < limit; ++i) {
        const auto* describe = describes->Get(i);
        const auto* blob     = describe->blob();
        printf("    [%d] %-24s type=%s dims=[", describe->index(), describe->name() ? describe->name()->str().c_str() : "",
               blob ? MNN::EnumNameDataType(blob->dataType()) : "?");
        if (blob && blob->dims()) {
            for (size_t d = 0; d < blob->dims()->size(); ++d) {
                printf("%s%d", d ? "," : "", blob->dims()->Get(d));
            }
        }
        printf("]");
        if (blob && blob->float32s()) {
            printf(" floats=%d", static_cast<int>(blob->float32s()->size()));
        }
        printf("\n");
    }
}

void printTensorDescribesT(const std::vector<std::unique_ptr<MNN::TensorDescribeT>>& describes, int limit) {
    if (describes.empty()) {
        printf("  (no extraTensorDescribe)\n");
        return;
    }
    printf("  tensor describes: %d\n", static_cast<int>(describes.size()));
    for (size_t i = 0; i < describes.size() && static_cast<int>(i) < limit; ++i) {
        const MNN::TensorDescribeT& describe = *describes[i];
        const MNN::BlobT* blob               = describe.blob.get();
        printf("    [%d] %-24s type=%s dims=[", describe.index, describe.name.c_str(),
               blob ? MNN::EnumNameDataType(blob->dataType) : "?");
        if (blob) {
            for (size_t d = 0; d < blob->dims.size(); ++d) {
                printf("%s%d", d ? "," : "", blob->dims[d]);
            }
        }
        printf("]");
        if (blob && !blob->float32s.empty()) {
            printf(" floats=%d", static_cast<int>(blob->float32s.size()));
        }
        printf("\n");
    }
}

void dumpGraph(const MNN::NetT& net) {
    if (!net.subgraphs.empty()) {
        printf("format: subgraphs (%d)\n", static_cast<int>(net.subgraphs.size()));
        for (size_t s = 0; s < net.subgraphs.size(); ++s) {
            const auto& subgraph = net.subgraphs[s];
            printf("== subgraph %d name=%s tensors=%d nodes=%d\n", static_cast<int>(s),
                   subgraph->name.c_str(), static_cast<int>(subgraph->tensors.size()),
                   static_cast<int>(subgraph->nodes.size()));
            printf("inputs=[");
            for (size_t i = 0; i < subgraph->inputs.size(); ++i) {
                printf("%s%d", i ? "," : "", subgraph->inputs[i]);
            }
            printf("] outputs=[");
            for (size_t i = 0; i < subgraph->outputs.size(); ++i) {
                printf("%s%d", i ? "," : "", subgraph->outputs[i]);
            }
            printf("]\n");
            if (!subgraph->extraTensorDescribe.empty()) {
                printTensorDescribesT(subgraph->extraTensorDescribe, 40);
            }
            for (size_t i = 0; i < subgraph->nodes.size(); ++i) {
                const auto& op = subgraph->nodes[i];
                printf("  %3d %-26s in=[", static_cast<int>(i), opTypeName(op->type));
                for (size_t k = 0; k < op->inputIndexes.size(); ++k) {
                    printf("%s%d", k ? "," : "", op->inputIndexes[k]);
                }
                printf("] out=[");
                for (size_t k = 0; k < op->outputIndexes.size(); ++k) {
                    printf("%s%d", k ? "," : "", op->outputIndexes[k]);
                }
                printf("]");
                if (op->main.type == MNN::OpParameter_Convolution2D) {
                    const auto* conv = op->main.AsConvolution2D();
                    printf(" conv{out=%d in=%d k=%dx%d s=%dx%d g=%d w=%d b=%d}", conv->common->outputCount,
                           conv->common->inputCount, conv->common->kernelX, conv->common->kernelY,
                           conv->common->strideX, conv->common->strideY, conv->common->group,
                           static_cast<int>(conv->weight.size()), static_cast<int>(conv->bias.size()));
                }
                printf("\n");
            }
        }
        return;
    }
    printf("format: legacy oplists (%d)\n", static_cast<int>(net.oplists.size()));
    printf("sourceType=%s\n", MNN::EnumNameNetSource(net.sourceType));
    printf("tensorName=%d outputName=%d\n", static_cast<int>(net.tensorName.size()),
           static_cast<int>(net.outputName.size()));
    if (!net.extraTensorDescribe.empty()) {
        printTensorDescribesT(net.extraTensorDescribe, 40);
    }
    for (size_t i = 0; i < net.oplists.size(); ++i) {
        const auto& op = net.oplists[i];
        printf("  %3d %-26s in=[", static_cast<int>(i), opTypeName(op->type));
        for (size_t k = 0; k < op->inputIndexes.size(); ++k) {
            printf("%s%d", k ? "," : "", op->inputIndexes[k]);
        }
        printf("] out=[");
        for (size_t k = 0; k < op->outputIndexes.size(); ++k) {
            printf("%s%d", k ? "," : "", op->outputIndexes[k]);
        }
        printf("]");
        if (op->main.type == MNN::OpParameter_Convolution2D) {
            const auto* conv = op->main.AsConvolution2D();
            printf(" conv{out=%d in=%d k=%dx%d s=%dx%d g=%d w=%d b=%d}", conv->common->outputCount,
                   conv->common->inputCount, conv->common->kernelX, conv->common->kernelY, conv->common->strideX,
                   conv->common->strideY, conv->common->group, static_cast<int>(conv->weight.size()),
                   static_cast<int>(conv->bias.size()));
        } else if (op->main.type == MNN::OpParameter_Input) {
            const auto* input = op->main.AsInput();
            printf(" input{dtype=%d dims=[", static_cast<int>(input->dtype));
            for (size_t d = 0; d < input->dims.size(); ++d) {
                printf("%s%d", d ? "," : "", input->dims[d]);
            }
            printf("]}");
        } else if (op->main.type == MNN::OpParameter_Reshape) {
            const auto* reshape = op->main.AsReshape();
            printf(" reshape{dimType=%d dims=[", static_cast<int>(reshape->dimType));
            for (size_t d = 0; d < reshape->dims.size(); ++d) {
                printf("%s%d", d ? "," : "", reshape->dims[d]);
            }
            printf("]}");
        } else if (op->main.type == MNN::OpParameter_Axis) {
            printf(" axis=%d", op->main.AsAxis()->axis);
        } else if (op->main.type == MNN::OpParameter_SqueezeParam) {
            const auto* squeeze = op->main.AsSqueezeParam();
            printf(" squeeze{ndim=%d dims=[", static_cast<int>(squeeze->squeezeDims.size()));
            for (size_t d = 0; d < squeeze->squeezeDims.size(); ++d) {
                printf("%s%d", d ? "," : "", squeeze->squeezeDims[d]);
            }
            printf("]}");
        } else if (op->main.type == MNN::OpParameter_Blob) {
            const auto* blob = op->main.AsBlob();
            printf(" const{type=%s count=%d", MNN::EnumNameDataType(blob->dataType),
                   static_cast<int>(blob->int32s.size() + blob->float32s.size() + blob->int8s.size()));
            if (!blob->int32s.empty()) {
                printf(" ints=[");
                for (size_t k = 0; k < blob->int32s.size() && k < 8; ++k) {
                    printf("%s%d", k ? "," : "", blob->int32s[k]);
                }
                printf("]");
            }
            printf("}");
        } else if (op->main.type == MNN::OpParameter_Interp) {
            const auto* interp = op->main.AsInterp();
            printf(" interp{type=%d out=%dx%d scale=%.4fx%.4f offset=%.4fx%.4f align=%d half=%d}",
                   interp->resizeType, interp->outputWidth, interp->outputHeight,
                   interp->widthScale, interp->heightScale, interp->widthOffset, interp->heightOffset,
                   static_cast<int>(interp->alignCorners), static_cast<int>(interp->halfPixelCenters));
        } else if (op->main.type == MNN::OpParameter_TensorConvertInfo) {
            const auto* info = op->main.AsTensorConvertInfo();
            printf(" convert{src=%d dst=%d}", static_cast<int>(info->source), static_cast<int>(info->dest));
        } else if (op->main.type == MNN::OpParameter_ReductionParam) {
            const auto* reduction = op->main.AsReductionParam();
            printf(" reduction{op=%d keepDims=%d dims=[", static_cast<int>(reduction->operation),
                   static_cast<int>(reduction->keepDims));
            for (size_t d = 0; d < reduction->dim.size(); ++d) {
                printf("%s%d", d ? "," : "", reduction->dim[d]);
            }
            printf("]}");
        } else if (op->main.type == MNN::OpParameter_Pool) {
            const auto* pool = op->main.AsPool();
            printf(" pool{type=%d k=%dx%d s=%dx%d padType=%d pads=[", static_cast<int>(pool->type), pool->kernelX,
                   pool->kernelY, pool->strideX, pool->strideY, static_cast<int>(pool->padType));
            for (size_t d = 0; d < pool->pads.size(); ++d) {
                printf("%s%d", d ? "," : "", pool->pads[d]);
            }
            printf("] global=%d ceil=%d}", static_cast<int>(pool->isGlobal), static_cast<int>(pool->ceilModel));
        }
        printf("\n");
    }
}

int fillWeights(MNN::NetT& net, uint32_t seed) {
    Rng rng(seed);
    int filled = 0;
    auto processOp = [&](std::unique_ptr<MNN::OpT>& op) {
        if (op->main.type == MNN::OpParameter_Convolution2D) {
            auto* conv = op->main.AsConvolution2D();
            if (!conv || !conv->common) {
                return;
            }
            const int outputCount = conv->common->outputCount;
            const int kernelX     = conv->common->kernelX;
            const int kernelY     = conv->common->kernelY;
            int group             = conv->common->group > 0 ? conv->common->group : 1;
            if (outputCount <= 0 || kernelX <= 0 || kernelY <= 0) {
                return;
            }
            int weightSize = 0;
            int fanIn      = 0;
            if (op->type == MNN::OpType_ConvolutionDepthwise) {
                weightSize = outputCount * kernelX * kernelY;
                fanIn      = kernelX * kernelY;
            } else {
                int inputCount = conv->common->inputCount;
                if (inputCount <= 0) {
                    inputCount = conv->weight.empty() ? 0 : static_cast<int>(conv->weight.size()) /
                                                                 (outputCount / group * kernelX * kernelY);
                }
                if (inputCount <= 0) {
                    fprintf(stderr, "skip conv with unknown inputCount (op %s)\n", op->name.c_str());
                    return;
                }
                weightSize = outputCount * (inputCount / group) * kernelX * kernelY;
                fanIn      = (inputCount / group) * kernelX * kernelY;
            }
            conv->weight.assign(weightSize, 0.0f);
            // Xavier-like scaling keeps activations bounded across deep stacks,
            // so the final softmax stays informative.
            rng.fill(conv->weight, std::sqrt(1.5f / std::max(1, fanIn)));
            conv->bias.assign(outputCount, 0.0f);
            rng.fill(conv->bias, 0.05f);
            ++filled;
            return;
        }
        if (op->main.type == MNN::OpParameter_InnerProduct) {
            auto* ip = op->main.AsInnerProduct();
            if (!ip || ip->outputCount <= 0) {
                return;
            }
            int inputCount = ip->weightSize / std::max(1, ip->outputCount);
            if (inputCount <= 0) {
                inputCount = static_cast<int>(ip->weight.size()) / std::max(1, ip->outputCount);
            }
            if (inputCount <= 0) {
                return;
            }
            ip->weight.assign(static_cast<size_t>(inputCount) * ip->outputCount, 0.0f);
            rng.fill(ip->weight, std::sqrt(1.5f / std::max(1, inputCount)));
            ip->bias.assign(ip->outputCount, 0.0f);
            rng.fill(ip->bias, 0.05f);
            ++filled;
            return;
        }
        if (op->main.type == MNN::OpParameter_Scale) {
            auto* scale = op->main.AsScale();
            if (!scale || scale->channels <= 0) {
                return;
            }
            if (scale->scaleData.empty()) {
                scale->scaleData.assign(scale->channels, 1.0f);
                rng.fill(scale->scaleData, 0.2f);
                for (auto& v : scale->scaleData) {
                    v += 1.0f;
                }
            }
            if (scale->biasData.empty()) {
                scale->biasData.assign(scale->channels, 0.0f);
                rng.fill(scale->biasData, 0.1f);
            }
            ++filled;
            return;
        }
        if (op->main.type == MNN::OpParameter_PRelu) {
            auto* prelu = op->main.AsPRelu();
            if (prelu && prelu->slope.empty() && prelu->slopeCount > 0) {
                prelu->slope.assign(prelu->slopeCount, 0.25f);
                rng.fill(prelu->slope, 0.1f);
                for (auto& v : prelu->slope) {
                    v += 0.2f;
                }
                ++filled;
            }
        }
    };

    for (auto& op : net.oplists) {
        processOp(op);
    }
    for (auto& subgraph : net.subgraphs) {
        for (auto& op : subgraph->nodes) {
            processOp(op);
        }
    }
    return filled;
}

bool readFile(const std::string& path, std::vector<uint8_t>& data) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        return false;
    }
    const std::streamsize size = file.tellg();
    if (size <= 0) {
        return false;
    }
    data.resize(static_cast<size_t>(size));
    file.seekg(0, std::ios::beg);
    return static_cast<bool>(file.read(reinterpret_cast<char*>(data.data()), size));
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr,
                "usage: %s <input.mnn> [output.mnn] [--dump] [--no-fill] [--seed N] [--outputs name1,name2]\n"
                "  without output.mnn the model is only inspected\n",
                argv[0]);
        return 1;
    }
    const std::string inputPath = argv[1];
    std::string outputPath;
    bool dump    = false;
    bool fill    = true;
    uint32_t seed = 12345;
    std::string outputs;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--dump") {
            dump = true;
        } else if (arg == "--no-fill") {
            fill = false;
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (arg == "--outputs" && i + 1 < argc) {
            outputs = argv[++i];
        } else if (!arg.empty() && arg[0] != '-') {
            outputPath = arg;
        }
    }

    std::vector<uint8_t> data;
    if (!readFile(inputPath, data)) {
        fprintf(stderr, "failed to read %s\n", inputPath.c_str());
        return 1;
    }
    auto* net = flatbuffers::GetRoot<MNN::Net>(data.data());
    if (!net) {
        fprintf(stderr, "not a MNN model\n");
        return 1;
    }
    std::unique_ptr<MNN::NetT> netT(net->UnPack());
    if (dump) {
        dumpGraph(*netT);
    }
    const int filled = fill ? fillWeights(*netT, seed) : 0;
    printf("filled %d weight tensors\n", filled);

    if (!outputs.empty()) {
        netT->outputName.clear();
        size_t start = 0;
        while (start <= outputs.size()) {
            const size_t comma = outputs.find(',', start);
            const std::string name =
                outputs.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
            if (!name.empty()) {
                netT->outputName.push_back(name);
            }
            if (comma == std::string::npos) {
                break;
            }
            start = comma + 1;
        }
    }

    if (outputPath.empty()) {
        return 0;
    }
    flatbuffers::FlatBufferBuilder builder(1024);
    auto offset = MNN::Net::Pack(builder, netT.get());
    builder.Finish(offset);
    std::ofstream out(outputPath, std::ios::binary);
    out.write(reinterpret_cast<const char*>(builder.GetBufferPointer()), builder.GetSize());
    printf("wrote %s (%d bytes)\n", outputPath.c_str(), static_cast<int>(builder.GetSize()));
    return 0;
}
