#include "infer/ModelLoader.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <set>
#include <unordered_map>

#include "MNN_generated.h"
#include "flatbuffers/flatbuffers.h"

namespace infer {
namespace {

bool isSupportedOp(int t) {
    switch (static_cast<OpType>(t)) {
        case OpType_ArgMax:
        case OpType_ArgMin:
        case OpType_BinaryOp:
        case OpType_Cast:
        case OpType_Concat:
        case OpType_Const:
        case OpType_Convolution:
        case OpType_ConvolutionDepthwise:
        case OpType_Eltwise:
        case OpType_Flatten:
        case OpType_GatherV2:
        case OpType_InnerProduct:
        case OpType_Input:
        case OpType_Interp:
        case OpType_MatMul:
        case OpType_Pack:
        case OpType_Padding:
        case OpType_Permute:
        case OpType_Pooling:
        case OpType_PReLU:
        case OpType_Reduction:
        case OpType_ReLU:
        case OpType_ReLU6:
        case OpType_Reshape:
        case OpType_Scale:
        case OpType_Shape:
        case OpType_SliceTf:
        case OpType_Softmax:
        case OpType_Squeeze:
        case OpType_StridedSlice:
        case OpType_Transpose:
        case OpType_UnaryOp:
        case OpType_Unpack:
        case OpType_ConvertTensor:
        case OpType_Identity:
            return true;
        default:
            return false;
    }
}

template <typename T>
std::vector<int> copyIntVector(const flatbuffers::Vector<T>* v) {
    std::vector<int> out;
    if (!v) {
        return out;
    }
    out.reserve(v->size());
    for (size_t i = 0; i < v->size(); ++i) {
        out.push_back(static_cast<int>(v->Get(i)));
    }
    return out;
}

template <typename T>
std::vector<float> copyFloatVector(const flatbuffers::Vector<T>* v) {
    std::vector<float> out;
    if (!v) {
        return out;
    }
    out.reserve(v->size());
    for (size_t i = 0; i < v->size(); ++i) {
        out.push_back(static_cast<float>(v->Get(i)));
    }
    return out;
}

// Copies the raw content of a MNN Blob into a byte buffer.
bool blobData(const MNN::Blob* blob, DataType type, std::vector<uint8_t>& out) {
    if (!blob) {
        return false;
    }
    if (type == DT_FLOAT && blob->float32s()) {
        const size_t n = blob->float32s()->size();
        out.resize(n * sizeof(float));
        std::memcpy(out.data(), blob->float32s()->data(), out.size());
        return n > 0;
    }
    if (type == DT_INT32 && blob->int32s()) {
        const size_t n = blob->int32s()->size();
        out.resize(n * sizeof(int32_t));
        std::memcpy(out.data(), blob->int32s()->data(), out.size());
        return n > 0;
    }
    if (type == DT_INT64 && blob->int64s()) {
        const size_t n = blob->int64s()->size();
        out.resize(n * sizeof(int64_t));
        std::memcpy(out.data(), blob->int64s()->data(), out.size());
        return n > 0;
    }
    if (type == DT_INT8 && blob->int8s()) {
        const size_t n = blob->int8s()->size();
        out.resize(n);
        std::memcpy(out.data(), blob->int8s()->data(), n);
        return n > 0;
    }
    if (blob->uint8s()) {
        const size_t n = blob->uint8s()->size();
        out.resize(n);
        std::memcpy(out.data(), blob->uint8s()->data(), n);
        return n > 0;
    }
    return false;
}

void parseConvCommon(const MNN::Convolution2DCommon* common, Conv2DCommon& out) {
    if (!common) {
        return;
    }
    out.padX        = common->padX();
    out.padY        = common->padY();
    out.kernelX     = common->kernelX();
    out.kernelY     = common->kernelY();
    out.strideX     = common->strideX();
    out.strideY     = common->strideY();
    out.dilateX     = common->dilateX();
    out.dilateY     = common->dilateY();
    out.padMode     = common->padMode();
    out.group       = common->group() > 0 ? common->group() : 1;
    out.outputCount = common->outputCount();
    out.inputCount  = common->inputCount();
    out.relu        = common->relu();
    out.relu6       = common->relu6();
    out.pads        = copyIntVector(common->pads());
}

void parsePool(const MNN::Pool* pool, PoolCommon& out) {
    out.padX      = pool->padX();
    out.padY      = pool->padY();
    out.kernelX   = pool->kernelX();
    out.kernelY   = pool->kernelY();
    out.strideX   = pool->strideX() > 0 ? pool->strideX() : 1;
    out.strideY   = pool->strideY() > 0 ? pool->strideY() : 1;
    out.isGlobal  = pool->isGlobal();
    out.ceilModel = pool->ceilModel();
    out.type      = pool->type();
    out.padType   = pool->padType();
    out.countType = pool->countType();
    out.pads      = copyIntVector(pool->pads());
    if (out.strideX <= 0) {
        out.strideX = 1;
    }
    if (out.strideY <= 0) {
        out.strideY = 1;
    }
}

ErrorCode parseOp(const MNN::Op* op, OpDesc& desc) {
    desc.name = op->name() ? op->name()->str() : "";
    desc.inputs  = copyIntVector(op->inputIndexes());
    desc.outputs = copyIntVector(op->outputIndexes());
    const int rawType = static_cast<int>(op->type());
    if (!isSupportedOp(rawType)) {
        return NOT_SUPPORT;
    }
    desc.type = static_cast<OpType>(rawType);
    OpAttr& attr = desc.attr;

    switch (desc.type) {
        case OpType_Convolution:
        case OpType_ConvolutionDepthwise: {
            const auto* conv = op->main_as_Convolution2D();
            if (!conv) {
                return INVALID_INPUT;
            }
            parseConvCommon(conv->common(), attr.conv);
            attr.weight = copyFloatVector(conv->weight());
            attr.bias   = copyFloatVector(conv->bias());
            attr.conv.outputCount = attr.conv.outputCount > 0 ? attr.conv.outputCount
                                                             : static_cast<int>(attr.bias.size());
            break;
        }
        case OpType_Pooling: {
            const auto* pool = op->main_as_Pool();
            if (!pool) {
                return INVALID_INPUT;
            }
            parsePool(pool, attr.pool);
            break;
        }
        case OpType_Concat: {
            const auto* axis = op->main_as_Axis();
            attr.axis        = axis ? axis->axis() : 0;
            break;
        }
        case OpType_Softmax: {
            const auto* axis = op->main_as_Axis();
            attr.axis        = axis ? axis->axis() : 1;
            break;
        }
        case OpType_Unpack: {
            const auto* axis = op->main_as_Axis();
            attr.axis        = axis ? axis->axis() : 0;
            break;
        }
        case OpType_Transpose:
        case OpType_SliceTf:
            // Permutation / slice bounds arrive as input tensors.
            break;
        case OpType_Input: {
            const auto* input = op->main_as_Input();
            if (input) {
                attr.dims     = copyIntVector(input->dims());
                attr.dataType = static_cast<DataType>(input->dtype());
            }
            break;
        }
        case OpType_ConvertTensor: {
            const auto* info = op->main_as_TensorConvertInfo();
            if (info) {
                attr.srcFormat = static_cast<int>(info->source());
                attr.dstFormat = static_cast<int>(info->dest());
            }
            break;
        }
        case OpType_Reshape: {
            const auto* reshape = op->main_as_Reshape();
            if (reshape) {
                attr.dims = copyIntVector(reshape->dims());
                attr.dataType = static_cast<DataType>(reshape->dimType());
            }
            break;
        }
        case OpType_Squeeze: {
            const auto* squeeze = op->main_as_SqueezeParam();
            if (squeeze) {
                attr.squeezeDims = copyIntVector(squeeze->squeezeDims());
            }
            break;
        }
        case OpType_Flatten: {
            const auto* flatten = op->main_as_Flatten();
            if (flatten) {
                attr.axis  = flatten->axis();
                attr.axis2 = flatten->endAxis();
            }
            break;
        }
        case OpType_Permute: {
            const auto* permute = op->main_as_Permute();
            if (permute) {
                attr.permuteDims = copyIntVector(permute->dims());
            }
            break;
        }
        case OpType_BinaryOp: {
            const auto* binary = op->main_as_BinaryOp();
            if (binary) {
                attr.binaryOp = binary->opType();
            }
            break;
        }
        case OpType_UnaryOp: {
            const auto* unary = op->main_as_UnaryOp();
            if (unary) {
                attr.unaryOp = unary->opType();
            }
            break;
        }
        case OpType_Reduction: {
            const auto* reduction = op->main_as_ReductionParam();
            if (reduction) {
                attr.reductionOp   = reduction->operation();
                attr.reductionDims = copyIntVector(reduction->dim());
                attr.keepDims      = reduction->keepDims();
            }
            break;
        }
        case OpType_Eltwise: {
            const auto* eltwise = op->main_as_Eltwise();
            if (eltwise) {
                attr.eltwiseType = eltwise->type();
                attr.coeff       = copyFloatVector(eltwise->coeff());
            }
            break;
        }
        case OpType_ReLU: {
            const auto* relu = op->main_as_Relu();
            attr.slopeValue  = relu ? relu->slope() : 0.0f;
            break;
        }
        case OpType_ReLU6: {
            const auto* relu6 = op->main_as_Relu6();
            attr.minValue     = relu6 ? relu6->minValue() : 0.0f;
            attr.maxValue     = relu6 ? relu6->maxValue() : 6.0f;
            break;
        }
        case OpType_PReLU: {
            const auto* prelu = op->main_as_PRelu();
            if (prelu) {
                attr.slope = copyFloatVector(prelu->slope());
            }
            break;
        }
        case OpType_Scale: {
            const auto* scale = op->main_as_Scale();
            if (scale) {
                attr.scale   = copyFloatVector(scale->scaleData());
                attr.offset  = copyFloatVector(scale->biasData());
                attr.channels = scale->channels();
            }
            break;
        }
        case OpType_InnerProduct: {
            const auto* ip = op->main_as_InnerProduct();
            if (!ip) {
                return INVALID_INPUT;
            }
            attr.weight       = copyFloatVector(ip->weight());
            attr.bias         = copyFloatVector(ip->bias());
            attr.axis         = ip->axis();
            attr.transposeB   = ip->transpose();
            attr.outputCount  = ip->outputCount();
            break;
        }
        case OpType_MatMul: {
            const auto* mm = op->main_as_MatMul();
            if (mm) {
                attr.transposeA = mm->transposeA();
                attr.transposeB = mm->transposeB();
            }
            break;
        }
        case OpType_Cast: {
            const auto* cast = op->main_as_CastParam();
            if (cast) {
                attr.dataType = static_cast<DataType>(cast->dstT());
            }
            break;
        }
        case OpType_ArgMax:
        case OpType_ArgMin: {
            const auto* argmax = op->main_as_ArgMax();
            if (argmax) {
                attr.axis      = argmax->axis();
                attr.topK      = argmax->topK();
                attr.outMaxVal = argmax->outMaxVal();
            }
            break;
        }
        case OpType_Padding: {
            const auto* padding = op->main_as_PadParam();
            if (padding) {
                attr.padMode = padding->mode();
            }
            break;
        }
        case OpType_Interp: {
            const auto* interp = op->main_as_Interp();
            if (interp) {
                attr.resizeType      = interp->resizeType();
                attr.widthScale      = interp->widthScale();
                attr.heightScale     = interp->heightScale();
                attr.widthOffset     = interp->widthOffset();
                attr.heightOffset    = interp->heightOffset();
                attr.alignCorners    = interp->alignCorners();
                attr.halfPixelCenters = interp->halfPixelCenters();
                attr.outputWidth     = interp->outputWidth();
                attr.outputHeight    = interp->outputHeight();
            }
            break;
        }
        case OpType_StridedSlice: {
            const auto* ss = op->main_as_StridedSliceParam();
            if (ss) {
                attr.beginMask     = ss->beginMask();
                attr.endMask       = ss->endMask();
                attr.ellipsisMask  = ss->ellipsisMask();
                attr.newAxisMask   = ss->newAxisMask();
                attr.shrinkAxisMask = ss->shrinkAxisMask();
            }
            break;
        }
        case OpType_Pack: {
            const auto* pack = op->main_as_PackParam();
            if (pack) {
                attr.axis = pack->axis();
            }
            break;
        }
        case OpType_Shape: {
            const auto* shape = op->main_as_ShapeParam();
            if (shape) {
                attr.hasStart = shape->hasStart();
                attr.hasEnd   = shape->hasEnd();
                attr.start    = shape->start();
                attr.end      = shape->end();
            }
            break;
        }
        case OpType_Const: {
            // Data is carried by the output tensor blob; see below.
            break;
        }
        case OpType_GatherV2: {
            // axis comes from the third input at runtime, nothing to parse.
            break;
        }
        default:
            break;
    }
    return NO_ERROR;
}

// Reads a constant blob attached to the operator itself (Const op).
void readConstOpBlob(const MNN::Op* op, OpDesc& desc, std::vector<TensorDesc>& tensors) {
    if (op->main_type() != MNN::OpParameter_Blob || desc.outputs.empty()) {
        return;
    }
    const auto* blob = op->main_as_Blob();
    if (!blob) {
        return;
    }
    TensorDesc& tensor = tensors[desc.outputs[0]];
    if (blob->dims()) {
        tensor.dims = copyIntVector(blob->dims());
    }
    if (blob->dataType() != MNN::DataType_DT_INVALID) {
        tensor.type = static_cast<DataType>(blob->dataType());
    }
    std::vector<uint8_t> raw;
    if (blobData(blob, tensor.type, raw)) {
        tensor.data    = std::make_shared<std::vector<uint8_t>>(std::move(raw));
        tensor.hasData = true;
    }
}

void readConvWeightsFromTensors(const std::vector<TensorDesc>& tensors, OpDesc& desc) {
    OpAttr& attr = desc.attr;
    if (!attr.weight.empty() || desc.inputs.size() < 2) {
        return;
    }
    const int weightIndex = desc.inputs[1];
    if (weightIndex < 0 || weightIndex >= static_cast<int>(tensors.size())) {
        return;
    }
    const TensorDesc& weightTensor = tensors[weightIndex];
    if (!weightTensor.hasData || !weightTensor.data) {
        return;
    }
    const size_t count = weightTensor.data->size() / sizeof(float);
    const float* src    = reinterpret_cast<const float*>(weightTensor.data->data());
    attr.weight.assign(src, src + count);
    if (desc.inputs.size() >= 3) {
        const int biasIndex = desc.inputs[2];
        if (biasIndex >= 0 && biasIndex < static_cast<int>(tensors.size()) && tensors[biasIndex].hasData &&
            tensors[biasIndex].data) {
            const size_t biasCount = tensors[biasIndex].data->size() / sizeof(float);
            const float* biasSrc   = reinterpret_cast<const float*>(tensors[biasIndex].data->data());
            attr.bias.assign(biasSrc, biasSrc + biasCount);
            desc.inputs.resize(1);
        }
    } else {
        desc.inputs.resize(1);
    }
}

} // namespace

ErrorCode loadModelFromBuffer(const void* buffer, size_t size, std::shared_ptr<Model>* outModel) {
    if (!buffer || size < 8) {
        return INVALID_INPUT;
    }
    auto model  = std::make_shared<Model>();
    auto copied = std::make_shared<std::vector<uint8_t>>(size);
    std::memcpy(copied->data(), buffer, size);
    model->buffer = copied;

    auto* net = flatbuffers::GetRoot<MNN::Net>(copied->data());
    if (!net) {
        return INVALID_INPUT;
    }

    const flatbuffers::Vector<flatbuffers::Offset<MNN::TensorDescribe>>* describes = nullptr;
    const flatbuffers::Vector<flatbuffers::Offset<MNN::Op>>* ops                   = nullptr;
    std::vector<std::string> tensorNames;
    std::vector<int> graphInputs;
    std::vector<int> graphOutputs;

    if (net->subgraphs() && net->subgraphs()->size() > 0) {
        const auto* subgraph = net->subgraphs()->Get(0);
        if (subgraph->tensors()) {
            tensorNames.reserve(subgraph->tensors()->size());
            for (size_t i = 0; i < subgraph->tensors()->size(); ++i) {
                tensorNames.push_back(subgraph->tensors()->Get(i)->str());
            }
        }
        ops = subgraph->nodes();
        describes = subgraph->extraTensorDescribe() ? subgraph->extraTensorDescribe() : net->extraTensorDescribe();
        graphInputs  = copyIntVector(subgraph->inputs());
        graphOutputs = copyIntVector(subgraph->outputs());
    } else {
        if (net->tensorName()) {
            tensorNames.reserve(net->tensorName()->size());
            for (size_t i = 0; i < net->tensorName()->size(); ++i) {
                tensorNames.push_back(net->tensorName()->Get(i)->str());
            }
        }
        ops      = net->oplists();
        describes = net->extraTensorDescribe();
        if (net->outputName()) {
            for (size_t i = 0; i < net->outputName()->size(); ++i) {
                const std::string name = net->outputName()->Get(i)->str();
                for (size_t t = 0; t < tensorNames.size(); ++t) {
                    if (tensorNames[t] == name) {
                        graphOutputs.push_back(static_cast<int>(t));
                        break;
                    }
                }
            }
        }
    }
    if (!ops) {
        return INVALID_INPUT;
    }

    int tensorCount = static_cast<int>(tensorNames.size());
    if (describes) {
        for (size_t i = 0; i < describes->size(); ++i) {
            tensorCount = std::max(tensorCount, describes->Get(i)->index() + 1);
        }
    }
    if (net->tensorNumber() > tensorCount) {
        tensorCount = net->tensorNumber();
    }
    if (tensorCount <= 0) {
        return INVALID_INPUT;
    }

    model->tensors.resize(tensorCount);
    for (int i = 0; i < tensorCount; ++i) {
        model->tensors[i].name = i < static_cast<int>(tensorNames.size()) ? tensorNames[i] : "";
    }

    // Tensor descriptions carry dims/data types and constant blobs.
    if (describes) {
        for (size_t i = 0; i < describes->size(); ++i) {
            const auto* describe = describes->Get(i);
            const int index      = describe->index();
            if (index < 0 || index >= tensorCount) {
                continue;
            }
            TensorDesc& tensor = model->tensors[index];
            if (describe->name()) {
                tensor.name = describe->name()->str();
            }
            const auto* blob = describe->blob();
            if (!blob) {
                continue;
            }
            if (blob->dims()) {
                tensor.dims = copyIntVector(blob->dims());
            }
            if (blob->dataType() != MNN::DataType_DT_INVALID) {
                tensor.type = static_cast<DataType>(blob->dataType());
            }
            std::vector<uint8_t> raw;
            if (blobData(blob, tensor.type, raw)) {
                tensor.data    = std::make_shared<std::vector<uint8_t>>(std::move(raw));
                tensor.hasData = true;
            }
        }
    }

    // Parse operators, checking support while doing so.
    model->ops.reserve(ops->size());
    for (size_t i = 0; i < ops->size(); ++i) {
        const auto* op = ops->Get(i);
        OpDesc desc;
        const ErrorCode code = parseOp(op, desc);
        if (code != NO_ERROR) {
            return code;
        }
        readConstOpBlob(op, desc, model->tensors);
        if (desc.type == OpType_Convolution || desc.type == OpType_ConvolutionDepthwise) {
            readConvWeightsFromTensors(model->tensors, desc);
        }
        if (desc.type == OpType_Input) {
            // Legacy models carry the graph input shape on the Input op. A batch
            // dimension of -1 means "any", which the framework fixes to 1.
            for (int t : desc.outputs) {
                if (t < 0 || t >= tensorCount) {
                    continue;
                }
                TensorDesc& tensor = model->tensors[t];
                tensor.dims        = desc.attr.dims;
                for (int& d : tensor.dims) {
                    if (d < 0) {
                        d = 1;
                    }
                }
                if (desc.attr.dataType != DT_INVALID) {
                    tensor.type = desc.attr.dataType;
                }
            }
        }
        model->ops.emplace_back(std::move(desc));
    }

    // Legacy benchmark models ship without convolution weights; they are only
    // usable for latency measurements, not accuracy comparison.
    int weightlessConv = 0;
    for (const OpDesc& op : model->ops) {
        if ((op.type == OpType_Convolution || op.type == OpType_ConvolutionDepthwise) &&
            op.attr.weight.empty()) {
            ++weightlessConv;
        }
    }
    if (weightlessConv > 0) {
        fprintf(stderr, "infer: %d convolution op(s) without weights (benchmark-only model)\n",
                weightlessConv);
    }

    // Topological sort (Kahn) over tensors.
    std::vector<std::vector<int>> consumers(tensorCount);
    std::vector<int> producer(tensorCount, -1);
    for (size_t i = 0; i < model->ops.size(); ++i) {
        const OpDesc& op = model->ops[i];
        for (int t : op.inputs) {
            if (t >= 0 && t < tensorCount) {
                consumers[t].push_back(static_cast<int>(i));
            }
        }
        for (int t : op.outputs) {
            if (t >= 0 && t < tensorCount) {
                producer[t] = static_cast<int>(i);
            }
        }
    }
    std::vector<int> inDegree(model->ops.size(), 0);
    for (size_t i = 0; i < model->ops.size(); ++i) {
        std::set<int> deps;
        for (int t : model->ops[i].inputs) {
            if (t >= 0 && t < tensorCount && producer[t] >= 0) {
                deps.insert(producer[t]);
            }
        }
        inDegree[i] = static_cast<int>(deps.size());
    }

    std::vector<std::vector<int>> edges(model->ops.size());
    {
        std::vector<std::set<int>> depsSet(model->ops.size());
        for (size_t i = 0; i < model->ops.size(); ++i) {
            for (int t : model->ops[i].inputs) {
                if (t >= 0 && t < tensorCount && producer[t] >= 0) {
                    depsSet[i].insert(producer[t]);
                }
            }
            for (int p : depsSet[i]) {
                edges[p].push_back(static_cast<int>(i));
            }
        }
    }
    std::vector<int> order;
    order.reserve(model->ops.size());
    std::vector<int> queue;
    for (size_t i = 0; i < model->ops.size(); ++i) {
        if (inDegree[i] == 0) {
            queue.push_back(static_cast<int>(i));
        }
    }
    for (size_t head = 0; head < queue.size(); ++head) {
        const int current = queue[head];
        order.push_back(current);
        for (int next : edges[current]) {
            if (--inDegree[next] == 0) {
                queue.push_back(next);
            }
        }
    }
    if (order.size() != model->ops.size()) {
        // Fall back to the original order if the graph is malformed.
        order.clear();
        for (size_t i = 0; i < model->ops.size(); ++i) {
            order.push_back(static_cast<int>(i));
        }
    }
    std::vector<OpDesc> sortedOps;
    sortedOps.reserve(model->ops.size());
    for (int index : order) {
        sortedOps.emplace_back(std::move(model->ops[index]));
    }
    model->ops = std::move(sortedOps);

    // Model inputs: tensors consumed but never produced (and not constants).
    std::set<int> produced;
    for (const OpDesc& op : model->ops) {
        for (int t : op.outputs) {
            produced.insert(t);
        }
    }
    if (!graphInputs.empty()) {
        model->inputIndices = graphInputs;
    } else {
        std::set<int> inputSet;
        for (const OpDesc& op : model->ops) {
            for (int t : op.inputs) {
                if (t >= 0 && t < tensorCount && produced.count(t) == 0 && !model->tensors[t].hasData) {
                    inputSet.insert(t);
                }
            }
        }
        model->inputIndices.assign(inputSet.begin(), inputSet.end());
    }
    // Input ops also declare graph inputs and carry their shapes.
    for (const OpDesc& op : model->ops) {
        if (op.type == OpType_Input) {
            for (int t : op.outputs) {
                auto it = std::find(model->inputIndices.begin(), model->inputIndices.end(), t);
                if (it == model->inputIndices.end()) {
                    model->inputIndices.push_back(t);
                }
            }
        }
    }

    if (!graphOutputs.empty()) {
        model->outputIndices = graphOutputs;
    } else {
        std::set<int> consumed;
        for (const OpDesc& op : model->ops) {
            for (int t : op.inputs) {
                consumed.insert(t);
            }
        }
        for (int t = 0; t < tensorCount; ++t) {
            if (produced.count(t) > 0 && consumed.count(t) == 0) {
                model->outputIndices.push_back(t);
            }
        }
    }

    *outModel = model;
    return NO_ERROR;
}

} // namespace infer
