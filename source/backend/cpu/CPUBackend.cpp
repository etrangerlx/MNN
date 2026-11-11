#include "CPUBackend.h"

namespace infer {

std::shared_ptr<Execution> CPUBackend::onCreate(const OpDesc& op, const std::vector<Tensor*>& inputs,
                                                const std::vector<Tensor*>& outputs) {
    switch (op.type) {
        case OpType_Convolution:
        case OpType_ConvolutionDepthwise:
            return createCPUConvolutionOp(this, op, inputs, outputs);
        case OpType_Pooling:
            return createCPUPoolingOp(this, op, inputs, outputs);
        case OpType_ReLU:
        case OpType_ReLU6:
        case OpType_PReLU:
        case OpType_Softmax:
        case OpType_UnaryOp:
        case OpType_BinaryOp:
        case OpType_Eltwise:
        case OpType_Reduction:
        case OpType_Scale:
            return createCPUUnaryOp(this, op, inputs, outputs);
        case OpType_Concat:
        case OpType_Const:
        case OpType_Input:
        case OpType_Reshape:
        case OpType_Squeeze:
        case OpType_Flatten:
        case OpType_Permute:
        case OpType_Shape:
        case OpType_Cast:
        case OpType_ConvertTensor:
        case OpType_Identity:
        case OpType_StridedSlice:
        case OpType_GatherV2:
        case OpType_Pack:
        case OpType_ArgMax:
        case OpType_ArgMin:
        case OpType_Transpose:
        case OpType_Unpack:
        case OpType_SliceTf:
            return createCPUShapeOp(this, op, inputs, outputs);
        case OpType_InnerProduct:
        case OpType_MatMul:
        case OpType_Interp:
        case OpType_Padding:
            return createCPUInnerProductOp(this, op, inputs, outputs);
        default:
            return nullptr;
    }
}

std::shared_ptr<Backend> createCPUBackend(ThreadPool* pool) {
    return std::make_shared<CPUBackend>(pool);
}

} // namespace infer
