//
//  QNNDeconvolution.cpp
//  MNN
//
//  Created by MNN on 2025/07/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "QNNDeconvolution.hpp"
#include "core/ConvolutionCommon.hpp"

namespace MNN {
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE

ErrorCode QNNDeconvolution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto conv2D     = mOp->main_as_Convolution2D();
    auto common     = conv2D->common();
    Qnn_DataType_t dataType = mBackend->getNativeTensor(inputs[0])->v1.dataType;

    MNN_PRINT("MNN_QNN: Deconvolution onEncode start, op name: %s\n", mOp->name() ? mOp->name()->c_str() : "unnamed");

    int n, ih, iw, ic;
    int oh, ow, oc;
    int kernelH, kernelW;
    int strideH, strideW;
    int padTop, padBottom, padLeft, padRight;
    int outPadH, outPadW;
    int dilationH, dilationW;
    int group;

    // compute shape
    {
        n = inputs[0]->batch();
        ih = inputs[0]->height(); iw = inputs[0]->width(); ic = inputs[0]->channel();
        oh = outputs[0]->height(); ow = outputs[0]->width(); oc = outputs[0]->channel();
        kernelH = common->kernelY(); kernelW = common->kernelX();
        strideH = common->strideY(); strideW = common->strideX();
        dilationH = common->dilateY(); dilationW = common->dilateX();
        group = common->group();

        // QNN TransposeConv2d does not support dilation
        if (dilationH != 1 || dilationW != 1) {
            MNN_PRINT("MNN_QNN: TransposeConv2d does not support dilation (%d, %d)\n", dilationH, dilationW);
            return NOT_SUPPORT;
        }

        // Compute padding
        if (nullptr != common->pads()) {
            MNN_ASSERT(common->pads()->size() >= 4);
            padTop    = common->pads()->data()[0];
            padLeft   = common->pads()->data()[1];
            padBottom = common->pads()->data()[2];
            padRight  = common->pads()->data()[3];
        } else {
            padTop = padBottom = common->padY();
            padLeft = padRight = common->padX();
        }

        // Output padding
        outPadH = 0; outPadW = 0;
        if (nullptr != common->outPads()) {
            MNN_ASSERT(common->outPads()->size() >= 2);
            outPadH = common->outPads()->data()[0];
            outPadW = common->outPads()->data()[1];
        }
    }

    // create all params (QNN TransposeConv2d: stride, pad_amount, output_padding, group)
    // mParamTensorWrappers[0] = stride
    // mParamTensorWrappers[1] = pad_amount
    // mParamTensorWrappers[2] = output_padding
    // mParamScalarWrappers[0] = group
    {
        std::vector<uint32_t> strideData = {(uint32_t)strideH, (uint32_t)strideW};
        std::vector<uint32_t> padAmountData = {(uint32_t)padTop, (uint32_t)padBottom, (uint32_t)padLeft, (uint32_t)padRight};
        std::vector<uint32_t> outputPaddingData = {(uint32_t)outPadH, (uint32_t)outPadW};
        this->createParamTensor("stride", QNN_DATATYPE_UINT_32, {2}, (void *)strideData.data());
        this->createParamTensor("pad_amount", QNN_DATATYPE_UINT_32, {2, 2}, (void *)padAmountData.data());
        this->createParamTensor("output_padding", QNN_DATATYPE_UINT_32, {2}, (void *)outputPaddingData.data());
        this->createParamScalar("group", (uint32_t)group);
    }

    // create weight and bias
    // mTempTensorWrappers[0] = weight
    // mTempTensorWrappers[1] = bias
    // mTempTensorWrappers[2] = relu stage tensor (optional)
    createWeightAndBias(dataType, oc, ic, kernelH, kernelW, group);

    // Handle relu/relu6 fusion
    if (common->relu() || common->relu6()) {
        this->createStageTensor("ReluTensor", dataType, getNHWCShape(outputs[0]), outputs[0]);
    }

    // add nodes
    {
        if (common->relu() || common->relu6()) {
            // Stage one: TransposeConv2d
            {
                mNodeType = "TransposeConv2d";
                std::string name = mNodeName + "_transpose_conv";
                mParams.push_back(*(mParamTensorWrappers[0]->getNativeParam())); // stride
                mParams.push_back(*(mParamTensorWrappers[1]->getNativeParam())); // pad_amount
                mParams.push_back(*(mParamTensorWrappers[2]->getNativeParam())); // output_padding
                mParams.push_back(*(mParamScalarWrappers[0]->getNativeParam())); // group

                mInputs.push_back(*(mBackend->getNativeTensor(inputs[0])));  // input
                mInputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor())); // weight
                mInputs.push_back(*(mTempTensorWrappers[1]->getNativeTensor())); // bias

                mOutputs.push_back(*(mTempTensorWrappers[2]->getNativeTensor())); // stage tensor
                mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
            }

            // Stage two: Relu/Relu6
            {
                mNodeType.clear();
                mParams.clear();
                mInputs.clear();
                mOutputs.clear();
                mNodeType = common->relu6() ? "Relu6" : "Relu";
                std::string name = mNodeName + "_relu";

                mInputs.push_back(*(mTempTensorWrappers[2]->getNativeTensor())); // stage tensor
                mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0])));    // output
                mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
            }
        } else {
            // Direct TransposeConv2d
            mNodeType = "TransposeConv2d";
            mParams.push_back(*(mParamTensorWrappers[0]->getNativeParam())); // stride
            mParams.push_back(*(mParamTensorWrappers[1]->getNativeParam())); // pad_amount
            mParams.push_back(*(mParamTensorWrappers[2]->getNativeParam())); // output_padding
            mParams.push_back(*(mParamScalarWrappers[0]->getNativeParam())); // group

            mInputs.push_back(*(mBackend->getNativeTensor(inputs[0])));  // input
            mInputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor())); // weight
            mInputs.push_back(*(mTempTensorWrappers[1]->getNativeTensor())); // bias

            mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0]))); // output
            mBackend->addNodeToGraph(mOpConfigVersion, mNodeName.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
        }
    }
    MNN_PRINT("MNN_QNN: Deconvolution onEncode done, op name: %s\n", mOp->name() ? mOp->name()->c_str() : "unnamed");
    return NO_ERROR;
}

void QNNDeconvolution::createWeightAndBias(Qnn_DataType_t dataType, int oc, int ic, int kernelH, int kernelW, int group) {
    // Get weight data
    std::vector<float> weightData;
    const float* source = nullptr;
    int weightElementNum = 0;
    std::shared_ptr<ConvolutionCommon::Int8Common> quanWeight;
    ConvolutionCommon::getConvParameters(&quanWeight, mBackend, mOp, &source, &weightElementNum);

    // Reorder weight from MNN format [OC, IC, KH, KW] to QNN TransposeConv2d format [KH, KW, IC, OC]
    weightData.resize(weightElementNum);
    convertWeight(source, (float *)weightData.data(), oc, ic / group, kernelH, kernelW);

    Qnn_DataType_t floatDatatype = QNN_DATATYPE_FLOAT_32;
    if (mBackend->getUseFP16()) {
        floatDatatype = QNN_DATATYPE_FLOAT_16;
    }
    this->createStaticFloatTensor("weight", floatDatatype,
                                  {(uint32_t)kernelH, (uint32_t)kernelW, (uint32_t)ic / (uint32_t)group, (uint32_t)oc},
                                  weightData.data());

    // Create bias
    createBias(dataType, oc, group);
}

void QNNDeconvolution::createBias(Qnn_DataType_t /*dataType*/, int oc, int /*group*/) {
    int biasElementNum = oc;
    std::vector<float> biasData;
    biasData.resize(biasElementNum, 0);
    auto bias = mOp->main_as_Convolution2D()->bias();
    if (nullptr != bias) {
        ::memcpy((void *)biasData.data(), (void *)bias->data(), biasElementNum * sizeof(float));
    }
    Qnn_DataType_t floatDatatype = QNN_DATATYPE_FLOAT_32;
    if (mBackend->getUseFP16()) {
        floatDatatype = QNN_DATATYPE_FLOAT_16;
    }
    this->createStaticFloatTensor("bias", floatDatatype, {(uint32_t)oc}, biasData.data());
}

class QNNDeconvolutionCreator : public QnnBackend::Creator {
public:
    virtual QNNCommonExecution * onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        if (inputs.size() > 1) {
            MNN_ERROR("QNN only support single deconv input\n");
            return nullptr;
        }
        return new QNNDeconvolution(backend, op);
    }
};

REGISTER_QNN_OP_CREATOR(QNNDeconvolutionCreator, OpType_Deconvolution)
#endif
} // end namespace QNN
} // end namespace MNN
