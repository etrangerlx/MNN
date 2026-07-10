//
//  QNNDeconvolution.cpp
//  MNN
//
//  Created by MNN on 2025/07/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "QNNDeconvolution.hpp"

namespace MNN {
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE

ErrorCode QNNDeconvolution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto conv2D = mOp->main_as_Convolution2D();
    auto common = conv2D->common();
    Qnn_DataType_t dataType = mBackend->getNativeTensor(inputs[0])->v1.dataType;

    int n;
    int ih, iw, ic;
    int oh, ow, oc;
    int kernelH, kernelW;
    int strideH, strideW;
    int padTop, padBottom, padLeft, padRight;
    int dilationH, dilationW;
    int group;
    int outPadH = 0, outPadW = 0;

    // compute shape
    {
        n  = inputs[0]->batch();
        ih = inputs[0]->height();
        iw = inputs[0]->width();
        ic = inputs[0]->channel();
        oh = outputs[0]->height();
        ow = outputs[0]->width();
        oc = outputs[0]->channel();
        kernelH   = common->kernelY();
        kernelW   = common->kernelX();
        strideH   = common->strideY();
        strideW   = common->strideX();
        dilationH = common->dilateY();
        dilationW = common->dilateX();
        group     = common->group();

        // QNN TransposeConv2d does not support dilation
        if (dilationH > 1 || dilationW > 1) {
            MNN_QNN_NOT_SUPPORT_SPECIAL_CASE;
        }

        // Use convolutionTransposePad for deconv padding calculation
        auto pad  = ConvolutionCommon::convolutionTransposePad(inputs[0], outputs[0], common);
        padTop    = pad.second;
        padBottom = pad.second;
        padLeft   = pad.first;
        padRight  = pad.first;

        // If explicit pads array is provided, use it (can be asymmetric)
        if (nullptr != common->pads() && common->pads()->size() >= 4) {
            padTop    = common->pads()->Get(0);
            padLeft   = common->pads()->Get(1);
            padBottom = common->pads()->Get(2);
            padRight  = common->pads()->Get(3);
        }

        // Output padding (additional padding on the output side)
        if (nullptr != common->outPads()) {
            if (common->outPads()->size() >= 2) {
                outPadH = common->outPads()->data()[0];
                outPadW = common->outPads()->data()[1];
            }
        }
    }

    // create params: stride, pad_amount (required); group, output_padding (optional)
    // Note: QNN TransposeConv2d does NOT support dilation
    {
        std::vector<uint32_t> strideData    = {(uint32_t)strideH, (uint32_t)strideW};
        std::vector<uint32_t> padAmountData = {(uint32_t)padTop, (uint32_t)padBottom, (uint32_t)padLeft, (uint32_t)padRight};
        this->createParamTensor("stride", QNN_DATATYPE_UINT_32, {2}, (void *)strideData.data());
        this->createParamTensor("pad_amount", QNN_DATATYPE_UINT_32, {2, 2}, (void *)padAmountData.data());

        if (group > 1) {
            this->createParamScalar("group", (uint32_t)group);
        }

        if (outPadH > 0 || outPadW > 0) {
            std::vector<uint32_t> outputPaddingData = {(uint32_t)outPadH, (uint32_t)outPadW};
            this->createParamTensor("output_padding", QNN_DATATYPE_UINT_32, {2}, (void *)outputPaddingData.data());
        }
    }

    this->createWeightAndBias(dataType, inputs[0], oc, ic, kernelH, kernelW, group);

    // dequant input and quant output
    if (dataType != QNN_DATATYPE_FLOAT_16 && dataType != QNN_DATATYPE_FLOAT_32) {
        return this->onEncodeQuantDequantConv(inputs[0], outputs[0], n, ic, oc, group, outPadH, outPadW, common);
    }

    if (common->relu() || common->relu6()) {
        this->createStageTensor("ReluTensor", dataType, getNHWCShape(outputs[0]), outputs[0]);
    }

    // Helper to push the common params: stride, pad_amount [, group] [, output_padding]
    auto addCommonParams = [&]() {
        mParams.push_back(*(mParamTensorWrappers[0]->getNativeParam())); // stride
        mParams.push_back(*(mParamTensorWrappers[1]->getNativeParam())); // pad_amount
        if (group > 1) {
            mParams.push_back(*(mParamScalarWrappers[0]->getNativeParam())); // group
        }
        if (outPadH > 0 || outPadW > 0) {
            int paramIdx = 2 + (group > 1 ? 1 : 0);
            mParams.push_back(*(mParamTensorWrappers[paramIdx]->getNativeParam())); // output_padding
        }
    };

    // add nodes
    if (common->relu() || common->relu6()) {
        // Stage one: TransposeConv2d
        {
            mNodeType     = "TransposeConv2d";
            std::string name = mNodeName + "_transposeConv";
            addCommonParams();

            mInputs.push_back(*(mBackend->getNativeTensor(inputs[0])));        // input
            mInputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor()));   // weight
            mInputs.push_back(*(mTempTensorWrappers[1]->getNativeTensor()));   // bias

            mOutputs.push_back(*(mTempTensorWrappers[2]->getNativeTensor()));  // stage tensor
            mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(),
                                     mParams, mInputs, mOutputs);
        }

        // Stage two: Activation
        {
            mNodeType.clear();
            mParams.clear();
            mInputs.clear();
            mOutputs.clear();
            mNodeType     = common->relu6() ? "Relu6" : "Relu";
            std::string name = mNodeName + "_relu";

            mInputs.push_back(*(mTempTensorWrappers[2]->getNativeTensor()));   // stage tensor
            mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0])));       // output
            mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(),
                                     mParams, mInputs, mOutputs);
        }
    } else {
        mNodeType = "TransposeConv2d";
        addCommonParams();

        mInputs.push_back(*(mBackend->getNativeTensor(inputs[0])));        // input
        mInputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor()));   // weight
        mInputs.push_back(*(mTempTensorWrappers[1]->getNativeTensor()));   // bias

        mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0])));       // output
        mBackend->addNodeToGraph(mOpConfigVersion, mNodeName.c_str(), mPackageName.c_str(), mNodeType.c_str(),
                                 mParams, mInputs, mOutputs);
    }

    return NO_ERROR;
}

ErrorCode QNNDeconvolution::onEncodeQuantDequantConv(Tensor *input, Tensor *output, const int n, const int ic,
                                                     const int oc, const int group, const int outPadH,
                                                     const int outPadW, const Convolution2DCommon *common) {
    Qnn_DataType_t dataType = QNN_DATATYPE_FLOAT_32;
    if (mBackend->getUseFP16()) {
        dataType = QNN_DATATYPE_FLOAT_16;
    }

    // create dequant input stage tensor and quant output stage tensor
    this->createStageTensor("DequantInput", dataType, getNHWCShape(input));    // mTempTensorWrappers[2]
    this->createStageTensor("QuantOutput", dataType, getNHWCShape(output));    // mTempTensorWrappers[3]

    // Helper to push the common params: stride, pad_amount [, group] [, output_padding]
    auto addCommonParams = [&]() {
        mParams.push_back(*(mParamTensorWrappers[0]->getNativeParam())); // stride
        mParams.push_back(*(mParamTensorWrappers[1]->getNativeParam())); // pad_amount
        if (group > 1) {
            mParams.push_back(*(mParamScalarWrappers[0]->getNativeParam())); // group
        }
        if (outPadH > 0 || outPadW > 0) {
            int paramIdx = 2 + (group > 1 ? 1 : 0);
            mParams.push_back(*(mParamTensorWrappers[paramIdx]->getNativeParam())); // output_padding
        }
    };

    // Dequantize input
    {
        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        mNodeType     = "Dequantize";
        std::string name = mNodeName + "_dequant_input";

        mInputs.push_back(*(mBackend->getNativeTensor(input)));             // input
        mOutputs.push_back(*(mTempTensorWrappers[2]->getNativeTensor()));   // DequantInput
        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams,
                                 mInputs, mOutputs);
    }

    if (common->relu() || common->relu6()) {
        this->createStageTensor("ReluTensor", dataType, getNHWCShape(output)); // mTempTensorWrappers[4]

        // TransposeConv2d
        {
            mParams.clear();
            mInputs.clear();
            mOutputs.clear();
            mNodeType     = "TransposeConv2d";
            std::string name = mNodeName + "_transposeConv";
            addCommonParams();

            mInputs.push_back(*(mTempTensorWrappers[2]->getNativeTensor()));   // DequantInput
            mInputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor()));   // weight
            mInputs.push_back(*(mTempTensorWrappers[1]->getNativeTensor()));   // bias

            mOutputs.push_back(*(mTempTensorWrappers[4]->getNativeTensor()));  // ReluTensor
            mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(),
                                     mParams, mInputs, mOutputs);
        }

        // Activation
        {
            mParams.clear();
            mInputs.clear();
            mOutputs.clear();
            mNodeType     = common->relu6() ? "Relu6" : "Relu";
            std::string name = mNodeName + "_relu";

            mInputs.push_back(*(mTempTensorWrappers[4]->getNativeTensor()));   // ReluTensor
            mOutputs.push_back(*(mTempTensorWrappers[3]->getNativeTensor()));  // QuantOutput
            mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(),
                                     mParams, mInputs, mOutputs);
        }
    } else {
        mParams.clear();
        mInputs.clear();
        mOutputs.clear();
        mNodeType = "TransposeConv2d";
        addCommonParams();

        mInputs.push_back(*(mTempTensorWrappers[2]->getNativeTensor()));   // DequantInput
        mInputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor()));   // weight
        mInputs.push_back(*(mTempTensorWrappers[1]->getNativeTensor()));   // bias

        mOutputs.push_back(*(mTempTensorWrappers[3]->getNativeTensor()));  // QuantOutput
        mBackend->addNodeToGraph(mOpConfigVersion, mNodeName.c_str(), mPackageName.c_str(), mNodeType.c_str(),
                                 mParams, mInputs, mOutputs);
    }

    // Quantize output
    {
        auto QuantOutputTensor = mTempTensorWrappers[3]->getNativeTensor();
        if (mBackend->getUseFP16()) {
            this->createStageTensor("CastOutput", QNN_DATATYPE_FLOAT_32, getNHWCShape(output));

            mParams.clear();
            mInputs.clear();
            mOutputs.clear();
            mNodeType     = "Cast";
            std::string name = mNodeName + "_Cast_Output";

            mInputs.push_back(*(mTempTensorWrappers[3]->getNativeTensor()));        // QuantOutput
            mOutputs.push_back(*(mTempTensorWrappers.back()->getNativeTensor()));    // CastOutput
            mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(),
                                     mParams, mInputs, mOutputs);
            QuantOutputTensor = mTempTensorWrappers.back()->getNativeTensor();
        }
        {
            mParams.clear();
            mInputs.clear();
            mOutputs.clear();
            mNodeType     = "Quantize";
            std::string name = mNodeName + "_Quant_Output";

            mInputs.push_back(*(QuantOutputTensor));                         // stage tensor
            mOutputs.push_back(*(mBackend->getNativeTensor(output)));        // output
            mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(),
                                     mParams, mInputs, mOutputs);
        }
    }

    return NO_ERROR;
}

bool QNNDeconvolution::createWeightAndBias(Qnn_DataType_t dataType, const Tensor *input, int oc, int ic, int kernelH,
                                           int kernelW, int group) {
    Qnn_DataType_t floatDatatype = QNN_DATATYPE_FLOAT_32;
    if (mBackend->getUseFP16()) {
        floatDatatype = QNN_DATATYPE_FLOAT_16;
    }

    std::vector<float> weightData;
    const float *source        = nullptr;
    int weightElementNum        = 0;
    std::shared_ptr<ConvolutionCommon::Int8Common> quanWeight;
    ConvolutionCommon::getConvParameters(&quanWeight, mBackend, mOp, &source, &weightElementNum);

    // MNN deconv weight: [IC, OC/group, KH, KW] -> QNN TransposeConv2d filter: [KH, KW, IC/group, OC]
    int icPerGroup = ic / group;
    weightData.resize(weightElementNum);
    convertWeightDeconv(source, (float *)weightData.data(), oc, icPerGroup, kernelH, kernelW);

    this->createStaticFloatTensor("weight", floatDatatype,
                                  {(uint32_t)kernelH, (uint32_t)kernelW, (uint32_t)icPerGroup, (uint32_t)oc},
                                  weightData.data());

    // create bias
    std::vector<float> biasData;
    biasData.resize(oc, 0);
    auto bias = mOp->main_as_Convolution2D()->bias();
    if (nullptr != bias) {
        ::memcpy((void *)biasData.data(), (void *)bias->data(), oc * sizeof(float));
    }
    this->createStaticFloatTensor("bias", floatDatatype, {(uint32_t)oc}, biasData.data());

    return true;
}

class QNNDeconvolutionCreator : public QnnBackend::Creator {
public:
    virtual QNNCommonExecution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                         const MNN::Op *op, Backend *backend) const override {
        if (inputs.size() > 1) {
            MNN_ERROR("QNN only support single deconv input\n");
            return nullptr;
        }
        return new QNNDeconvolution(backend, op);
    }
};

REGISTER_QNN_OP_CREATOR(QNNDeconvolutionCreator, OpType_Deconvolution)
#endif
} // namespace QNN
} // namespace MNN
