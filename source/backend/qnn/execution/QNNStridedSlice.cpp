//
//  QNNStridedSlice.cpp
//  MNN
//
//  Created by MNN on b'2025/04/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "QNNStridedSlice.hpp"

#define CLIP(input, min, max) ((input) < (min) ? (min) : ((input) > (max) ? (max) : (input)))

namespace MNN {
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE

QNNStridedSlice::QNNStridedSlice(Backend *backend, const Op *op) : QNNCommonExecution(backend, op) {
    if(op->type() == OpType_Slice) {
        mIsSlice = true;
    }
}

ErrorCode QNNStridedSlice::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto inputTensor = inputs[0];
    mInputDim = inputTensor->dimensions();
    mDimType = inputTensor->getDimensionType();
    auto inputShape = inputTensor->shape();
    if (TensorUtils::getDescribe(inputs[0])->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
        // Turn to nhwc
        for (int index = 2; index < mInputDim; ++index) {
            inputShape[index - 1] = inputTensor->length(index);
        }
        if (mInputDim >= 2) {
            inputShape[mInputDim-1] = inputTensor->length(1);
        }
    }

    if(mIsSlice) {
        auto param = mOp->main_as_Slice();
        auto axis = param->axis();
        if (axis < 0) {
            axis = inputTensor->dimensions() + axis;
        }
        auto shape = inputShape;
        int realAxis = axis;
        if (TensorUtils::getDescribe(inputs[0])->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
            if (axis > 1) {
                realAxis = axis - 1;
            } else if (axis == 1) {
                realAxis = mInputDim - 1;
            }
        }

        // Determine slice sizes for each output based on sourceType.
        // - CAFFE: slicePoints are cumulative split positions.
        // - TF/Torch (size > 1): slicePoints are output sizes.
        // - TF/Torch (null or size 1): even split.
        int inputDimSize = inputs[0]->length(axis);
        int numOutputs = (int)outputs.size();
        std::vector<int> sliceSizes;
        auto sourceType = param->sourceType();
        auto slicePoints = param->slicePoints();

        if (sourceType == MNN::NetSource_CAFFE) {
            // CAFFE: slicePoints are cumulative split positions
            int previous = 0;
            if (slicePoints != nullptr) {
                for (int i = 0; i < slicePoints->size() && (int)sliceSizes.size() < numOutputs; ++i) {
                    int splitPos = slicePoints->Get(i);
                    sliceSizes.push_back(splitPos - previous);
                    previous = splitPos;
                }
            }
            // Last output gets the remainder
            if ((int)sliceSizes.size() < numOutputs) {
                sliceSizes.push_back(inputDimSize - previous);
            }
        } else if (slicePoints == nullptr || slicePoints->size() == 1) {
            // TF/Torch: single or no slicePoint → even split
            int numSplits = numOutputs;
            int splitSize = inputDimSize / std::max(1, numSplits);
            if (slicePoints != nullptr && slicePoints->size() == 1) {
                if (sourceType == MNN::NetSource_TORCH) {
                    // Torch: single value is split_size
                    splitSize = slicePoints->Get(0);
                    numSplits = inputDimSize / splitSize;
                } else {
                    // TF: single value is num_splits (override if != outputSize)
                    if (slicePoints->Get(0) != numOutputs) {
                        numSplits = slicePoints->Get(0);
                    }
                    splitSize = inputDimSize / std::max(1, numSplits);
                }
            }
            numSplits = std::min(numSplits, numOutputs);
            for (int i = 0; i < numSplits; i++) {
                sliceSizes.push_back(splitSize);
            }
        } else {
            // TF/Torch: slicePoints are output sizes (possibly with -1 as unknown)
            int numSplits = std::min((int)slicePoints->size(), numOutputs);
            int knownSize = 0;
            int unknownIdx = -1;
            for (int i = 0; i < numSplits; i++) {
                int len = slicePoints->Get(i);
                if (len != -1) {
                    sliceSizes.push_back(len);
                    knownSize += len;
                } else {
                    sliceSizes.push_back(0); // placeholder
                    unknownIdx = i;
                }
            }
            // Handle -1 placeholder (infer from remaining size)
            if (unknownIdx >= 0) {
                sliceSizes[unknownIdx] = inputDimSize - knownSize;
            }
        }

        #ifdef QNN_VERBOSE
        MNN_PRINT("slice:%d %d %d %d, axis:%d, realAxis:%d, output_num:%d, dim:%d, sourceType:%d\n",
                  shape[0], shape[1], shape[2], shape[3], axis, realAxis, numOutputs, mInputDim, (int)sourceType);
        #endif

        // Create StridedSlice for each output
        int offset = 0;
        int numSlices = std::min((int)sliceSizes.size(), numOutputs);
        for (int index = 0; index < numSlices; index++) {
            std::vector<int> rangeData(mInputDim * 3, 0);
            for (int i = 0; i < mInputDim; i++) {
                rangeData[3 * i + 0] = 0;
                rangeData[3 * i + 1] = shape[i];
                rangeData[3 * i + 2] = 1;
            }
            rangeData[3 * realAxis + 0] = offset;
            rangeData[3 * realAxis + 1] = offset + sliceSizes[index];
            offset += sliceSizes[index];

            this->createParamTensor("ranges", QNN_DATATYPE_INT_32, {(uint32_t) mInputDim, 3}, (void *) rangeData.data(), std::to_string(index));

            // Add Node.
            mNodeType = "StridedSlice";
            mParams.clear();
            mInputs.clear();
            mOutputs.clear();
            std::string name =  mNodeName + "_part" + std::to_string(index);
            mParams.push_back(*(mParamTensorWrappers[index]->getNativeParam()));
            mInputs.push_back(*(mBackend->getNativeTensor(inputs[0])));
            mOutputs.push_back(*(mBackend->getNativeTensor(outputs[index])));

            mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
        }
        return NO_ERROR;
    }
    if (TensorUtils::getDescribe(inputs[0])->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
        MNN_ERROR("[QNN] Don't Support NC4HW4 stridedslice now\n");
        return NOT_SUPPORT;
    }

    auto param = mOp->main_as_StridedSliceParam();
    mNodeType = "StridedSlice";

    // Deal with ranges.
    std::vector<int> beginRaw(mInputDim, 0);
    std::vector<int> endRaw = inputTensor->shape();
    std::vector<int> strideRaw(mInputDim, 1);
    if (param->fromType() == 0) {
        this->computeRangesType0(inputs, beginRaw, endRaw, strideRaw);
    } else {
        this->computeRangesType1(inputs, beginRaw, endRaw, strideRaw);
    }

    std::vector<int> rangeData(mInputDim * 3, 0);
    for (int axis = 0; axis < mInputDim; axis++) {
        rangeData[3 * axis + 0] = beginRaw[axis];
        rangeData[3 * axis + 1] = endRaw[axis];
        rangeData[3 * axis + 2] = strideRaw[axis];
    }
    this->createParamTensor("ranges", QNN_DATATYPE_INT_32, {(uint32_t) mInputDim, 3}, (void *) rangeData.data());

    // Deal with masks.
    uint32_t beginMaskData = computeMask(param->beginMask(), mInputDim, mDimType);
    uint32_t endMaskData =  computeMask(param->endMask(), mInputDim, mDimType);
    uint32_t shrinkAxesData =  computeMask(param->shrinkAxisMask(), mInputDim, mDimType);
    uint32_t newAxesMaskData = computeMask(param->newAxisMask(), mInputDim, mDimType);

    this->createParamScalar("begin_mask", beginMaskData);
    this->createParamScalar("end_mask", endMaskData);
    this->createParamScalar("shrink_axes", shrinkAxesData);
    this->createParamScalar("new_axes_mask", newAxesMaskData);

    // Add Node.
    mParams.push_back(*(mParamTensorWrappers[0]->getNativeParam()));
    for (int i = 0; i < mParamScalarWrappers.size(); i++) {
        mParams.push_back(*(mParamScalarWrappers[i]->getNativeParam()));
    }
    mInputs.push_back(*(mBackend->getNativeTensor(inputs[0])));
    mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0])));

    mBackend->addNodeToGraph(mOpConfigVersion, mNodeName.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);

    return NO_ERROR;
}

void QNNStridedSlice::computeRangesType0(const std::vector<Tensor *> &inputs, std::vector<int> & beginRaw, std::vector<int> & endRaw, std::vector<int> & strideRaw) {
    auto inputTensor = inputs[0];
    auto beginTensor = inputs[1];
    auto endTensor = inputs[2];
    auto strideTensor = inputs[3];
    auto beginRawSource = beginTensor->host<int>();
    auto endRawSource = endTensor->host<int>();
    auto strideRawSource = strideTensor->host<int>();

    int sliceDim = beginTensor->length(0);
    MNN_ASSERT(sliceDim == endTensor->length(0) && sliceDim == strideTensor->length(0));

    for (int i = 0; i < sliceDim; i++) {
        beginRaw[i] = CLIP(beginRawSource[i], 0, inputs[0]->length(i) - 1);
        endRaw[i] = CLIP(endRawSource[i], 1, inputs[0]->length(i));
        strideRaw[i] = strideRawSource[i];
    }
    return;
}

void QNNStridedSlice::computeRangesType1(const std::vector<Tensor *> &inputs, std::vector<int> & beginRaw, std::vector<int> & endRaw, std::vector<int> & strideRaw) {
    auto inputTensor = inputs[0];
    auto beginTensor = inputs[1];
    auto endTensor = inputs[2];
    auto strideTensor = inputs[4];
    auto beginRawSource = beginTensor->host<int>();
    auto endRawSource = endTensor->host<int>();
    auto strideRawSource = strideTensor->host<int>();

    auto axisTensor = inputs[3];
    int sliceDim = beginTensor->length(0);
    MNN_ASSERT(sliceDim == endTensor->length(0) && sliceDim == axisTensor->length(0) && sliceDim == strideTensor->length(0));

    for (int i = 0; i < sliceDim; i++) {
        int tempAxis = axisTensor->host<int>()[i];
        tempAxis = tempAxis >= 0 ? tempAxis : (tempAxis + mInputDim);
        beginRaw[tempAxis] = CLIP(beginRawSource[i], 0, inputs[0]->length(tempAxis) - 1);
        endRaw[tempAxis] = CLIP(endRawSource[i], 1, inputs[0]->length(tempAxis));
        strideRaw[tempAxis] = strideRawSource[i];
    }
    return;
}


uint32_t QNNStridedSlice::computeMask(uint32_t rawMask, int dim, Tensor::DimensionType mDimType) {
    if (rawMask == 0) return 0;

    uint32_t result = 0;
    for (int axis = 0; axis < dim; axis++) {
        int realAxis = axis;
        result |= ((rawMask >> axis) & 1) << realAxis; // If the axis-th bit of rawMask is 1, set the realAxis-th bit of result to 1.
    }

    return result;
}

class QNNStridedSliceCreator : public QnnBackend::Creator {
public:
    virtual QNNCommonExecution * onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        if(op->type() == OpType_Slice) {
            return new QNNStridedSlice(backend, op);
        }
        auto param = op->main_as_StridedSliceParam();

        // <begin>, <end> and <stride> should be static.
        for (int i = 1; i < inputs.size(); i++) {
            MNN_ASSERT(TensorUtils::getDescribe(inputs[i])->usage == Tensor::InsideDescribe::Usage::CONSTANT);
        }

        if (param->fromType() == 1) {
            MNN_ASSERT(param->shrinkAxisMask() == 0 && param->newAxisMask() == 0 && param->ellipsisMask() == 0);
            if (inputs.size() != 5) {
                return nullptr;
            }
            return new QNNStridedSlice(backend, op);
        }

        // [TODO] 把newAxisMask和ellipsisMask考虑在内
        if (param->fromType() == 0) {
            if (inputs.size() == 4 && param->newAxisMask() == 0 && param->ellipsisMask() == 0) {
                return new QNNStridedSlice(backend, op);
            } else {
                return nullptr;
            }
        }

        // Shouldn't reach here.
        return nullptr;
    }
};

REGISTER_QNN_OP_CREATOR(QNNStridedSliceCreator, OpType_StridedSlice)
REGISTER_QNN_OP_CREATOR(QNNStridedSliceCreator, OpType_Slice)
#endif
} // end namespace QNN
} // end namespace MNN
