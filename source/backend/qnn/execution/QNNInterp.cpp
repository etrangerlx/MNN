//
//  QNNInterp.cpp
//  MNN
//
//  Created by MNN on 2025/07/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "QNNInterp.hpp"

namespace MNN {
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE

ErrorCode QNNInterp::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mParams.clear();
    mInputs.clear();
    mOutputs.clear();

    auto interp = mOp->main_as_Interp();
    int resizeType = interp->resizeType(); // 1=nearest, 2=bilinear, 3=cubic, 4=nearest_round

    if (resizeType == 2) {
        // Bilinear: use QNN_OP_RESIZE_BILINEAR
        mNodeType = "ResizeBilinear";
        this->createParamScalar("align_corners", interp->alignCorners());
    } else {
        // Nearest / Cubic: use QNN_OP_RESIZE
        mNodeType = "Resize";

        // QNN: 0=NEAREST, 1=LINEAR, 2=CUBIC
        uint32_t interpolationMode;
        if (resizeType == 1 || resizeType == 4) {
            interpolationMode = 0; // NEAREST
        } else if (resizeType == 3) {
            interpolationMode = 2; // CUBIC
        } else {
            MNN_QNN_NOT_SUPPORT_SPECIAL_CASE;
        }
        this->createParamScalar("interpolation_mode", interpolationMode);

        // Map CoordinateTransformationMode
        uint32_t transformationMode = 0;
        auto ctm = interp->ctm();
        if (ctm == CoordinateTransformationMode_NotSet) {
            if (interp->alignCorners()) {
                transformationMode = 2; // ALIGN_CORNERS
            } else if (interp->halfPixelCenters()) {
                transformationMode = 0; // HALF_PIXEL
            } else {
                transformationMode = 3; // ASYMMETRIC
            }
        } else if (ctm == CoordinateTransformationMode_AlignCorners || ctm == CoordinateTransformationMode_TensorflowCropAndResize) {
            transformationMode = 2;
        } else if (ctm == CoordinateTransformationMode_PytorchHalfPixels || ctm == CoordinateTransformationMode_TensorflowHalfPixels) {
            transformationMode = 1;
        } else if (ctm == CoordinateTransformationMode_HalfPixels) {
            transformationMode = 0;
        } else if (ctm == CoordinateTransformationMode_Asymmetric) {
            transformationMode = 3;
        }
        this->createParamScalar("transformation_mode", transformationMode);

        if (resizeType == 4) {
            this->createParamScalar("nearest_mode", (uint32_t)1);
        } else if (resizeType == 1) {
            this->createParamScalar("nearest_mode", (uint32_t)0);
        }
        if (resizeType == 3) {
            this->createParamScalar("cubic_coeff", interp->cubicCoeffA());
        }
    }

#ifdef QNN_VERBOSE
    {
        auto inShape = inputs[0]->shape();
        auto outShape = outputs[0]->shape();
        MNN_PRINT("QNN Interp: %s, resizeType=%d, ctm=%d, alignCorners=%d, halfPixelCenters=%d, transformMode=%d, interpMode=%d\n",
            mOp->name() ? mOp->name()->c_str() : "null",
            resizeType, (int)ctm, (int)interp->alignCorners(), (int)interp->halfPixelCenters(),
            transformationMode, interpolationMode);
        MNN_PRINT("  input shape: %d x %d x %d x %d\n", inShape[0], inShape[1], inShape[2], inShape[3]);
        MNN_PRINT("  output shape: %d x %d x %d x %d\n", outShape[0], outShape[1], outShape[2], outShape[3]);
    }
#endif

    this->addNodeCommon(inputs, outputs);

    return NO_ERROR;
}

class QNNInterpCreator : public QnnBackend::Creator {
public:
    virtual QNNCommonExecution * onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        return new QNNInterp(backend, op);
    }
};

REGISTER_QNN_OP_CREATOR(QNNInterpCreator, OpType_Interp)
#endif
} // end namespace QNN
} // end namespace MNN
