//
//  segment.cpp
//  MNN
//
//  Created by MNN on 2019/07/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include <MNN/ImageProcess.hpp>
#define MNN_OPEN_TIME_TRACE
#include <algorithm>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <vector>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/MathOp.hpp>
#include <MNN/AutoTime.hpp>
#include <MNN/Interpreter.hpp>
#include "core/TensorUtils.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace MNN;
using namespace MNN::CV;
using namespace MNN::Express;

int main(int argc, const char* argv[]) {
    if (argc < 4) {
        MNN_PRINT("Usage: ./segment.out model.mnn input.jpg output.jpg\n");
        return 0;
    }
    std::shared_ptr<Interpreter> net;
    net.reset(Interpreter::createFromFile(argv[1]));
    if (net == nullptr) {
        MNN_ERROR("Invalid Model\n");
        return 0;
    }
    ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    config.numThread = 4;
    auto session = net->createSession(config);
    auto input = net->getSessionInput(session, nullptr);
    auto shape = input->shape();
    // Detect NHWC format (common for TFLite-converted models) and extract correct dims
    auto fmt = TensorUtils::getDescribe(input)->dimensionFormat;
    int size_w, size_h, bpp;
    if (fmt == MNN_DATA_FORMAT_NHWC) {
        // NHWC: [N, H, W, C]
        bpp    = shape[3];  // C
        size_h = shape[1];  // H
        size_w = shape[2];  // W
    } else {
        // NCHW: [N, C, H, W]
        bpp    = shape[1];  // C
        size_h = shape[2];  // H
        size_w = shape[3];  // W
    }
    if (shape[0] != 1) {
        shape[0] = 1;
        net->resizeTensor(input, shape);
        net->resizeSession(session);
    }
    {
        MNN_PRINT("input: w:%d , h:%d, bpp: %d\n", size_w, size_h, bpp);

        auto inputPatch = argv[2];
        int width, height, channel;
        auto inputImage = stbi_load(inputPatch, &width, &height, &channel, 4);
        if (nullptr == inputImage) {
            MNN_ERROR("Can't open %s\n", inputPatch);
            return 0;
        }
        MNN_PRINT("origin size: %d, %d\n", width, height);
        Matrix trans;
        // Center-aligned resize matching Python cv.resize:
        // fx = iw/ow, fy = ih/oh, then translate by 0.5*(fx-1)
        float fx = (float)width / size_w;
        float fy = (float)height / size_h;
        trans.postScale(fx, fy);
        trans.postTranslate(0.5f * (fx - 1.0f), 0.5f * (fy - 1.0f));
        ImageProcess::Config config;
        config.filterType = CV::BILINEAR;
        float mean[3]     = {103.94f, 116.78f, 123.68f};
        float normals[3] = {0.017f, 0.017f, 0.017f};
        ::memcpy(config.mean, mean, sizeof(mean));
        ::memcpy(config.normal, normals, sizeof(normals));
        config.sourceFormat = RGBA;
        config.destFormat   = BGR;

        std::shared_ptr<ImageProcess> pretreat(ImageProcess::create(config));
        pretreat->setMatrix(trans);
        // Use raw-pointer convert with explicit output dimensions,
        // works correctly for both NCHW and NHWC session tensors
        pretreat->convert((uint8_t*)inputImage, width, height, 0,
                          input->host<float>(), size_w, size_h, bpp, 0,
                          halide_type_of<float>());
        stbi_image_free(inputImage);
    }
    // Run model
    net->runSession(session);

    // Post treat by MNN-Express
    {
        /* Create VARP by tensor Begin*/
        auto outputTensor = net->getSessionOutput(session, nullptr);
        // First Create a Expr, then create Variable by the 0 index of expr
        auto output = Variable::create(Expr::create(outputTensor));
        if (nullptr == output->getInfo()) {
            MNN_ERROR("Alloc memory or compute size error\n");
            return 0;
        }
        /* Create VARP by tensor End*/

        // Turn dataFormat to NHWC for easy to run TopKV2
        output = _Convert(output, NHWC);
        auto width = output->getInfo()->dim[2];
        auto height = output->getInfo()->dim[1];
        auto channel = output->getInfo()->dim[3];
        MNN_PRINT("output w = %d, h=%d\n", width, height);

        const int humanIndex = 15;
        output = _Reshape(output, {-1, channel});
        // Use ArgMax to get per-pixel class index (matching Python's argmax)
        auto index = _ArgMax(output, -1);
        // If is human, set 255, else set 0
        auto mask = _Select(_Equal(index, _Scalar<int>(humanIndex)), _Scalar<int>(255), _Scalar<int>(0));

        mask = _Cast<uint8_t>(mask);
        // Reshape to [height, width] for grayscale image output (matching Python)
        mask = _Reshape(mask, {height, width});
        stbi_write_png(argv[3], width, height, 1, mask->readMap<uint8_t>(), width);
    }
    return 0;
}
