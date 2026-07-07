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
#include <MNN/AutoTime.hpp>
#include <MNN/Interpreter.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace MNN;
using namespace MNN::CV;
using namespace MNN::Express;
struct BBoxRect {
    float score;
    float xMin;
    float yMin;
    float xMax;
    float yMax;
    void* reserved[24];
};
static inline float intersection_area(const BBoxRect &B1, const BBoxRect &B2) {
    if (B1.xMin > B2.xMax ||
        B1.xMax < B2.xMin ||
        B1.yMin > B2.yMax ||
        B1.yMax < B2.yMin) {
        // no intersection
        return 0.f;
    }
    float inter_width = std::min(B1.xMax, B2.xMax) - std::max(B1.xMin, B2.xMin);
    float inter_height = std::min(B1.yMax, B2.yMax) - std::max(B1.yMin, B2.yMin);
    return inter_width * inter_height;
}
void nms_bboxes(const std::vector<BBoxRect> &bboxes,std::vector<size_t> &picked,
                                 float nms_threshold) {
    picked.clear();
    const size_t n = bboxes.size();
    for (size_t i = 0; i < n; ++i) {
        const BBoxRect &B1 = bboxes[i];
        bool keep = true;
        for (int j = 0; j < (int) picked.size(); ++j) {
            const BBoxRect &B2 = bboxes[picked[j]];
            // intersection over union
            float inter_area = intersection_area(B1, B2);
            float B1_Area=(B1.xMax - B1.xMin) * (B1.yMax - B1.yMin);
            float B2_Area=(B2.xMax - B2.xMin) * (B2.yMax - B2.yMin);
            float union_area = B1_Area + B2_Area - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area > nms_threshold * union_area) {
                keep = false;
                break;
            }
        }
        if (keep) {
            picked.push_back(i);
        }
    }
}
void qsort_descent_inplace(std::vector<BBoxRect> &datas, int left, int right) {
    int i = left;
    int j = right;
    float p = datas[(left + right) / 2].score;

    while (i <= j) {
        while (datas[i].score > p)
            i++;

        while (datas[j].score < p)
            j--;
        if (i <= j) {
            // swap
            std::swap(datas[i], datas[j]);
            i++;
            j--;
        }
    }

    if (left < j) qsort_descent_inplace(datas, left, j);

    if (i < right) qsort_descent_inplace(datas, i, right);
}
void PostProcess(const std::vector<BBoxRect> &Boxes, std::vector<BBoxRect> &detect_results,
    int input_width_,int input_height_,int orange_width_,int orange_height_) {
    std::vector<size_t> picked;
    //
    nms_bboxes(Boxes,picked,0.4f);

    size_t max_index = 0;
    float max_area = 0.f; // //
    for (size_t i = 0; i < picked.size(); ++i) {
        size_t z = picked[i];
        float area=(Boxes[z].xMax - Boxes[z].xMin) * (Boxes[z].yMax - Boxes[z].yMin);
        if (max_area < area) {
            max_area = area;
            max_index = z;
        }
    }

    if (picked.size()) {
        float min_x = Boxes[max_index].xMin >= 0 ? Boxes[max_index].xMin : 0;
        float min_y = Boxes[max_index].yMin >= 0 ? Boxes[max_index].yMin : 0;
        float max_x = Boxes[max_index].xMax + 1 < input_width_ ? Boxes[max_index].xMax : input_width_ - 1;
        float max_y = Boxes[max_index].yMax + 1 < input_height_ ? Boxes[max_index].yMax: input_height_ - 1;
        //
        float inv_scale = 1.0f;
        if (orange_height_ > orange_width_) {
            float x_ex = input_height_ * (1.f - 1.f * orange_width_/ orange_height_) / 2;
            // 剔除 padding
            min_x -= x_ex;
            max_x -= x_ex;
            inv_scale = (float)orange_height_ / input_height_;

        }
        else {
            float y_ex = input_width_ * (1.f - 1.f * orange_height_/ orange_width_) / 2;
            min_y -= y_ex;
            max_y -= y_ex;
            inv_scale = (float)orange_width_ / input_width_;
        }
        min_y = std::max((float)0, min_y * inv_scale);
        max_y = std::min((float)orange_height_, max_y * inv_scale);
        min_x = std::max((float)0, min_x * inv_scale);
        max_x = std::min((float)orange_width_, max_x * inv_scale);
        detect_results.push_back(BBoxRect{Boxes[max_index].score, min_x, min_y, max_x, max_y});
    }
}


static int drawBBox(uint32_t* src, float xMin, float yMin, float xMax, float yMax, int width, int height) {
    int x0 = static_cast<int>(xMin);
    int y0 = static_cast<int>(yMin);
    int x1 = static_cast<int>(xMax);
    int y1 = static_cast<int>(yMax);

    x0 = std::max(0, x0);
    y0 = std::max(0, y0);
    x1 = std::min(width - 1, x1);
    y1 = std::min(height - 1, y1);

    for (int x = x0; x <= x1; ++x) {
        src[y0 * width + x] = 0xFFFF00FF;
        src[y1 * width + x] = 0xFFFF00FF;
    }

    for (int y = y0; y <= y1; ++y) {
        src[y * width + x0] = 0xFFFF00FF;
        src[y * width + x1] = 0xFFFF00FF;
    }

    return 0;
}



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
    auto session = net->createSession(config);
    auto input = net->getSessionInput(session, nullptr);
    auto shape = input->shape();
    if (shape[0] != 1) {
        shape[0] = 1;
        net->resizeTensor(input, shape);
        net->resizeSession(session);
    }
    int size_w   = 0;
    int size_h   = 0;
    int width, height, channel;
    {

        int bpp      = 0;
        bpp = shape[1];
        size_h = shape[2];
        size_w = shape[3];
        if (bpp == 0)
            bpp = 1;
        if (size_h == 0)
            size_h = 1;
        if (size_w == 0)
            size_w = 1;
        MNN_PRINT("input: w:%d , h:%d, bpp: %d\n", size_w, size_h, bpp);

        auto inputPatch = argv[2];
        
        auto inputImage = stbi_load(inputPatch, &width, &height, &channel, 4);
        if (nullptr == inputImage) {
            MNN_ERROR("Can't open %s\n", inputPatch);
            return 0;
        }
        MNN_PRINT("origin size: %d, %d\n", width, height);
        // Compute letterbox params (preserve aspect ratio, same as ResizePadding)
        int newW, newH, left = 0, top = 0;
        if (height > width) {
            newH = size_h;
            newW = std::max(1, (int)(size_w * (float)width / height));
            left = (size_w - newW) / 2;
        } else if (width > height) {
            newW = size_w;
            newH = std::max(1, (int)(size_h * (float)height / width));
            top  = (size_h - newH) / 2;
        } else {
            newW = size_w;
            newH = size_h;
        }
        MNN_PRINT("letterbox: newW=%d, newH=%d, left=%d, top=%d\n", newW, newH, left, top);

        // Step 1: resize source image to (newW, newH) into temp uint8 buffer
        std::vector<uint8_t> resized(newW * newH * 3);
        {
            Matrix resizeTrans;
            resizeTrans.setScale((float)(width-1) / (newW-1), (float)(height-1) / (newH-1));
            ImageProcess::Config cfg;
            cfg.filterType   = CV::BILINEAR;
            cfg.sourceFormat = BGRA;
            cfg.destFormat   = RGB;
            std::shared_ptr<ImageProcess> resizer(ImageProcess::create(cfg));
            resizer->setMatrix(resizeTrans);
            resizer->convert((uint8_t*)inputImage, width, height, 0,
                             resized.data(), newW, newH, 3, newW * 3,
                             halide_type_of<uint8_t>());
        }

        // Step 2: zero-fill padded buffer, place resized image centered (letterbox)
        std::vector<uint8_t> padded(size_w * size_h * 3, 0);
        for (int r = 0; r < newH; r++) {
            ::memcpy(padded.data() + ((top + r) * size_w + left) * 3,
                     resized.data() + r * newW * 3,
                     newW * 3);
        }

        // Step 3: copy padded uint8 HWC → float NCHW input tensor
        {
            auto inputPtr = input->host<float>();
            for (int c = 0; c < 3; c++) {
                for (int h = 0; h < size_h; h++) {
                    for (int w = 0; w < size_w; w++) {
                        inputPtr[c * size_h * size_w + h * size_w + w] =
                            (float)padded[(h * size_w + w) * 3 + c];
                    }
                }
            }
        }

        // Dump input tensor to JPEG for verification
        {
            uint8_t* ptr = padded.data();
            stbi_write_jpg("input_dump.jpg", size_w, size_h, 3, ptr, 90);
            MNN_PRINT("Dumped input tensor to input_dump.jpg (%dx%d)\n", size_w, size_h);
        }

        stbi_image_free(inputImage);
    }
    // Run model
    net->runSession(session);

    // Post treat by MNN-Express
    {
        /* Create VARP by tensor Begin*/
        auto bboxesTensor = net->getSessionOutput(session, "bboxes");
        auto classesTensor = net->getSessionOutput(session, "classes");
        // First Create a Expr, then create Variable by the 0 index of expr
        auto bboxesoutput = Variable::create(Expr::create(bboxesTensor));
        if (nullptr == bboxesoutput->getInfo()) {
            MNN_ERROR("Alloc memory or compute size error\n");
            return 0;
        }
        auto bboxes_output_ptr = bboxesoutput->readMap<float>();
        int bboxes_output_size = bboxesoutput->getInfo()->size;
        MNN_PRINT("bboxesoutput w = %p, h=%d\n", bboxes_output_ptr, bboxes_output_size);
        
        auto classesoutput = Variable::create(Expr::create(classesTensor));
        if (nullptr == classesoutput->getInfo()) {
            MNN_ERROR("Alloc memory or compute size error\n");
            return 0;
        }
        auto classes_output_ptr = classesoutput->readMap<float>();
        int classes_output_size = classesoutput->getInfo()->size;
        MNN_PRINT("classesoutput w = %p, h=%d\n", classes_output_ptr, classes_output_size);
        std::vector<BBoxRect> dets;
        dets.clear();
        for (int j = 0; j < 2100; ++j) {
            float tmp_score =  std::exp(classes_output_ptr[j]) / (1.0 + std::exp(classes_output_ptr[j]));
            if (tmp_score > 0.5) {
                BBoxRect temp_bbox{};
                int offset = j * 4; // j*4
                temp_bbox.xMin = bboxes_output_ptr[offset] > 0 ? bboxes_output_ptr[offset] : 0;
                temp_bbox.yMin = bboxes_output_ptr[offset + 1] > 0 ? bboxes_output_ptr[offset + 1] : 0;
                temp_bbox.xMax = bboxes_output_ptr[offset + 2];
                temp_bbox.yMax = bboxes_output_ptr[offset + 3];
                temp_bbox.score = tmp_score;
                dets.push_back(temp_bbox);
            }
        }
        if(!dets.empty()) {
            qsort_descent_inplace(dets,0,dets.size()-1);
        }
        std::vector<BBoxRect> res;
        res.clear();
        PostProcess(dets,res,size_w,size_h,width, height);
        // Reload original image for drawing
        auto originalImage = stbi_load(argv[2], &width, &height, &channel, 4);
        if (originalImage) {
            uint32_t* src = (uint32_t*)originalImage;
            for (const auto &box: res) {
                drawBBox(src, box.xMin, box.yMin, box.xMax, box.yMax, width, height);
                printf("box: (%f, %f, %f, %f), score: %f\n", box.xMin, box.yMin, box.xMax, box.yMax, box.score);
            }
            stbi_write_png(argv[3], width, height, 4, originalImage, width * 4);
            stbi_image_free(originalImage);
            MNN_PRINT("Saved output to %s\n", argv[3]);
        }
            // cv::imwrite(output_image_path, image);
        // const int humanIndex = 15;
        // output = _Reshape(output, {-1, channel});
        // auto kv = _TopKV2(output, _Scalar<int>(1));
        // // Use indice in TopKV2's C axis
        // auto index = kv[1];
        // // If is human, set 255, else set 0
        // auto mask = _Select(_Equal(index, _Scalar<int>(humanIndex)), _Scalar<int>(255), _Scalar<int>(0));

        // //If need faster, use this code
        // //auto mask = _Equal(index, _Scalar<int>(humanIndex)) * _Scalar<int>(255);

        // mask = _Cast<uint8_t>(mask);
        // stbi_write_png(argv[3], width, height, 1, mask->readMap<uint8_t>(), width);
    }
    return 0;
}
