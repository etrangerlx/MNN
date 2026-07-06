# QNN Backend: Add ResizeBilinear (Interp) Op Support

## 1. 问题现象

执行 `MNN2QNNModel` 转换 DeepLabV3 模型时报错：

```
param dims: 1x257x257x3
param dims: 1x257x257x3
Total input shape type size:1
[Temp Product]: Qnn temp product generate at ./output_seg//deeplabv3_257_mv_gpu_0
The device supports: i8sdot:0, fp16:0, i8mm: 0, sve2: 0, sme2: 0
[Warning]: No QnnDevice_getPlatformInfo APILoad Cache file error.
MNN_QNN: Not registered type 35, ResizeBilinear_1.
MNN_QNN: Not registered type 35, ResizeBilinear_2.
MNN_QNN: Not registered type 35, ResizeBilinear_3.
Tensor usage is 0.
Error for /home/leixian/MNN/source/backend/qnn/backend/QNNBackend.cpp, 1899
MNN2QNNModel: ... QNNBackend.cpp:1899: int MNN::QNN::QnnBackend::getTensorIdx(const MNN::Tensor*) const: Assertion `res' failed.
Aborted (core dumped)
```

**核心问题:** QNN 后端缺少 `Interp` (OpType 35) 算子的注册，导致模型中的 3 个 ResizeBilinear 层无法处理。

---

## 2. 诊断过程

### 2.1 确认算子类型

通过 `MNNDump2Json` 导出模型结构，确认三个 ResizeBilinear 节点的参数：

| 节点名称 | OpType | resizeType | outputWidth | outputHeight | alignCorners | halfPixelCenters |
|---------|--------|------------|-------------|--------------|--------------|------------------|
| ResizeBilinear_1 | Interp (35) | 2 | 33 | 33 | true | false |
| ResizeBilinear_2 | Interp (35) | 2 | 33 | 33 | true | false |
| ResizeBilinear_3 | Interp (35) | 2 | 257 | 257 | true | false |

```bash
../../build/MNNDump2Json deeplabv3_257_mv_gpu.mnn /tmp/deeplab_dump.json
```

### 2.2 理解 QNN 算子注册机制

通过搜索 `REGISTER_QNN_OP_CREATOR` 宏的使用方式，发现 QNN 后端采用**显式注册**模式（与 CPU 后端的自动注册不同）：

1. 算子 `.cpp` 文件中通过 `REGISTER_QNN_OP_CREATOR(Creator, OpType)` 生成注册函数 `___Creator__OpType__()`
2. `QNNUtils.hpp` 中需要 `extern` 声明该函数
3. `QNNUtils.cpp` 的 `registerQNNOps()` 中需要**显式调用**该函数

缺失任何一步都会导致运行时 "Not registered type N" 错误。

### 2.3 模型中的 concat 结构分析

通过 JSON dump 分析 concat 结构：

```
Tensor 62 (backbone output)  → AvgPool2D/AvgPool → Tensor 63 (1x1)
Tensor 63 → image_pooling/Relu (Conv 1x1) → Tensor 64 (1x1)
Tensor 64 → ResizeBilinear_1 → Tensor 65 (33x33)
Tensor 62 → aspp0/Relu (Conv) → Tensor 66 (33x33)
Tensor 65 + Tensor 66 → concat (axis=1) → Tensor 67
```

concat 输出形状为 `{1, 33, 33, 512}` → 通道翻倍(256+256=512)，说明实际是沿通道维拼接。

### 2.4 发现 Concat axis 转换 bug

MNN 内部使用 **NC4HW4** 格式存储 tensor，Concat 的 axis 也以此格式存储。但 QNN 使用 **NHWC** 格式。axis=1 在 NC4HW4 中代表通道维，在 NHWC 中代表高度维。直接将 axis=1 传给 QNN 导致 QNN 尝试沿高度拼接(期望输出 {1,66,33,256})，但 MNN 形状推导计算的是沿通道拼接(输出 {1,33,33,512})，产生维度不匹配。

**NC4HW4 → NHWC axis 映射 (4D):**
```
{0→0, 1→3, 2→1, 3→2}
```

| NC4HW4 axis | 含义 | NHWC axis |
|-------------|------|-----------|
| 0 | batch | 0 |
| 1 | channel_blocks | **3** (channels) |
| 2 | height | 1 |
| 3 | width | 2 |

### 2.5 Segfault 排查

测试过程中遇到 SIGSEGV，通过 GDB 定位：

```gdb
gdb -batch -ex run -ex bt --args ../../build/MNN2QNNModel ...
```

```
#0  __GI__IO_fwrite (buf=..., size=1, count=971, fp=0x0) at iofwrite.c:37
#1  MNN::QNN::QNNConvertor::Translate (cmd=...) at QNNConvertor.cpp:165
```

根因：输出目录不存在导致文件指针为 NULL。**与我们的代码修改无关。**

---

## 3. 解决方案

### 3.1 新建文件

#### `source/backend/qnn/execution/QNNInterp.hpp`

```cpp
//
//  QNNInterp.hpp
//  MNN
//
//  Created by MNN on 2025/07/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_QNNINTERP_HPP
#define MNN_QNNINTERP_HPP

#include "QNNCommonExecution.hpp"
#include "QnnTypes.h"

namespace MNN {
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE

class QNNInterp : public QNNCommonExecution {
public:
    QNNInterp(Backend *backend, const Op *op) : QNNCommonExecution(backend, op) {}
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};
#endif
} // end namespace QNN
} // end namespace MNN

#endif // end MNN_QNNINTERP_HPP
```

#### `source/backend/qnn/execution/QNNInterp.cpp`

```cpp
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
        } else if (ctm == CoordinateTransformationMode_AlignCorners ||
                   ctm == CoordinateTransformationMode_TensorflowCropAndResize) {
            transformationMode = 2; // ALIGN_CORNERS
        } else if (ctm == CoordinateTransformationMode_PytorchHalfPixels ||
                   ctm == CoordinateTransformationMode_TensorflowHalfPixels) {
            transformationMode = 1; // PYTORCH_HALF_PIXEL
        } else if (ctm == CoordinateTransformationMode_HalfPixels) {
            transformationMode = 0; // HALF_PIXEL
        } else if (ctm == CoordinateTransformationMode_Asymmetric) {
            transformationMode = 3; // ASYMMETRIC
        }
        this->createParamScalar("transformation_mode", transformationMode);

        if (resizeType == 4) {
            this->createParamScalar("nearest_mode", (uint32_t)1); // ROUND_PREFER_CEIL
        } else if (resizeType == 1) {
            this->createParamScalar("nearest_mode", (uint32_t)0); // ROUND_PREFER_FLOOR
        }
        if (resizeType == 3) {
            this->createParamScalar("cubic_coeff", interp->cubicCoeffA());
        }
    }

    this->addNodeCommon(inputs, outputs);

    return NO_ERROR;
}

class QNNInterpCreator : public QnnBackend::Creator {
public:
    virtual QNNCommonExecution * onCreate(const std::vector<Tensor*>& inputs,
                                          const std::vector<Tensor*>& outputs,
                                          const MNN::Op* op,
                                          Backend* backend) const override {
        return new QNNInterp(backend, op);
    }
};

REGISTER_QNN_OP_CREATOR(QNNInterpCreator, OpType_Interp)
#endif
} // end namespace QNN
} // end namespace MNN
```

### 3.2 算子参数映射详解

#### MNN Interp → QNN ResizeBilinear (resizeType=2)

| MNN 参数 | QNN 参数 | 说明 |
|----------|---------|------|
| `alignCorners` | `align_corners` (bool) | 直接映射 |

QNN ResizeBilinear 只有一个参数 `align_corners`，输出尺寸通过 output tensor 的 shape 决定。

#### MNN Interp → QNN Resize (resizeType=1/3/4)

| MNN resizeType | `interpolation_mode` | 额外参数 |
|----------------|---------------------|---------|
| 1 (nearest) | 0 = NEAREST | `nearest_mode=0` (ROUND_PREFER_FLOOR) |
| 3 (cubic) | 2 = CUBIC | `cubic_coeff=-0.75` |
| 4 (nearest_round) | 0 = NEAREST | `nearest_mode=1` (ROUND_PREFER_CEIL) |

#### CoordinateTransformationMode 映射

| MNN CoordinateTransformationMode | `alignCorners` | `halfPixelCenters` | QNN `transformation_mode` |
|----------------------------------|----------------|--------------------|---------------------------|
| NotSet | true | - | **2** (ALIGN_CORNERS) |
| NotSet | false | true | **0** (HALF_PIXEL) |
| NotSet | false | false | **3** (ASYMMETRIC) |
| AlignCorners | - | - | **2** |
| PytorchHalfPixels | - | - | **1** |
| HalfPixels | - | - | **0** |
| Asymmetric | - | - | **3** |
| TensorflowCropAndResize | - | - | **2** |
| TensorflowHalfPixels | - | - | **1** |

### 3.3 修改已有文件

#### `source/backend/qnn/backend/QNNUtils.hpp` (+1 line)

在 `__QNNTopKV2Creator__OpType_TopKV2__` 声明之后添加：

```cpp
extern void ___QNNInterpCreator__OpType_Interp__();
```

#### `source/backend/qnn/backend/QNNUtils.cpp` (+1 line)

在 `registerQNNOps()` 函数的 `___QNNTopKV2Creator__OpType_TopKV2__();` 调用之后添加：

```cpp
___QNNInterpCreator__OpType_Interp__();
```

### 3.4 修复 Concat axis 转换 bug

#### `source/backend/qnn/execution/QNNConcat.cpp` (+13 lines)

原代码：
```cpp
int dim = outputs[0]->dimensions();
if (axis < 0) {
    axis = dim + axis;
}
MNN_ASSERT(axis >= 0 && axis < dim);
this->createParamScalar("axis", (uint32_t)axis);
```

修改为：
```cpp
int dim = outputs[0]->dimensions();
if (axis < 0) {
    axis = dim + axis;
}
MNN_ASSERT(axis >= 0 && axis < dim);

// Convert axis from MNN internal format (NC4HW4) to QNN format (NHWC).
// NC4HW4 4D: [batch, channel_blocks, height, width]
// NHWC 4D:    [batch, height, width, channels]
// Mapping: 0→0, 1→3, 2→1, 3→2
auto dataFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
if (dataFormat == MNN_DATA_FORMAT_NC4HW4) {
    if (dim == 4) {
        int axisMap[] = {0, 3, 1, 2};
        axis = axisMap[axis];
    } else if (dim == 5) {
        // NC4HW4 5D: [batch, extra, channel_blocks, height, width]
        // NHWC 5D:   [batch, extra, height, width, channels]
        int axisMap[] = {0, 1, 4, 2, 3};
        axis = axisMap[axis];
    }
}
this->createParamScalar("axis", (uint32_t)axis);
```

这个 bug 是之前就存在的，但被 "Not registered type 35" 错误掩盖了（因为转换在 Interp 算子处就中断了，不会执行到 Concat 验证）。

---

## 4. 所有修改文件清单

```
M source/backend/qnn/backend/QNNUtils.cpp    (+1 line)  添加注册函数调用
M source/backend/qnn/backend/QNNUtils.hpp    (+1 line)  添加 extern 声明
M source/backend/qnn/execution/QNNConcat.cpp (+13 lines) 修复 axis 转换 bug
?? source/backend/qnn/execution/QNNInterp.hpp (new)      Interp 算子头文件
?? source/backend/qnn/execution/QNNInterp.cpp (new)      Interp 算子实现
```

## 5. QNN 算子注册核心模式

QNN 后端采用**显式注册**（与 CPU 后端的自动注册不同），添加新算子必须完成三步：

```
Step 1: 算子 .cpp 文件末尾
    REGISTER_QNN_OP_CREATOR(CreatorClass, OpType)
    → 生成 void ___CreatorClass__OpType__()

Step 2: source/backend/qnn/backend/QNNUtils.hpp
    → extern void ___CreatorClass__OpType__();

Step 3: source/backend/qnn/backend/QNNUtils.cpp → registerQNNOps()
    → ___CreatorClass__OpType__();  // 显式调用
```

**缺失任何一步**都会导致 "Not registered type N" 错误，即使代码编译通过。

## 6. 踩坑记录

| # | 现象 | 根因 | 解决 |
|---|------|------|------|
| 1 | `Not registered type 35` | Interp 算子未注册 | 创建 QNNInterp 并完成三步注册 |
| 2 | concat validation ERROR: Expected 66 got 33 | Concat axis 未从 NC4HW4 转为 NHWC | 添加 axis 转换映射 |
| 3 | SIGSEGV (fp=0x0 in fwrite) | 输出目录不存在 | 预先 `mkdir -p` 输出目录 |
| 4 | QNN_OP_RESIZE 导致 SIGSEGV | QNN_OP_RESIZE 可能在当前 HTP 后端不可用 | 使用 QNN_OP_RESIZE_BILINEAR 处理 bilinear |

## 7. 编译 & 测试

```bash
# 编译
cd build
cmake --build . --target MNN2QNNModel -j8

# 测试 (注意: 输出目录必须预先存在!)
cd ../demo/model
mkdir -p ./output
../../build/MNN2QNNModel /mnt/d/SoftWare/qairt/2.45.0.260326 72 81 \
    deeplabv3_257_mv_gpu.mnn ./output/ 1 1x257x257x3
```

预期输出:
```
[Pass]: qnn-model-lib-generator success!
[Pass]: qnn-context-binary-generator success!
[All passed]
```
