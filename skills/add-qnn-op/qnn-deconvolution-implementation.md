# QNN Backend: Add Deconvolution (ConvTranspose) Op Support

## 1. 问题现象

执行 `MNN2QNNModel` 转换 face_det 模型时报错：

```
MNN_QNN: Not registered type 17, /model/fpn/upsample1/ConvTranspose_output_0.
MNN_QNN: Not registered type 17, /model/fpn/upsample2/ConvTranspose_output_0.
```

**核心问题:** QNN 后端缺少 `Deconvolution` (OpType 17) 算子的注册，导致模型中的 2 个 ConvTranspose 层无法处理。

---

## 2. 诊断过程

### 2.1 确认算子类型

通过搜索 `MNN_generated.h` 确认类型 17 为 `OpType_Deconvolution`：

```cpp
// schema/current/MNN_generated.h:136
OpType_Deconvolution = 17,
```

在 ONNX/PyTorch 术语中，ConvTranspose 在 MNN 内部被称为 Deconvolution。

同时存在 `OpType_DeconvolutionDepthwise = 18` 用于深度可分离反卷积。

### 2.2 确认 QNN SDK 支持的对应算子

查看 QNN SDK v2.42.0 的 `QnnOpDef.h`：

```cpp
#define QNN_OP_TRANSPOSE_CONV_2D                      "TransposeConv2d"
#define QNN_OP_TRANSPOSE_CONV_2D_PARAM_STRIDE         "stride"
#define QNN_OP_TRANSPOSE_CONV_2D_PARAM_PAD_AMOUNT     "pad_amount"
#define QNN_OP_TRANSPOSE_CONV_2D_PARAM_GROUP          "group"
#define QNN_OP_TRANSPOSE_CONV_2D_PARAM_OUTPUT_PADDING "output_padding"
```

**关键发现：QNN TransposeConv2d 不支持 `dilation` 参数！**（与 Conv2d 不同，Conv2d 有 `QNN_OP_CONV_2D_PARAM_DILATION`）

### 2.3 确认 MNN Deconvolution 权重格式

| 后端 | 权重布局 |
|------|---------|
| MNN Convolution weight | [OC, IC, KH, KW] (NCHW) |
| MNN Deconvolution weight | [IC, OC, KH, KW] (NCHW) |
| QNN Conv2d filter | [KH, KW, IC/group, OC] (HWIO) |
| QNN TransposeConv2d filter | [KH, KW, IC/group, OC] (HWIO) |

Deconvolution 的权重转换与 Convolution **不同**：源布局从 `[OC, IC, KH, KW]` 变为 `[IC, OC, KH, KW]`，需要专用的转换函数。

### 2.4 确认 Padding 计算方式

Deconvolution 使用 `ConvolutionCommon::convolutionTransposePad()` 计算 padding，与 Convolution 的 `convolutionPadFull()` 不同。同时支持 `outPads` (output_padding) 参数。

### 2.5 理解 QNN 算子注册机制

QNN 后端采用**显式注册**模式，需要三步（缺一不可）：

1. `.cpp` 文件中 `REGISTER_QNN_OP_CREATOR(Creator, OpType)` → 生成 `___Creator__OpType__()`
2. `QNNUtils.hpp` 中 `extern void ___Creator__OpType__();`
3. `QNNUtils.cpp` 的 `registerQNNOps()` 中显式调用

---

## 3. 解决方案

### 3.1 新建文件

#### `source/backend/qnn/execution/QNNDeconvolution.hpp`

```cpp
#ifndef MNN_QNNDECONVOLUTION_HPP
#define MNN_QNNDECONVOLUTION_HPP

#include "QNNCommonExecution.hpp"
#include "QnnTypes.h"

namespace MNN {
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE

class QNNDeconvolution : public QNNCommonExecution {
public:
    QNNDeconvolution(Backend *backend, const Op *op) : QNNCommonExecution(backend, op) {}
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs,
                               const std::vector<Tensor *> &outputs) override;

private:
    // MNN deconv weight: [IC, OC, KH, KW] -> QNN filter: [KH, KW, IC/group, OC]
    template <typename T>
    void convertWeightDeconv(const T *src, T *dst, int oc, int ic, int kernelH, int kernelW);

    bool createWeightAndBias(Qnn_DataType_t dataType, const Tensor *input,
                             int oc, int ic, int kernelH, int kernelW, int group);

    ErrorCode onEncodeQuantDequantConv(Tensor *input, Tensor *output,
                                       const int n, const int ic, const int oc,
                                       const int group, const int outPadH, const int outPadW,
                                       const Convolution2DCommon *common);
};

#endif
} // namespace QNN
} // namespace MNN
#endif
```

#### `source/backend/qnn/execution/QNNDeconvolution.cpp`

关键实现点：

**1. Dilation 检查（QNN TransposeConv2d 不支持）**

```cpp
if (dilationH > 1 || dilationW > 1) {
    MNN_QNN_NOT_SUPPORT_SPECIAL_CASE;
}
```

**2. Padding 计算（使用 deconv 专用函数）**

```cpp
auto pad = ConvolutionCommon::convolutionTransposePad(inputs[0], outputs[0], common);
padTop    = pad.second;
padBottom = pad.second;
padLeft   = pad.first;
padRight  = pad.first;

// 支持非对称 pads
if (nullptr != common->pads() && common->pads()->size() >= 4) {
    padTop    = common->pads()->Get(0);
    padLeft   = common->pads()->Get(1);
    padBottom = common->pads()->Get(2);
    padRight  = common->pads()->Get(3);
}

// 支持 output_padding
if (nullptr != common->outPads()) {
    if (common->outPads()->size() >= 2) {
        outPadH = common->outPads()->data()[0];
        outPadW = common->outPads()->data()[1];
    }
}
```

**3. 权重格式转换**

```cpp
template <typename T>
void convertWeightDeconv(const T *src, T *dst, int oc, int ic, int kernelH, int kernelW) {
    for (int o = 0; o < oc; o++) {
        for (int i = 0; i < ic; i++) {
            for (int h = 0; h < kernelH; h++) {
                for (int w = 0; w < kernelW; w++) {
                    // src: MNN [IC, OC, KH, KW] → dst: QNN [KH, KW, IC, OC]
                    uint32_t srcOffset = w + kernelW * (h + kernelH * (o + oc * i));
                    uint32_t dstOffset = o + oc * (i + ic * (w + kernelW * h));
                    dst[dstOffset] = src[srcOffset];
                }
            }
        }
    }
}
```

**4. QNN 节点参数（无 dilation）**

QNN TransposeConv2d 仅接受：`stride`, `pad_amount`, `group`(可选), `output_padding`(可选)。不传 `dilation`。

**5. 支持 relu/relu6 激活融合**

分两阶段：`TransposeConv2d → 中间 tensor → Relu/Relu6`

**6. 支持 int8 量化路径**

`Dequantize → TransposeConv2d → Quantize`

### 3.2 算子参数映射

| MNN Convolution2DCommon | QNN TransposeConv2d | 说明 |
|-------------------------|---------------------|------|
| `kernelX/Y` | filter 维度 | 通过 weight tensor shape 体现 |
| `strideX/Y` | `stride` [2] | 直接映射 |
| `padX/Y` / `pads[]` | `pad_amount` [2,2] | 使用 `convolutionTransposePad()` 计算 |
| `dilateX/Y` | **不支持** | dilation > 1 时返回 NOT_SUPPORT |
| `group` | `group` | 直接映射 (仅 group > 1 时传递) |
| `outPads[]` | `output_padding` [2] | 直接映射 |
| `relu/relu6` | 融合为两阶段节点 | TransposeConv2d → Relu/Relu6 |

### 3.3 修改已有文件

#### `source/backend/qnn/backend/QNNUtils.hpp` (+1 line)

在 `__QNNConvolutionCreator__OpType_Convolution__` 声明之后添加：

```cpp
extern void ___QNNDeconvolutionCreator__OpType_Deconvolution__();
```

#### `source/backend/qnn/backend/QNNUtils.cpp` (+1 line)

在 `registerQNNOps()` 函数的 `___QNNInterpCreator__OpType_Interp__();` 之后添加：

```cpp
___QNNDeconvolutionCreator__OpType_Deconvolution__();
```

### 3.4 不需要修改 CMakeLists.txt

`source/backend/qnn/CMakeLists.txt` 使用 `file(GLOB EXECUTION_SRCS ...)` 自动包含 `execution/` 目录下所有 `.cpp` 文件，新增文件会被自动编译。

---

## 4. 过程中暴露并修复的 Bug

### 4.1 QNNFlatten::ReshapeTranspose 未初始化变量 → segfault

**文件:** `source/backend/qnn/execution/QNNFlatten.cpp`

**根因:** `ReshapeTranspose()` 函数中，当 `outputDim <= 2` 时 `outputTempIndex` 未被初始化，但后续 `nchw→nhwc` 转置块（第 115-129 行）无条件访问 `mTempTensorWrappers[outputTempIndex]`。

**原因:** 我们的 Deconvolution 注册改变了图的内存分配布局，使得 Reshape/Flatten 算子进入了 `ReshapeTranspose` 路径（之前在 CPU 回退路径中不会触发）。

**修复:** 将 `nchw→nhwc` 转置块包裹在 `if (permuteOutput)` 条件中：

```cpp
// Before:
// nchw -> nhwc
{
    // ... unconditional access to outputTempIndex
}

// After:
// nchw -> nhwc (only needed when output was permuted)
if (permuteOutput) {
    // ... safe access to outputTempIndex
}
```

### 4.2 QNNConvertor CppFilePointer 空指针 → segfault

**文件:** `source/backend/qnn/convertor/QNNConvertor.cpp`

**根因:** `MNNCreateDir()` 使用 POSIX `mkdir()` 而非 `mkdir -p`，不支持嵌套目录创建。当 `MNN2QNNModel` 传入 `<outputDir>/<modelName>` 时，父目录 `<outputDir>` 不存在导致 `mkdir` 失败，后续 `RecordBegin` 无法打开 .cpp 文件，`CppFilePointer` 保持 NULL。

**说明:** 这是一个**预先存在的 bug**，与 Deconvolution 实现无关。在未注册 Deconvolution 时同样会触发。

**临时解决方法:** 运行 `MNN2QNNModel` 前预先创建输出目录：

```bash
mkdir -p ./qnn_output
```

---

## 5. 所有修改文件清单

```
M source/backend/qnn/backend/QNNUtils.cpp              (+1 line)  添加 Deconvolution 注册调用
M source/backend/qnn/backend/QNNUtils.hpp              (+1 line)  添加 extern 声明
M source/backend/qnn/execution/QNNFlatten.cpp          (+1 line)  修复 nchw→nhwc 转置块条件
A source/backend/qnn/execution/QNNDeconvolution.hpp    (new)      Deconvolution 算子头文件
A source/backend/qnn/execution/QNNDeconvolution.cpp    (new)      Deconvolution 算子实现
```

---

## 6. QNN 算子注册核心模式（三步法）

```
Step 1: 算子 .cpp 文件末尾 (在 #ifdef ENABLE_QNN_ONLINE_FINALIZE 内部)
    REGISTER_QNN_OP_CREATOR(CreatorClass, OpType)
    → 生成 void ___CreatorClass__OpType__()

Step 2: source/backend/qnn/backend/QNNUtils.hpp
    → extern void ___CreatorClass__OpType__();  // 必须与 Step 1 同名

Step 3: source/backend/qnn/backend/QNNUtils.cpp → registerQNNOps()
    → ___CreatorClass__OpType__();  // 显式调用
```

**⚠️ 缺失任何一步都会导致编译或运行时 "Not registered type N" 错误。**

### 实现模板

```cpp
// === MyOp.hpp ===
#include "QNNCommonExecution.hpp"
namespace MNN { namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE
class QNNMyOp : public QNNCommonExecution {
public:
    QNNMyOp(Backend *b, const Op *op) : QNNCommonExecution(b, op) {}
    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) override;
};
#endif
}}

// === MyOp.cpp ===
#include "QNNMyOp.hpp"
namespace MNN { namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE
ErrorCode QNNMyOp::onEncode(...) {
    // 1. 读取 MNN op 参数
    // 2. 映射到 QNN 参数 (createParamScalar / createParamTensor / createStaticFloatTensor)
    // 3. 设置 mNodeType (QNN op 名称字符串)
    // 4. 搭建 mParams / mInputs / mOutputs
    // 5. 调用 mBackend->addNodeToGraph(...)
    return NO_ERROR;
}

class QNNMyOpCreator : public QnnBackend::Creator {
    virtual QNNCommonExecution* onCreate(...) const override {
        return new QNNMyOp(backend, op);
    }
};
REGISTER_QNN_OP_CREATOR(QNNMyOpCreator, OpType_XXX)
#endif
}}
```

---

## 7. 踩坑记录

| # | 现象 | 根因 | 解决 |
|---|------|------|------|
| 1 | `Not registered type 17` | Deconvolution 算子未注册 | 创建 QNNDeconvolution 并完成三步注册 |
| 2 | SIGSEGV in QNNFlatten `outputTempIndex` | `outputTempIndex` 在 `outputDim <= 2` 时未初始化 | 将 nchw→nhwc 转置块包裹在 `if (permuteOutput)` 中 |
| 3 | SIGSEGV in QNNConvertor::Translate `fp=0x0` | 输出目录不存在 (mkdir 不支持嵌套) | 预先 `mkdir -p` 输出目录 |
| 4 | extern 函数未声明编译错误 | 忘记在 QNNUtils.hpp 添加 extern 声明 | 添加 extern 声明 |
| 5 | QNN TransposeConv2d 没有 dilation 参数 | QNN SDK 不支持 dilated transpose conv | 添加 dilation > 1 检查，返回 NOT_SUPPORT |

---

## 8. Deconvolution vs Convolution 实现差异

| 维度 | Convolution | Deconvolution |
|------|-------------|---------------|
| QNN op 名称 | `"Conv2d"` | `"TransposeConv2d"` |
| 权重布局 (MNN) | [OC, IC, KH, KW] | [IC, OC, KH, KW] |
| 权重转换函数 | `convertWeight` | `convertWeightDeconv` |
| Dilation 支持 | ✓ 支持 | ✗ QNN 不支持 |
| Padding 计算 | `convolutionPadFull()` | `convolutionTransposePad()` |
| output_padding | 不支持 | ✓ `outPads` 映射 |
| relu/relu6 融合 | ✓ | ✓ |
| 量化路径 | ✓ | ✓ |

## 9. 编译 & 测试

```bash
# 编译
cd build
cmake --build . --target MNN2QNNModel -j$(nproc)

# 测试 (注意: 输出目录必须预先存在!)
cd ../demo/model
mkdir -p ./qnn_output
../../build/MNN2QNNModel \
    /workspace/software/qnn_sdk_v2.42.0_auto_lnx/qaisw-v2.42.0.251225135753_193295-auto-lnx \
    72 81 \
    face_det.mnn \
    ./qnn_output \
    1 1x3x320x320
```

预期：不再出现 `Not registered type 17` 错误。
