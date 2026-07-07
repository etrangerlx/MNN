# CV模型 ONNX 转化与 QNN 推理完整指南

> 基于 MNN 源码分析，版本：master 分支，分析日期：2026-07-03

---

## 目录

1. [架构总览](#1-架构总览)
2. [CV模型 ONNX → MNN 转化](#2-cv模型-onnx--mnn-转化)
3. [QNN 推理详解](#3-qnn-推理详解)
4. [完整工作流](#4-完整工作流)
5. [常见CV模型适配指南](#5-常见cv模型适配指南)
6. [常见问题排查](#6-常见问题排查)
7. [关键文件索引](#7-关键文件索引)

---

## 1. 架构总览

MNN 的 QNN 推理涉及两条流水线：

```
┌─────────────────────────────────────────────────────────────────┐
│                        开发侧 (Host/PC)                          │
│                                                                  │
│  PyTorch ──→ ONNX ──→ MNNConvert ──→ MNN模型                    │
│                                          │                       │
│                           ┌──────────────┴──────────────┐        │
│                           │   compilefornpu /           │        │
│                           │   MNN2QNNModel              │        │
│                           └──────────────┬──────────────┘        │
│                                          │                       │
│                           QNN 离线缓存 (.bin) 嵌入 MNN           │
│                                                                  │
└──────────────────────────────────────┬───────────────────────────┘
                                       │ 部署
┌──────────────────────────────────────┴───────────────────────────┐
│                        设备侧 (Android)                           │
│                                                                  │
│  MNN Runtime + QNN Backend ──→ HTP/DSP 推理                     │
│  (libQnnHtp.so, libQnnSystem.so, libQnnHtpVxxSkel.so)           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 关键概念

| 概念 | 说明 |
|------|------|
| **MNNConvert** | 通用模型格式转换工具，支持 TF/Caffe/ONNX/TFLite → MNN |
| **compilefornpu** | MNN → NPU离线编译工具，生成 Plugin Op 嵌入 MNN 模型 |
| **MNN2QNNModel** | 调用 QNN SDK 生成离线缓存的工具（SOC 级别） |
| **QNN Backend** | MNN 的 QNN 运行时后端（`source/backend/qnn/`） |
| **HTP** | Hexagon Tensor Processor（高通 AI 引擎） |
| **QNN SDK** | 高通神经网络 SDK（通过 `prepare_qnn_deps.sh` 下载） |

---

## 2. CV模型 ONNX → MNN 转化

### 2.1 编译转换工具

```bash
cd MNN
mkdir build && cd build
cmake .. -DMNN_BUILD_CONVERTER=true
make -j$(nproc)
```

生成的可执行文件 (`tools/converter/CMakeLists.txt`):

| 工具 | 说明 |
|------|------|
| `MNNConvert` | 主转换器（ONNX/TF/Caffe/TFLite → MNN） |
| `MNNDump2Json` | MNN 模型 dump 为可读 JSON |
| `MNNRevert2Buffer` | MNN 模型还原为 buffer |
| `TestConvertResult` | 转换结果测试 |
| `TestPassManager` | Pass 管理器测试 |

**编译依赖（`tools/converter/CMakeLists.txt:8-12`）:**
- Protobuf >= 3.0（可通过 `-DMNN_BUILD_PROTOBUFFER=ON` 使用内置 protobuf）
- flatbuffers（内置在 `3rd_party/flatbuffers/`）

### 2.2 PyTorch → ONNX → MNN

```python
# Step 1: PyTorch → ONNX
import torch

model = YourCVModel().eval()
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    do_constant_folding=True,
    opset_version=11          # MNN 推荐 opset 11
)
```

```bash
# Step 2: ONNX → MNN
./MNNConvert \
  -f ONNX \
  --modelFile model.onnx \
  --MNNModel model.mnn \
  --bizCode MNN
```

**MNNConvert 参数说明（`tools/converter/README_CN.md:24-38`）:**

| 参数 | 必需 | 说明 |
|------|------|------|
| `-f, --framework` | 是 | 模型类型: `TF`, `CAFFE`, `ONNX`, `TFLITE`, `MNN` |
| `--modelFile` | 是 | 输入模型文件路径 |
| `--MNNModel` | 是 | 输出 MNN 模型路径 |
| `--prototxt` | Caffe专用 | Caffe 网络定义文件 |
| `--bizCode` | 是 | 模型标识，如 `MNN` |
| `--benchmarkModel` | 否 | 移除权重只保留结构，用于性能测试 |
| `--debug` | 否 | 开启调试模式 |

### 2.3 转换后的优化 Pipeline

转换时会自动运行优化流水线（`tools/converter/source/optimizer/PostConverter.cpp`），按以下顺序：

1. **后处理 Passes**: RemoveInplace, RemoveUnusefulOp, RemoveDropout, RemoveInvalidCast 等
2. **框架专项 Passes**: OnnxExtra / TfExtra / CaffeExtra（如 `OnnxBatchNormMerge`, `OnnxGemm`）
3. **图融合 Passes** (4个优先级): TemplateMerge 驱动的算子融合
4. **BN/Scale 合并**: BatchNorm→Conv, Scale→Conv, ReLU→Conv, ReLU6→Conv
5. **最终处理**: RemoveCopy, AddTensorFormatConverter, TransformGroupConvolution

### 2.4 CV 模型转换脚本

MNN 提供了便捷脚本（`tools/script/forconvert/`）：

```bash
# ONNX 转换
bash tools/script/forconvert/convertOnnx.sh model.onnx model.mnn

# 等效于:
./MNNConvert -f ONNX --modelFile model.onnx --MNNModel model.mnn --bizCode AliNNTest
```

---

## 3. QNN 推理详解

### 3.1 QNN 依赖准备

MNN 通过 `prepare_qnn_deps.sh` 下载高通 QNN SDK：

```bash
bash prepare_qnn_deps.sh
```

该脚本从 `http://meta.alicdn.com/data/mnn/libs/qnn_inc_libs_2_37.zip` 下载 QNN SDK 头文件和库，解压到 `source/backend/qnn/3rdParty/`，并设置环境变量 `QNN_SDK_ROOT`。

### 3.2 模式一：在线推理（MNN_QNN=ON）

适用的场景：开发调试阶段，快速验证。

**编译：**
```bash
mkdir build_qnn && cd build_qnn

cmake .. \
  -DMNN_QNN=ON \
  -DQNN_SDK_ROOT=$PWD/../source/backend/qnn/3rdParty \
  -DCMAKE_TOOLCHAIN_FILE=../cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_NATIVE_API_LEVEL=android-28

make -j$(nproc)
```

**推理代码（C++）：**
```cpp
#include "MNN/Interpreter.hpp"
#include "MNN/MNNForwardType.h"

// 创建 QNN Runtime
MNN::ScheduleConfig config;
config.type = MNN_FORWARD_NN;  // 使用 NPU 后端
config.numThread = 1;

auto session = interpreter->createSession(config);
// ... 正常推理 ...
```

### 3.3 模式二：离线编译（compilefornpu）

适用的场景：生产部署，追求最佳性能。

**核心工具：** `compilefornpu`（`tools/cpp/compilefornpu.cpp`）

```
./compilefornpu <src.mnn> <dst.mnn> <npu_config.json>
```

**NPU 配置 JSON 格式（来自 `compilefornpu.cpp:1473-1537`）：**

```json
{
    "type": "QNN",
    "cache": "./qnn_cache_dir",
    "graph_name": "my_cv_model",
    "skips": ["Reshape", "Flatten"],
    "testdir": []
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | string | NPU 类型：`"QNN"` / `"MLDA"` / `"CoreML"` / 其他（走 Native NPU） |
| `cache` | string | 离线缓存输出目录 |
| `graph_name` | string | 图名称标识 |
| `skips` | string[] | 强制跳过、走 CPU 的算子列表 |
| `testdir` | string[] | 测试数据目录（用于精度校验） |
| `KVCACHE_SIZE_LIMIT` | int | KV Cache 大小限制（LLM 专用） |

**工作原理：**

1. 加载 MNN 模型，解析计算图
2. 识别"断点算子"（不支持算子和控制流算子），将图切分为子模块
3. 每个子模块通过 QNN Backend 编译为 QNN 图
4. 对比 QNN 输出与 CPU 参考输出（误差阈值 < 10%）
5. 将编译好的子模块替换为 `OpType_Plugin` (opcode 256)
6. 输出合并后的 MNN 模型 + `npu_postreat.json`

**计算图切分逻辑：**

```
原始 MNN 图:
  [Conv] → [BN] → [ReLU] → [Conv] → [Pool] → [UnsupportedOp] → [FC] → [Softmax]

切分后:
  SubGraph1 (QNN): [Conv] → [BN] → [ReLU] → [Conv] → [Pool]
  Break:           [UnsupportedOp] (CPU)
  SubGraph2 (QNN): [FC] → [Softmax]
```

### 3.4 模式三：MNN2QNNModel（SOC 级别离线编译）

适用的场景：针对特定 SOC 生成最优 QNN 缓存。

**工具：** `tools/cpp/MNN2QNNModel.cpp`

```bash
# 用法
./MNN2QNNModel <QNN_SDK路径> <SOC_ID> <Hexagon_Arch> <srcMNN> <outputDir> [inputShape...]

# 示例：骁龙 8Gen2
./MNN2QNNModel /path/to/qnn_sdk 43 73 cv_model.mnn ./qnn_output 1,3,224,224
```

**SOC 与 Hexagon 架构映射（来自 `apps/Android/MnnLlmChat/.../QnnModule.kt`）：**

| 芯片平台 | SOC Model ID | Hexagon Arch |
|----------|-------------|--------------|
| SM8350 (骁龙888) | 36 | V68 (69) |
| SM8450 (骁龙8Gen1) | 42 | V69 (69) |
| SM8475 (骁龙8+Gen1) | 42 | V69 (69) |
| SM8550 (骁龙8Gen2) | 43 | V73 (73) |
| SM8650 (骁龙8Gen3) | 57 | V75 (75) |
| SM8750 (骁龙8 Elite) | 69 | V79 (79) |

**批量生成脚本（`tools/script/genQNNModelsFromMNN.py`）：**
```bash
# 一次性为多个 SOC 生成 QNN 缓存
python tools/script/genQNNModelsFromMNN.py
# 覆盖: (36,'69'), (42,'69'), (43,'73'), (57,'75'), (69,'79')
```

### 3.5 QNN Backend 架构

**目录结构：** `source/backend/qnn/`

```
source/backend/qnn/
├── CMakeLists.txt              # QNN 后端构建配置
├── npu_convert.py              # QNN 离线缓存后处理脚本
├── backend/
│   ├── QNNBackend.hpp/cpp      # 核心：QNN 后端实现 (2268+ 行)
│   ├── QNNUtils.hpp/cpp        # 工具函数、QNN API 动态加载
│   ├── QNNWrapper.hpp/cpp      # Tensor/Param 包装器
│   ├── QNNPerf.hpp/cpp         # HTP 性能配置（功耗模式、RPC延迟）
│   ├── QnnTypeMacros.hpp       # QNN 类型访问宏 (597行，来自高通)
│   └── dsprpc_interface.h/cc   # FastRPC / ION 内存分配
├── convertor/
│   ├── QNNConvertor.hpp/cpp    # 离线转换逻辑
│   └── QNNConvertorInterface.hpp/cpp
└── execution/                  # 23 个算子执行实现
    ├── QNNActivation           # ReLU, ReLU6, Sigmoid, ELU
    ├── QNNBinary               # BinaryOp, Eltwise
    ├── QNNConvolution          # Convolution
    ├── QNNConvDepthwise        # ConvolutionDepthwise
    ├── QNNMatmul               # MatMul
    ├── QNNPool                 # Pooling, Pooling3D
    ├── QNNSoftmax              # Softmax
    ├── QNNConcat               # Concat, Pack, Unpack
    ├── QNNPermute              # Permute, Transpose
    ├── QNNReshape              # ConvertTensor
    ├── QNNFlatten              # Flatten, Reshape, Squeeze, Unsqueeze
    ├── QNNReduce               # Reduction
    ├── QNNGather               # GatherV2, GatherElements
    ├── QNNLayerNorm            # LayerNorm
    ├── QNNStridedSlice         # StridedSlice, Slice
    ├── QNNPadding              # Padding
    ├── QNNScale                # Scale
    ├── QNNUnary                # UnaryOp
    ├── QNNCast                 # Cast
    ├── QNNQuant                # FloatToInt8, Int8ToFloat
    ├── QNNBroadcastTo          # BroadcastTo
    ├── QNNArgmax               # ArgMax, ArgMin
    ├── QNNTopKV2               # TopKV2
    └── QNNAttention            # Attention（Transformer 融合）
```

**关键编译选项（`CMakeLists.txt:283-284`）：**

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `MNN_QNN` | OFF | 开启 QNN 后端编译 |
| `MNN_QNN_ONLINE_FINALIZE` | ON | 在线图编译模式（运行时完成图构建） |

**QNN 后端支持的数据类型（`QNNBackend.cpp`）：**
- `FLOAT_32` (FP32)
- `FLOAT_16` (FP16) — CV 模型默认推荐
- `INT_32` (INT32)
- `SFIXED_POINT_8` (INT8 量化)
- `UFIXED_POINT_16` (UINT16 量化)

### 3.6 MNNForwardType 中的 NPU 类型

定义在 `include/MNN/MNNForwardType.h`：

```cpp
MNN_FORWARD_NN          = 5,   // 通用 NPU（在线模式 Native NPU compute）
MNN_CONVERT_QNN         = 32,  // 高通 QNN（离线模式）
MNN_CONVERT_NEUROPILOT  = 33,  // 联发科 NeuroPilot
MNN_CONVERT_COREML      = 34,  // Apple CoreML
```

---

## 4. 完整工作流

### 4.1 为CV模型做 QNN 推理的推荐流程

```
Step 1: 准备模型
  PyTorch CV Model → ONNX (opset=11)

Step 2: 转 MNN
  ./MNNConvert -f ONNX --modelFile model.onnx --MNNModel model.mnn --bizCode MNN

Step 3: 验证 MNN 模型（CPU 推理正确性）
  ./MNNDump2Json model.mnn  # 检查图结构
  # 或写测试代码跑 CPU 推理对比

Step 4: QNN 离线编译
  ./compilefornpu model.mnn model_qnn.mnn npu_config.json

Step 5: 精度验证
  compilefornpu 内部已做误差校验（阈值 < 10%）
  如需更严格的验证，使用 testdir 配置提供测试数据

Step 6: 部署到设备
  将 model_qnn.mnn + QNN .so 库打包到 Android APK/AAR
```

### 4.2 npu_config.json 完整示例

```json
{
    "type": "QNN",
    "cache": "./qnn_output",
    "graph_name": "yolo_v8_npu",
    "skips": ["NMS", "NonMaxSuppression"],
    "testdir": [
        "./test_images/image1",
        "./test_images/image2"
    ]
}
```

### 4.3 设备端推理代码示例

```cpp
// C++ 推理
#include "MNN/Interpreter.hpp"
#include "MNN/ImageProcess.hpp"

// 1. 加载模型
std::shared_ptr<MNN::Interpreter> net(
    MNN::Interpreter::createFromFile("model_qnn.mnn"));

// 2. 配置 QNN 后端
MNN::ScheduleConfig config;
config.type      = MNN_FORWARD_NN;  // 走 NPU (QNN Plugin 自动识别)
config.numThread = 1;

MNN::BackendConfig backendConfig;
backendConfig.precision = MNN::BackendConfig::Precision_Low; // FP16
config.backendConfig = &backendConfig;

// 3. 创建 Session
MNN::Session* session = net->createSession(config);

// 4. 获取输入输出
MNN::Tensor* input  = net->getSessionInput(session, nullptr);
MNN::Tensor* output = net->getSessionOutput(session, nullptr);

// 5. 预处理 + 推理
// ... 填充 input tensor (NHWC 或 NC4HW4) ...
net->runSession(session);

// 6. 后处理
// ... 读取 output tensor ...
```

---

## 5. 常见CV模型适配指南

### 5.1 分类模型（ResNet / MobileNet / EfficientNet）

| 项目 | 建议 |
|------|------|
| opset | 11 |
| 输入格式 | NCHW (1,3,224,224) |
| QNN 适配度 | ★★★★★ (全卷积+BN+激活，所有算子都有 QNN 实现) |
| 注意事项 | 最后的 Softmax 层可能需要在 CPU 侧处理 |

### 5.2 检测模型（YOLO / SSD / Faster R-CNN）

| 项目 | 建议 |
|------|------|
| opset | 11 |
| 输出 | 多输出头（boxes, scores, classes） |
| QNN 适配度 | ★★★★☆ |
| 注意事项 | **NMS 后处理强制在 CPU 执行**，在 `skips` 中排除；anchor 生成等也在 CPU |
| 推荐配置 | `"skips": ["NonMaxSuppression", "NMS"]` |

### 5.3 分割模型（UNet / DeepLabV3）

| 项目 | 建议 |
|------|------|
| opset | 11 |
| 输入尺寸 | 必须固定，建议 512x512 或 1024x1024 |
| QNN 适配度 | ★★★★☆ |
| 注意事项 | Upsample/Resize 层确认 QNN 支持；部分插值模式可能 fallback |

### 5.4 人脸检测模型（RetinaFace / SCRFD / CenterFace）

| 项目 | 建议 |
|------|------|
| opset | 11 |
| 输出 | 多尺度特征图 (stride 8/16/32) |
| QNN 适配度 | ★★★★☆ |
| 注意事项 | anchor decode + NMS 放 CPU；多输出头命名需匹配模型 |

### 5.5 姿态估计模型（HRNet / OpenPose）

| 项目 | 建议 |
|------|------|
| opset | 11 |
| 输入尺寸 | 固定尺寸（如 256x256 或 384x288） |
| QNN 适配度 | ★★★☆☆ |
| 注意事项 | 多分支融合需验证；heatmap 后处理在 CPU |

---

## 6. 常见问题排查

### 6.1 算子不支持

**现象：** `compilefornpu` 输出大量 fallback 日志

**解决方案：**
1. 查看 `source/backend/qnn/execution/` 确认该算子是否有 QNN 实现
2. 在 `npu_config.json` 的 `skips` 中显式排除
3. 尝试替换为 QNN 支持的等价算子（如 `HardSwish` → `ReLU6` 变体）

**当前 QNN 支持的算子类别（23 个 Execution 实现）：**

| 类别 | 算子 |
|------|------|
| 卷积 | Convolution, ConvolutionDepthwise |
| 激活 | ReLU, ReLU6, Sigmoid, ELU |
| 池化 | Pooling, Pooling3D |
| 归一化 | LayerNorm, Scale |
| 矩阵运算 | MatMul |
| 二元运算 | BinaryOp, Eltwise |
| 一元运算 | UnaryOp |
| 归约 | Reduction |
| Softmax | Softmax |
| 拼接/分割 | Concat, Pack, Unpack, Slice, StridedSlice |
| 维度变换 | Permute, Transpose, Flatten, Reshape, Squeeze, Unsqueeze, BroadcastTo |
| 索引 | GatherV2, GatherElements |
| 填充 | Padding |
| 类型转换 | Cast, FloatToInt8, Int8ToFloat |
| 查找 | ArgMax, ArgMin, TopKV2 |
| Transformer | Attention (需 `MNN_SUPPORT_TRANSFORMER_FUSE`) |

### 6.2 精度下降

**现象：** QNN 推理结果与 CPU 差异较大

**排查步骤：**
1. 确认输入预处理一致（均值/方差归一化）
2. 检查 FP16 精度损失 —— 对关键层可用 INT8 量化改善
3. 使用 `testdir` 配置进行逐层精度对比
4. HTP 部分操作有 +/-1 的数值误差（浮点舍入），CV 模型通常容忍

### 6.3 动态 Shape 不支持

**现象：** QNN 要求固定输入 shape

**解决方案：**
- 在 `npu_config.json` 中指定明确的 input shape
- ONNX导出时使用固定尺寸的 dummy_input
- 如需多尺寸，为每种尺寸单独编译 QNN 模型

### 6.4 编译 QNN 后端失败

**检查项：**
1. `QNN_SDK_ROOT` 环境变量是否正确设置
2. QNN SDK 头文件完整性：`QnnInterface.h`, `QnnTypes.h`, `HTP/QnnHtpDevice.h`
3. 交叉编译工具链是否正确（Android NDK 版本）
4. `MNN_QNN_ONLINE_FINALIZE` 是否与目标模式匹配

---

## 7. 关键文件索引

### 转换工具相关

| 文件 | 说明 |
|------|------|
| `tools/converter/README.md` | 转换工具英文文档 |
| `tools/converter/README_CN.md` | 转换工具中文文档 |
| `tools/converter/CMakeLists.txt` | 转换器构建配置 |
| `tools/converter/source/MNNConverter.cpp` | 转换器主入口 |
| `tools/converter/source/common/cli.cpp` | CLI 参数解析 + convertModel 调度 |
| `tools/converter/source/common/writeFb.cpp` | 最终序列化 + 后处理 |
| `tools/converter/source/onnx/onnxConverter.cpp` | ONNX → MNN 转换核心 |
| `tools/converter/source/optimizer/PostConverter.cpp` | 优化流水线编排 |
| `tools/script/forconvert/convertOnnx.sh` | ONNX 转换快捷脚本 |

### NPU 编译相关

| 文件 | 说明 |
|------|------|
| `tools/cpp/compilefornpu.cpp` | NPU 离线编译工具（1857行） |
| `tools/cpp/MNN2QNNModel.cpp` | QNN 离线缓存生成工具 |
| `tools/script/genQNNModelsFromMNN.py` | 批量 QNN 模型生成 |
| `source/backend/qnn/npu_convert.py` | QNN 缓存后处理脚本 |

### QNN 后端核心

| 文件 | 说明 |
|------|------|
| `source/backend/qnn/CMakeLists.txt` | QNN 后端构建配置 |
| `source/backend/qnn/backend/QNNBackend.hpp` | QNN Backend + Runtime 声明 |
| `source/backend/qnn/backend/QNNBackend.cpp` | QNN Backend 实现（2268行，核心） |
| `source/backend/qnn/backend/QNNUtils.hpp` | QNN API 动态加载、FP16转换 |
| `source/backend/qnn/backend/QNNWrapper.hpp` | Tensor/Param 包装器 |
| `source/backend/qnn/backend/QNNPerf.hpp` | HTP 性能调优 |
| `source/backend/qnn/backend/dsprpc_interface.h` | FastRPC ION 内存分配 |
| `source/backend/qnn/convertor/QNNConvertor.hpp` | 离线转换引擎 |
| `source/backend/qnn/execution/` | 23 个 QNN 算子实现 |

### 构建与配置

| 文件 | 说明 |
|------|------|
| `CMakeLists.txt:283-284` | `MNN_QNN` / `MNN_QNN_ONLINE_FINALIZE` 选项 |
| `CMakeLists.txt:676-680` | QNN 后端链接配置 |
| `prepare_qnn_deps.sh` | QNN SDK 下载脚本 |
| `project/android/qnnprepare.gradle` | Android QNN 库打包 |

### 头文件

| 文件 | 说明 |
|------|------|
| `include/MNN/MNNForwardType.h:58` | `MNN_CONVERT_QNN = 32` |

---

## 参考命令速查

```bash
# ===== 环境准备 =====
bash prepare_qnn_deps.sh                          # 下载 QNN SDK
export QNN_SDK_ROOT=/path/to/source/backend/qnn/3rdParty

# ===== 编译 MNNConvert =====
cd MNN && mkdir build && cd build
cmake .. -DMNN_BUILD_CONVERTER=true
make -j$(nproc)

# ===== ONNX → MNN =====
./MNNConvert -f ONNX --modelFile model.onnx --MNNModel model.mnn --bizCode MNN

# ===== 检查 MNN 模型 =====
./MNNDump2Json model.mnn > model.json

# ===== 编译带 QNN 的 MNN (Android) =====
cmake .. \
  -DMNN_QNN=ON \
  -DQNN_SDK_ROOT=$QNN_SDK_ROOT \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_NATIVE_API_LEVEL=android-28
make -j$(nproc)

# ===== MNN → QNN 离线编译 =====
./compilefornpu model.mnn model_qnn.mnn npu_config.json

# ===== MNN → QNN (指定SOC) =====
./MNN2QNNModel /path/to/qnn_sdk 43 73 model.mnn ./qnn_output 1,3,224,224

# ===== 批量生成 QNN 模型 =====
python tools/script/genQNNModelsFromMNN.py
```

---

> **文档版本:** v1.0 | **基于 MNN master 分支 (0bff03cb)** | **分析日期: 2026-07-03**
