# 步骤 5：扩展后端与性能优化

> **目标**：将算子扩展到其他硬件后端（Metal/OpenCL/Vulkan/CUDA），并进行性能优化。
>
> **前置条件**：步骤 4 已通过（CPU 单元测试全部通过）。
>
> **注意**：这一步通常逐个后端实施，不需要一次全部完成。

---

## 5.0 优化路线

```
CPU 基础实现（已完成）
│
├─ CPU 多线程优化
│   └─ SIMD 优化（ARM NEON / x86 SSE/AVX）
│
├─ Metal 后端（Apple GPU）
│
├─ OpenCL 后端
│
├─ Vulkan 后端
│
├─ CUDA 后端
│
└─ QNN 后端（Qualcomm AI Engine）
```

每完成一个后端，都可以用步骤 4 的单元测试来验证正确性。

---

## 5.1 CPU 多线程优化

在 `onExecute` 中使用 MNN 的线程池：

```cpp
ErrorCode CPUMyCustomOp::onExecute(const std::vector<Tensor*>& inputs,
                                    const std::vector<Tensor*>& outputs) {
    auto input = inputs[0];
    auto output = outputs[0];

    int threadCount = static_cast<CPUBackend*>(backend())->threadNumber();
    int totalSize = input->elementSize();

    MNN_CONCURRENCY_BEGIN(tId, threadCount) {
        int start = tId * totalSize / threadCount;
        int end = (tId + 1) * totalSize / threadCount;
        for (int i = start; i < end; ++i) {
            output->host<float>()[i] = /* 计算 */;
        }
    }
    MNN_CONCURRENCY_END();

    return NO_ERROR;
}
```

---

## 5.2 Metal 后端

### 5.2.1 创建实现文件

在 `source/backend/metal/` 下创建 `MetalMyCustomOp.hpp` 和 `MetalMyCustomOp.cpp`：

**MetalMyCustomOp.hpp**：
```cpp
#ifndef MetalMyCustomOp_hpp
#define MetalMyCustomOp_hpp

#include "MetalExecution.hpp"

namespace MNN {
class MetalMyCustomOp : public MetalExecution {
public:
    MetalMyCustomOp(Backend* backend, const Op* op);
    virtual ~MetalMyCustomOp() = default;

    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) override;
    virtual void onEncode(const std::vector<Tensor*>& inputs,
                          const std::vector<Tensor*>& outputs,
                          id<MTLComputeCommandEncoder> encoder) override;
private:
    id<MTLComputePipelineState> mPipeline;
    MTLSize mThreads;
    MTLSize mThreadgroupSize;
};
} // namespace MNN
#endif
```

**MetalMyCustomOp.cpp**（关键部分）：
```cpp
#include "MetalMyCustomOp.hpp"
#include "backend/metal/MetalBackend.hpp"

namespace MNN {

MetalMyCustomOp::MetalMyCustomOp(Backend* backend, const Op* op)
    : MetalExecution(backend) {
    auto mtbn = static_cast<MetalBackend*>(backend);
    mPipeline = [mtbn->context() pipelineWithName:@"my_custom_op"]; // Metal shader 名
}

ErrorCode MetalMyCustomOp::onResize(const std::vector<Tensor*>& inputs,
                                     const std::vector<Tensor*>& outputs) {
    // 计算 thread group 大小
    int totalSize = outputs[0]->elementSize();
    mThreads = {(NSUInteger)totalSize, 1, 1};
    mThreadgroupSize = {256, 1, 1};
    return NO_ERROR;
}

void MetalMyCustomOp::onEncode(const std::vector<Tensor*>& inputs,
                                const std::vector<Tensor*>& outputs,
                                id<MTLComputeCommandEncoder> encoder) {
    auto input  = inputs[0];
    auto output = outputs[0];
    [encoder setComputePipelineState:mPipeline];
    MetalBackend::setTensor(input, encoder, 0);
    MetalBackend::setTensor(output, encoder, 1);
    [encoder dispatchThreads:mThreads threadsPerThreadgroup:mThreadgroupSize];
}

// 注册
class MetalMyCustomOpCreator : public MetalBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs,
                                const MNN::Op* op, Backend* backend) const {
        return new MetalMyCustomOp(backend, op);
    }
};
REGISTER_METAL_OP_CREATOR(MetalMyCustomOpCreator, OpType_MyCustomOp);

} // namespace MNN
```

### 5.2.2 编写 Metal Shader

在 `source/backend/metal/shader/` 下添加 `my_custom_op.metal` 或将 kernel 写入已有的 shader 文件。

### 5.2.3 更新工程

```bash
cd source/backend/metal
python3 MetalCodeGen.py .
```

---

## 5.3 OpenCL 后端

### 5.3.1 编写 Kernel

在 `source/backend/opencl/execution/cl/` 下创建 `my_custom_op.cl`：

```opencl
__kernel void my_custom_op(
    __read_only image2d_t input,
    __write_only image2d_t output,
    __private const int width,
    __private const int height
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    float4 in = read_imagef(input, SAMPLER, (int2)(x, y));
    float4 out = in; // ← 替换为实际计算
    write_imagef(output, (int2)(x, y), out);
}
```

### 5.3.2 生成 Kernel 映射

```bash
cd source/backend/opencl/execution/cl
python3 opencl_codegen.py
```

### 5.3.3 创建实现文件

在 `source/backend/opencl/execution/` 下创建 `MyCustomOp.h` 和 `MyCustomOp.cpp`，注册：

```cpp
OpenCLCreatorRegister<TypedCreator<MyCustomOp<cl_data_t>>> __my_custom_op(OpType_MyCustomOp);
```

---

## 5.4 Vulkan 后端

### 5.4.1 生成模板代码

```bash
cd source/backend/vulkan/image/compiler
python3 VulkanCodeGen.py
```

### 5.4.2 编写 Compute Shader

在 `source/backend/vulkan/image/execution/glsl/` 下创建 `myCustomOp.comp`。

### 5.4.3 编译 Shader

```bash
cd source/backend/vulkan/image/compiler
python3 makeshader.py
```

### 5.4.4 实现并注册

```cpp
class VulkanMyCustomOpCreator : public VulkanBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new VulkanMyCustomOp(op, backend);
    }
};
static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_MyCustomOp, new VulkanMyCustomOpCreator);
    return true;
}();
```

---

## 5.5 CUDA 后端

在 `source/backend/cuda/execution/` 下添加 `.cu` 和 `.cuh` 文件，编写 CUDA kernel 并注册。

---

## 5.6 QNN 后端（Qualcomm AI Engine）

> **关键差异**：QNN 后端采用**显式注册**模式，与其他所有后端（自动注册）完全不同。缺失任何一步都会导致运行时 "Not registered type N" 错误。

### 5.6.1 QNN 架构概述

```
source/backend/qnn/
├── backend/          ← QNNBackend 核心（图构建、tensor 管理）
│   ├── QNNBackend.cpp / .hpp
│   ├── QNNUtils.cpp / .hpp     ← **算子注册入口 registerQNNOps()**
│   └── QNNWrapper.cpp / .hpp
├── execution/        ← 各算子实现（每个 Op 一个 .cpp + .hpp）
│   ├── QNNCommonExecution.cpp / .hpp   ← 基类
│   ├── QNNConvolution.cpp / .hpp
│   ├── QNNPool.cpp / .hpp
│   └── ...
└── convertor/        ← 离线转换模式
```

### 5.6.2 创建算子实现文件

在 `source/backend/qnn/execution/` 下创建 `QNNMyOp.hpp` 和 `QNNMyOp.cpp`。

**模板 — QNNMyOp.hpp**：

```cpp
#ifndef MNN_QNNMYOP_HPP
#define MNN_QNNMYOP_HPP

#include "QNNCommonExecution.hpp"
#include "QnnTypes.h"

namespace MNN {
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE

class QNNMyOp : public QNNCommonExecution {
public:
    QNNMyOp(Backend *backend, const Op *op) : QNNCommonExecution(backend, op) {}
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs,
                               const std::vector<Tensor *> &outputs) override;
};
#endif
} // namespace QNN
} // namespace MNN

#endif
```

**模板 — QNNMyOp.cpp**：

```cpp
#include "QNNMyOp.hpp"

namespace MNN {
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE

ErrorCode QNNMyOp::onEncode(const std::vector<Tensor *> &inputs,
                             const std::vector<Tensor *> &outputs) {
    // Step 1: 清理上次调用的状态
    mParams.clear();
    mInputs.clear();
    mOutputs.clear();

    // Step 2: 设置 QNN 算子类型名（见 QnnOpDef.h）
    mNodeType = "QnnOpTypeName";  // 如 "ResizeBilinear", "Concat", "PoolAvg2d"

    // Step 3: 创建参数（使用基类辅助方法）
    this->createParamScalar("param_name", (uint32_t)value);     // 标量参数
    // this->createParamTensor("param_name", dtype, dims, ptr);  // 张量参数（可选）

    // Step 4: 添加节点到图（自动处理 inputs/outputs 的 tensor 获取）
    this->addNodeCommon(inputs, outputs);

    return NO_ERROR;
}

// Step 5: 创建 Creator 并注册（参见 5.6.3）
class QNNMyOpCreator : public QnnBackend::Creator {
public:
    virtual QNNCommonExecution * onCreate(const std::vector<Tensor*>& inputs,
                                          const std::vector<Tensor*>& outputs,
                                          const MNN::Op* op,
                                          Backend* backend) const override {
        return new QNNMyOp(backend, op);
    }
};

REGISTER_QNN_OP_CREATOR(QNNMyOpCreator, OpType_MyOp)
#endif
} // namespace QNN
} // namespace MNN
```

### 5.6.3 QNN 三步显式注册（⚠️ 最关键！）

与其他后端不同，QNN **不会**自动注册算子。必须手动完成以下三步：

```
┌─────────────────────────────────────────────────────────────┐
│  Step A: 算子 .cpp 中调用 REGISTER_QNN_OP_CREATOR           │
│          → 生成函数 void ___Creator__OpType__()              │
├─────────────────────────────────────────────────────────────┤
│  Step B: QNNUtils.hpp 中添加 extern 声明                     │
│          → extern void ___Creator__OpType__();               │
├─────────────────────────────────────────────────────────────┤
│  Step C: QNNUtils.cpp 的 registerQNNOps() 中显式调用         │
│          → ___Creator__OpType__();                           │
└─────────────────────────────────────────────────────────────┘
```

**缺失任何一步 = 编译通过但运行时报 "Not registered type N"**。

这是因为 QNN 运行时通过 dlopen 动态加载，静态初始化器不可靠，所以必须在 `registerQNNOps()` 中显式调用每个注册函数。

### 5.6.4 参数创建辅助方法

基类 `QNNCommonExecution` 提供以下方法：

```cpp
// 标量参数
this->createParamScalar("name", boolValue);
this->createParamScalar("name", (uint32_t)intValue);
this->createParamScalar("name", (int)intValue);
this->createParamScalar("name", (float)floatValue);

// 张量参数（需要拷贝数据到 QNN）
this->createParamTensor("name", QNN_DATATYPE_UINT_32, {dims}, (void*)dataPtr);

// 添加节点到图（自动获取 inputs/outputs 的 native tensor）
this->addNodeCommon(inputs, outputs);           // 所有 input/output
this->addNodeCommon(inputs, outputs, N, M);     // 前 N 个 input, 前 M 个 output
```

### 5.6.5 MNN 参数到 QNN 参数的映射

在 `onEncode` 中通过 `mOp->main_as_Xxx()` 获取 MNN 算子参数，然后映射为 QNN 参数。参考 `QNNPool.cpp`、`QNNBinary.cpp` 等现有实现。

**常用 QNN 算子类型常量**（定义在 `<QNN/QnnOpDef.h>`）：

| QNN 宏 | 字符串值 | 说明 |
|--------|---------|------|
| `QNN_OP_RESIZE_BILINEAR` | `"ResizeBilinear"` | 双线性插值 |
| `QNN_OP_RESIZE` | `"Resize"` | 通用 resize（支持 nearest/linear/cubic） |
| `QNN_OP_CONCAT` | `"Concat"` | 拼接 |
| `QNN_OP_POOL_AVG_2D` | `"PoolAvg2d"` | 平均池化 |
| `QNN_OP_RELU` | `"Relu"` | ReLU 激活 |

完整列表见 `/mnt/d/SoftWare/qairt/<version>/include/QNN/QnnOpDef.h`。

### 5.6.6 格式转换注意事项

- **NC4HW4 → NHWC**：MNN 内部使用 NC4HW4 格式，QNN 使用 NHWC。tensor 的 shape 通过 `getNHWCShape()` 转换。
- **Axis 转换**：如果算子涉及 axis 参数（如 Concat、Reduce），需要检查是否需要从 NC4HW4 约定转换为 NHWC 约定。4D 映射：`{0→0, 1→3, 2→1, 3→2}`
- **格式检测**：通过 `TensorUtils::getDescribe(tensor)->dimensionFormat` 获取当前格式，仅对 `MNN_DATA_FORMAT_NC4HW4` 进行转换。

### 5.6.7 编译 & 测试

```bash
# 编译
cd build
cmake --build . --target MNN2QNNModel -j8

# 测试（注意: 输出目录必须预先存在！否则会 SIGSEGV）
mkdir -p ./output
./MNN2QNNModel /path/to/qnn/sdk soc_id arch_id model.mnn ./output/ 1 input_shape
```

### 5.6.8 调试技巧

| 问题 | 排查方法 |
|------|---------|
| `Not registered type N` | 检查三步注册是否完整：extern 声明 + registerQNNOps() 调用 + REGISTER_QNN_OP_CREATOR |
| `Don't support type N` | `onCreate` 返回了 nullptr，检查 Creator 实现 |
| QNN 图验证失败 | 检查参数映射是否正确、axis 是否转换、tensor shape 是否匹配 |
| SIGSEGV (NULL fp) | 输出目录不存在，`mkdir -p` 提前创建 |
| SIGSEGV (其他) | 用 `gdb -batch -ex run -ex bt --args ...` 定位 |

---

## 步骤 5 测试标准

### 测试方法

每完成一个后端，都用同一套单元测试验证：

```bash
cd build

# CPU（已通过）
./run_test.out op/MyCustomOp

# Metal（需要 Mac + Metal 支持）
./run_test.out op/MyCustomOp 0 0 3    # 3 = Metal

# OpenCL
./run_test.out op/MyCustomOp 0 0 6    # 6 = OpenCL

# Vulkan
./run_test.out op/MyCustomOp 0 0 7    # 7 = Vulkan

# CUDA
./run_test.out op/MyCustomOp 0 0 1    # 1 = CUDA
```

> **后端 ID 参考**：0=CPU, 1=CUDA, 3=Metal, 6=OpenCL, 7=Vulkan

### 通过标准

- [ ] 目标后端的单元测试全部 `passed`
- [ ] 计算结果与 CPU 版本一致（在浮点精度范围内）

### 部分完成也是可以接受的

```
✅ 已完成：
- Schema 定义
- 形状计算
- CPU 实现 + 单元测试通过
- Metal 实现 + 单元测试通过

⏳ 待完成：
- OpenCL 后端
- Vulkan 后端
- CUDA 后端
- SIMD 性能优化
```

---

## 完成

**恭喜！当所需的后端测试全部通过后，算子支持工作完成。**
