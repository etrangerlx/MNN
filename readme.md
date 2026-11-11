# 背景
MNN本地路径：/home/leixian/workspaces/projects/AIInfer/MNN,版本是：3.6.2
# 任务
1. 借鉴MNN的推理框架，从零开始构建一个推理框架
2. 将MNN中的CPU推理部分移植到新的框架中

# 实现说明

## 目录结构

```
include/infer/        Tensor / Op / Backend / Execution / Interpreter / Model / ThreadPool
source/core/          模型加载（MNN flatbuffer）、会话、静态 shape 推断、内存规划
source/backend/cpu/   CPU 算子实现（Convolution / Pooling / Unary / Shape / InnerProduct）
tools/                infer_bench、infer_compare、mnn_weighted_model
tests/test_ops.cpp    42 项算子单元测试
models/               带确定性权重的 benchmark 模型副本
```

## 框架要点

- 后端抽象：`Backend` / `Execution`，CPU 后端为每种 `OpType` 注册一个 `Execution` 实现。
- 静态 shape 推断：`Session::resize` 依次调用各算子 `onResize`，不做运行时重排。
- 形状子图常量折叠：`Shape → StridedSlice → Pack → Reshape` 这类形状计算链在 resize 阶段直接求值，
  保证 SqueezeNet 等模型能在分配内存前得到正确的 Reshape 目标。
- 内存规划：按张量活跃区间（liveness）做 best-fit 复用，arena 按 64 字节对齐，只分配一次。
- 线程池：常驻线程池，`parallelFor` 由调用线程一起参与计算。
- 模型加载：直接读取 MNN 3.6.2 的 flatbuffer schema，支持 legacy oplists 与加权模型。

## 构建与测试

使用系统 cmake / gcc，脚本会自动链接 `/home/leixian/workspaces/projects/AIInfer/MNN/build/libMNN.so`
（可用 `MNN_BUILD_DIR` 覆盖）：

```bash
./scripts/build.sh     # 编译框架、单元测试与命令行工具
./scripts/test.sh      # 编译 + 单元测试 + models/ 下全部模型的精度对比
```

常用选项：

```bash
SKIP_BUILD=1 ./scripts/test.sh          # 跳过编译，直接测试已有产物
THREADS=8 RUNS=3 ./scripts/test.sh      # 指定线程数与性能统计次数
BUILD_DIR=/tmp/infer-build ./scripts/build.sh
```

也可以手动执行单个工具：

```bash
./build/infer_test
./build/infer_bench models/mobilenet_v2_1.0_224.tflite.mnn --threads 8 --runs 3
./build/infer_compare models/mobilenet_v2_1.0_224.tflite.mnn --threads 8 --runs 3
```

`mnn_weighted_model` 可为无权重模型注入确定性权重：

```bash
./build/mnn_weighted_model /home/leixian/workspaces/projects/AIInfer/MNN/benchmark/models/resnet-v2-50.mnn /tmp/resnet-weighted.mnn
```

# 验证结果

输入为确定性伪随机数据，对比对象为本地 MNN 3.6.2（CPU / High 精度），判据 `maxAbs < 1e-3`。
`models/` 中的 libMNN benchmark 模型（inception-v3、resnet-v2-50 等）本身不含卷积权重，
只能用于耗时测试，`test.sh` 会自动识别并跳过。

| 模型 | 输出 | maxAbs（infer vs MNN） | 耗时 infer | 耗时 MNN |
| --- | --- | --- | --- | --- |
| mobilenet_v1.caffe | [1,1000,1,1] | 4.9e-07 | 175 ms | 11.3 ms |
| mobilenet_v2.caffe | [1,1000,1,1] | 6.9e-08 | 189 ms | 15.3 ms |
| mobilenet_v2_1.0_224.tflite | [1,1001] | 3.1e-07 | 81 ms | 4.4 ms |
| squeezenet_v1.0.caffe | [10,1000,1,1] | 1.1e-06 | 1116 ms | 90.6 ms |
| squeezenet_v1.1.caffe | [1,1000,1,1] | 5.7e-07 | 55 ms | 4.2 ms |
| Portrait.tflite | [1,21,257,257] | 1.9e-05 | 134 ms | 16.1 ms |

`Transpose` / `Unpack` / `SliceTf` 均已实现；`Interp` 按 MNN 的几何推导规则计算采样变换
（`alignCorners` 用 `(in-1)/(out-1)`，`halfPixelCenters` 用 `in/out` 加半像素偏移，其余用 `in/out`），
与 MNN CPU 后端逐元素一致。

耗时为本机（i7-10875H，4 线程）单次运行的值；本实现为可读性优先的通用 im2col + 朴素 GEMM，
未做 MNN 那样的 SIMD/winograd 优化，因此单次推理比 MNN 慢。
