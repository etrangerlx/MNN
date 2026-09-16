# MNN to QNN NPU Conversion Flow

This document describes the end-to-end pipeline for converting an MNN model into a QNN-compatible format for Qualcomm HTP (NPU) inference.

---

## Overview

```
HuggingFace 模型
    ↓ (llmexport.py / Python export)
ONNX / MNN 模型 (llm.mnn + llm.mnn.weight)
    ↓
┌─────────────────────────────────────────────────────────────┐
│  generate_llm_qnn.py (Python entry, 4 steps)                │
│                                                             │
│  Step 1: generateIO        → Generate reference I/O data    │
│  Step 2: compilefornpu     → Graph splitting + Plugin op    │
│  Step 3: npu_convert.py    → QNN SDK offline compile        │
│  Step 4: output_qnn        → Move artifacts, write config   │
└─────────────────────────────────────────────────────────────┘
    ↓
qnn/llm.mnn (with Plugin ops pointing to QNN binaries) + QNN Context Binary
    ↓
MNN QNNBackend → QNNRuntime → QNN SDK inference
```

---

## Step 1: generateIO — Generate Reference Input/Output

**Entry:** [tools/cpp/generateIO.cpp](../../tools/cpp/generateIO.cpp)

**Purpose:** For each supported input shape, run the MNN model once and save the exact input / output tensors as `.mnn` files. These are used later for **accuracy verification** when compiling NPU subgraphs.

**Python side configuration** ([transformers/llm/export/npu/generate_llm_qnn.py](../../transformers/llm/export/npu/generate_llm_qnn.py)):
- LLM models typically define **2 shapes**: prefill (`[seq_len, 1, hidden_size]`) and decode (`[1, 1, hidden_size]`).
- Visual models may define multiple image resolutions.

**Generated files:**
```
tmp/testdir/0/input.mnn    ← prefill inputs
tmp/testdir/0/output.mnn   ← prefill reference output
tmp/testdir/1/input.mnn    ← decode inputs
tmp/testdir/1/output.mnn   ← decode reference output
```

---

## Step 2: compilefornpu — Graph Splitting (Core)

**Entry:** [tools/cpp/compilefornpu.cpp](../../tools/cpp/compilefornpu.cpp)

This is the **core** of the conversion process. It analyzes the full MNN graph and splits it into **CPU subgraphs** and **NPU subgraphs**, replacing each NPU subgraph with a single `Plugin` op.

### 2.1 Break Ops (What to Keep on CPU)

`isBreakOp()` determines which ops cannot be offloaded to QNN:
- Control flow: `While`, `If`, `Where`
- KV-Cache Attention / `LinearAttention` (dynamic KV-Cache not supported by QNN)
- `Segment`, `Unique`, `NonMaxSuppressionV2`

```
MNN Graph
 [Op1] ──►[Op2]──►[Attention]──►[Op5]──►[Op6]
    │        │    break ↑           │        │
    └─NPU───┘    CPU    └────NPU────┘
```

### 2.2 Extra Break: attention_mask → Attention Path

`_findMaskToAttentionOps()` ([compilefornpu.cpp:179](../../tools/cpp/compilefornpu.cpp:179)):
- Forward-propagates from `attention_mask` input through the graph.
- Stops at `Attention` / `LinearAttention` ops.
- Back-traces and marks **all intermediate ops on that path** as additional break ops.
- Rationale: mask-related computations (position encoding prep, etc.) are fragile on QNN and safer on CPU.

### 2.3 Sub-Module Creation and Splitting

`_createSubModuleInfo()` ([compilefornpu.cpp:643](../../tools/cpp/compilefornpu.cpp:643)):
1. Collects all ops needed for the target inputs/outputs via BFS/DFS.
2. Splits the graph at every `Break Op` into multiple `SubModuleInfo`.
3. For each sub-module, computes `inputs` / `outputs` tensor indexes.
4. **Shape-const splitting:** if an op has a shape input that is not a compile-time constant, it is also split out independently.

### 2.4 Compiling an NPU Sub-Module

`_compileSubModule()` / `_compileWholeModule()` ([compilefornpu.cpp:1058](../../tools/cpp/compilefornpu.cpp:1058)):

For each valid NPU sub-module:
1. **Load as MNN Module** with `config.type = MNN_CONVERT_QNN` (offline conversion mode).
2. **Offline conversion:** `Module::load()` internally triggers `QNNConvertor`, translating every MNN Op into QNN `Qnn_OpConfig_t`, producing `graph.cpp` + `graph.bin`.
3. **Accuracy check:** Runs `onForward()` with reference inputs and compares against Step 1 outputs. Relative error must be `< 10%`.
4. **Speed comparison:** Runs NPU vs CPU for 20 iterations; prints latency to decide whether to keep QNN.
5. **Fuse into Plugin Op:** Wraps the NPU sub-graph as an `OpType_Plugin` in the new MNN model.

**Plugin attributes carry:**
- `path`: QNN binary file path
- `inputs` / `outputs`: Tensor name mappings (e.g. `t23` = tensor index 23)
- `allGraphName`: List of graph names
- `allInputShape`: All input shapes concatenated into one 1-D list
- `o_{shapeIndex}_{outputIndex}`: Output tensor shape / type / data-format metadata

### 2.5 Outputs of this Step

- `qnn/llm.mnn` — New MNN model where NPU sub-graphs are replaced by `OpType_Plugin` ops.
- `npu_postreat.json` — Merge info for Step 3:
  ```json
  {
      "type": "QNN",
      "merge": {
          "res/graph0.bin": ["res/graph0", "res/graph1_0"]
      },
      "cache": "res"
  }
  ```

---

## Step 3: npu_convert.py — QNN SDK Offline Compile

**Entry:** [source/backend/qnn/npu_convert.py](../../source/backend/qnn/npu_convert.py)

Reads `npu_postreat.json` and runs the QNN SDK offline toolchain to produce the final DSP binaries.

### 3.1 process_src — Compile a Single Sub-Graph to `.so`

([npu_convert.py:57](../../source/backend/qnn/npu_convert.py:57))

```
graph0/
  ├── graph0.cpp       ← QNN API calls (generated by QNNConvertor)
  ├── graph0_0.raw     ← weights / constants
  ├── graph0_1.raw
  └── ...
        ↓ tar -cf graph0.bin *.raw
        ↓ python3 qnn-model-lib-generator
              -c graph0.cpp -b graph0.bin -t x86_64-linux-clang -o graph0/
graph0/x86_64-linux-clang/libgraph0.so
```

- `qnn-model-lib-generator`: Packages C++ API code + raw weight blobs into a loadable `.so`.
- Intermediate `.raw` / `.bin` files are cleaned up after success.

### 3.2 process_merge — Merge Multiple Graphs into a Context Binary

([npu_convert.py:137](../../source/backend/qnn/npu_convert.py))

For all sub-graphs under the same `merge` key (e.g. `graph0.bin`):

1. Writes `htp_backend_extensions_{N}.json` — HTP backend config:
   ```json
   {
       "graphs": [{"vtcm_mb": 8, "O": 3.0, "graph_names": ["graph0", "graph1_0"]}],
       "devices": [{"soc_id": 57, "dsp_arch": "v75", "cores": [{"perf_profile": "burst"}]}],
       "context": {"weight_sharing_enabled": true}
   }
   ```

2. Writes `context_config_{N}.json`.

3. Invokes `qnn-context-binary-generator`:
   ```
   qnn-context-binary-generator
       --model libgraph0.so,libgraph1_0.so
       --backend libQnnHtp.so
       --binary_file graph0.bin
       --config_file context_config_0.json
       --output_dir res/
   ```
   **Output:** `res/graph0.bin.cache` — the final DSP Context Binary loaded at runtime.

### 3.3 Parallelization

Both `process_src` and `process_merge` use `ProcessPoolExecutor` for parallelism.

---

## Step 4: output_qnn — Collect Artifacts

([generate_llm_qnn.py:246](../../transformers/llm/export/npu/generate_llm_qnn.py:246))

- Moves `tmp/cache_path/qnn/` → `<model_dir>/qnn/`.
- Generates `config_qnn.json`, used by the MNN LLM runtime to locate the QNN model:
  ```json
  {
      "llm_model": "qnn/llm.mnn",
      "chunk_limits": [128, 1]
  }
  ```

---

## Runtime Loading

### 6.1 QnnRuntime Initialization

[source/backend/qnn/backend/QNNBackend.hpp](../../source/backend/qnn/backend/QNNBackend.hpp)

`QnnRuntime::create()`:
1. `QnnInterface_getProviders()` — discover QNN provider
2. `qnnInterface.backendCreate()` — create QNN backend
3. `qnnInterface.deviceCreate()` — create HTP device
4. `qnnInterface.contextCreateFromBinary()` — load Context from `.bin.cache`
5. `qnnInterface.graphRetrieve()` — retrieve Graph handles

`QnnBackend::onCreate()`:
- Parses `path`, `graph_names`, `inputs`, `outputs` from the `Plugin` op attributes.
- Calls `createContextAndGraph()` to wire everything together.

### 6.2 Execution Flow

```
onExecuteBegin()
    └── executeGraph()
            └── qnnInterface.graphExecute()
                    ├── Input:  MNN Tensor → QNN Tensor
                    │            (NC4HW4→NCHW, FP16/FP32 cast if needed)
                    └── Output: QNN Tensor → MNN Tensor
onExecuteEnd()
```

### 6.3 QNNConvertor (Offline Compilation)

[source/backend/qnn/convertor/QNNConvertor.hpp](../../source/backend/qnn/convertor/QNNConvertor.hpp)

- `QNNConvertorInterface::onCreate()` intercepts `MNN_CONVERT_QNN` module load requests.
- Iterates every MNN Op in the sub-graph, dispatches to the matching QNN Op Creator (e.g. `QNNConvolution`, `QNNMatMul`, `QNNAttention`).
- Each Creator translates MNN parameters to `Qnn_OpConfig_t`.
- Calls `QnnGraph_finalize()` to seal the graph and emit `graph.cpp` + `graph.bin`.

---

## Key Files

| File | Role |
|------|------|
| [transformers/llm/export/npu/generate_llm_qnn.py](../../transformers/llm/export/npu/generate_llm_qnn.py) | Python entry script orchestrating the 4-step conversion |
| [tools/cpp/generateIO.cpp](../../tools/cpp/generateIO.cpp) | Step 1: Reference I/O generation |
| [tools/cpp/compilefornpu.cpp](../../tools/cpp/compilefornpu.cpp) | Step 2: Core graph splitting and Plugin op generation |
| [source/backend/qnn/npu_convert.py](../../source/backend/qnn/npu_convert.py) | Step 3: QNN SDK offline compilation |
| [source/backend/qnn/backend/QNNBackend.hpp](../../source/backend/qnn/backend/QNNBackend.hpp) | QNN runtime Backend header |
| [source/backend/qnn/convertor/QNNConvertor.hpp](../../source/backend/qnn/convertor/QNNConvertor.hpp) | MNN Op → QNN Op offline converter |
| [source/backend/qnn/execution/](../../source/backend/qnn/execution/) | Individual QNN op implementations (Conv, MatMul, Attention, etc.) |

---

## Appendix: How MNN Generates `.cpp` — Stub QNN API Interception

When `compilefornpu` runs with `MNN_CONVERT_QNN`, MNN does **not** load the real Qualcomm QNN SDK. Instead, it uses a **stub API layer** that intercepts every QNN API call and redirects it into C++ code generation.

### A.1 Architecture

```
compilefornpu / Module::load(..., MNN_CONVERT_QNN)
    ↓
QnnRuntime::create()
    ↓  (gQnnConvertorInterface — stub, not real QNN SDK)
QnnBackend::onCreate() ──► createContextAndGraph()

For each Tensor:
    tensorCreateGraphTensor() → QnnConvertorTensor_CreateGraphTensor (stub)
                                    └── RecordTensor() → write Tensor C++
                                    └── DumpBuffer()   → write {name}.raw

For each Op:
    addNodeToGraph()
        └── graphAddNode() → QnnConvertorGraph_AddNode (stub)
                                 └── RecordNode() → write Op C++

Finalize:
    graphFinalize() → QnnConvertorGraph_Finalize (stub)
                          └── RecordEnd() → close & flush file
```

### A.2 Why Stubs?

The QNN offline toolchain (`qnn-model-lib-generator` + `qnn-context-binary-generator`) requires:
1. A `.cpp` file that calls QNN C++ APIs.
2. A `.bin` archive containing static tensor data (weights / constants).

MNN does not need a real HTP device or real QNN SDK to produce these files. By replacing the real QNN API with stubs, the conversion can run on any x86_64 build machine.

### A.3 Stub API Layer

**File:** [source/backend/qnn/convertor/QNNConvertorInterface.cpp](../../source/backend/qnn/convertor/QNNConvertorInterface.cpp)

Key stub functions (all return `QNN_SUCCESS` and call `Record*`):

| Stub Function | Real QNN API | What it records |
|---|---|---|
| `QnnConvertorGraph_Create()` | `QnnGraph_create()` | Open `{graph}.cpp`, emit header, `QnnModel` init |
| `QnnConvertorGraph_AddNode()` | `QnnGraph_addNode()` | Emit `addNode(...)` call (params, inputs, outputs) |
| `QnnConvertorTensor_CreateGraphTensor()` | `QnnTensor_createGraphTensor()` | Emit tensor declaration + `addTensor(...)`; `DumpBuffer()` for static data |
| `QnnConvertorGraph_Finalize()` | `QnnGraph_finalize()` | Emit trailer, close file |

All other QNN APIs (`backendCreate`, `deviceCreate`, `contextCreate`, `logCreate`, ...) are also stubbed to no-ops returning success.

The stubs are compiled conditionally via `-DENABLE_QNN_CONVERT_MODE` in [source/backend/qnn/CMakeLists.txt](../../source/backend/qnn/CMakeLists.txt:33).

### A.4 Code Generation Details

**File:** [source/backend/qnn/convertor/QNNConvertor.cpp](../../source/backend/qnn/convertor/QNNConvertor.cpp)

#### `RecordBegin()` (QNNConvertor.cpp:57)

Opens `{OutputDir}/{GraphNameSymbol}.cpp` and writes the file header:
- `#include "QnnModel.hpp"`, `#include "QnnOpDef.h"`
- `extern "C" QnnModel_composeGraphs(...)` function signature
- `QnnModel {GraphNameSymbol};`
- `graph0.initialize(...)` call

Graph name is derived from the last directory segment of `OutputDir` (set by `rtmgr->setCache()` in `compilefornpu`).

#### `RecordTensor()` (QNNConvertor.cpp:79)

For each Tensor, emits C++ code:
1. **Dimensions array:** `uint32_t dimensions_name[] = {d0, d1, ...};`
2. **Quantization parameters:** `scale`, `offset`, axis-scale-offset, blockwise expansion (if quantized)
3. **`Qnn_Tensor_t` declaration and initialization:**
   ```cpp
   Qnn_Tensor_t tensor_name = QNN_TENSOR_INIT;
   tensor_name.v1.name = "name";
   tensor_name.v1.type = QNN_TENSOR_TYPE_APP_WRITE;   // or STATIC, APP_READ
   tensor_name.v1.dataType = QNN_DATATYPE_FLOAT_16;   // or INT_32, etc.
   tensor_name.v1.dimensions = dimensions_name;
   ```
4. **`addTensor()` call** (for INPUT and STATIC tensors)

For **STATIC** tensors, it additionally calls `DumpBuffer()` which writes the raw weight bytes to `{name}.raw`.

#### `RecordNode()` (QNNConvertor.cpp:124)

For each Op, emits C++ code:
1. **Param array:** `Qnn_Param_t params_op[] = {param_0, param_1, ...};`
2. **Input name array:** `const char* inputs_op[] = {"tensor_a", "tensor_b"};`
3. **Output tensor array:** `Qnn_Tensor_t outputs_op[] = {tensor_c};`
4. **`addNode()` call:**
   ```cpp
   VALIDATE(graph0.addNode(QNN_OPCONFIG_VERSION_1, "opName",
                           "qti.aisw", "MatMul",
                           params_op, numParams,
                           inputs_op, numInputs,
                           outputs_op, numOutputs), err);
   ```

#### `RecordEnd()` (QNNConvertor.cpp:146)

Emits the trailer and closes the file:
- `QnnModel* models[] = {&graph0};`
- `getGraphInfoFromModels(models, ...)`
- `return err;`
- `QnnModel_freeGraphsInfo(...)`

### A.5 Generated Files on Disk

After `compilefornpu` Step 2 completes (but before Step 3), the cache directory contains:

```
tmp/res/graph0/
  ├── graph0.cpp              ← C++ code (QNN API calls)
  ├── tensor_weight_0.raw     ← static weight data
  ├── tensor_weight_1.raw
  ├── tensor_bias_0.raw
  └── ...
```

These are consumed by Step 3 (`npu_convert.py`):
1. `tar -cf graph0.bin *.raw` — packs raw data into `.bin`
2. `qnn-model-lib-generator -c graph0.cpp -b graph0.bin` — compiles `.cpp` + `.bin` → `libgraph0.so`
3. `qnn-context-binary-generator --model lib*.so` → final `graph0.bin.cache`

### A.6 Key Files

| File | Role |
|------|------|
| [source/backend/qnn/convertor/QNNConvertorInterface.cpp](../../source/backend/qnn/convertor/QNNConvertorInterface.cpp) | Stub QNN API implementation — intercepts QNN calls for offline code generation |
| [source/backend/qnn/convertor/QNNConvertor.cpp](../../source/backend/qnn/convertor/QNNConvertor.cpp) | C++ code generator — `RecordBegin/Tensor/Node/End` |
| [source/backend/qnn/convertor/QNNConvertor.hpp](../../source/backend/qnn/convertor/QNNConvertor.hpp) | `QNNConvertor` / `QNNTranslator` class declarations |
| [source/backend/qnn/CMakeLists.txt:33](../../source/backend/qnn/CMakeLists.txt:33) | `-DENABLE_QNN_CONVERT_MODE` compile flag |
