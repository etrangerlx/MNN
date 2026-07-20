# QNN Backend: Support Deconvolution (ConvTranspose) Operator

## Problem

Face detection model (`face_det.mnn`) uses `ConvTranspose` (MNN `OpType_Deconvolution`, type 17) in FPN upsampling layers. QNN backend didn't have this op registered, causing:

1. `"Not registered type 17"` — op not found in creator map
2. `Assertion 'res' failed` at `QNNBackend.cpp:1899` — fallback geometry decomposition created intermediate tensors that QNN couldn't handle

Error output:
```
MNN_QNN: Not registered type 17, /model/fpn/upsample1/ConvTranspose_output_0.
MNN_QNN: Not registered type 17, /model/fpn/upsample2/ConvTranspose_output_0.
Tensor usage is 0.
MNN2QNNModel: QNNBackend.cpp:1899: int MNN::QNN::QnnBackend::getTensorIdx(const MNN::Tensor*) const: Assertion `res' failed.
```

## Files Changed

| File | Action | Purpose |
|------|--------|---------|
| [QNNDeconvolution.hpp](source/backend/qnn/execution/QNNDeconvolution.hpp) | **New** | Header for `QNNDeconvolution` class, includes weight conversion template |
| [QNNDeconvolution.cpp](source/backend/qnn/execution/QNNDeconvolution.cpp) | **New** | Maps MNN `OpType_Deconvolution` → QNN `TransposeConv2d` |
| [QNNUtils.hpp](source/backend/qnn/backend/QNNUtils.hpp#L117) | Modified | Added `extern void ___QNNDeconvolutionCreator__OpType_Deconvolution__();` |
| [QNNUtils.cpp](source/backend/qnn/backend/QNNUtils.cpp#L159) | Modified | Added `___QNNDeconvolutionCreator__OpType_Deconvolution__();` in `registerQNNOps()` |
| [QNNFlatten.cpp](source/backend/qnn/execution/QNNFlatten.cpp#L115-L129) | **Fixed** | Pre-existing bug: missing `if (permuteOutput)` guard around output transpose block |

## Key Technical Details

### QNN TransposeConv2d Parameters

From QNN SDK 2.45.0 (`include/QNN/QnnOpDef.h:737-741`):
```
QNN_OP_TRANSPOSE_CONV_2D                      "TransposeConv2d"
QNN_OP_TRANSPOSE_CONV_2D_PARAM_STRIDE         "stride"
QNN_OP_TRANSPOSE_CONV_2D_PARAM_PAD_AMOUNT     "pad_amount"
QNN_OP_TRANSPOSE_CONV_2D_PARAM_GROUP          "group"
QNN_OP_TRANSPOSE_CONV_2D_PARAM_OUTPUT_PADDING "output_padding"
```

**NOT supported**: `dilation` (unlike `Conv2d` which has `QNN_OP_CONV_2D_PARAM_DILATION`).

If `dilation != 1`, the implementation returns `NOT_SUPPORT`.

### Weight Format: IOHW vs OIHW

This was the most subtle issue. MNN stores Deconvolution weight in a **different format** than regular Convolution:

- **Conv weight**: MNN `[OC, IC, KH, KW]` (OIHW, from ONNX Conv)
- **Deconv weight**: MNN `[IC, OC, KH, KW]` (IOHW, from ONNX ConvTranspose)

Confirmed by `tools/converter/source/optimizer/onnxextra/OnnxConvolutionMerge.cpp:169-182`:
```cpp
bool isDeconv = originalOpType == "ConvTranspose";
int co = weightShape[0];  // ONNX ConvTranspose weight dims: [IC, OC, KH, KW]
int ci = weightShape[1];
if (isDeconv) {
    co = weightShape[1];  // OC — swapped for dimension calcs
    ci = weightShape[0];  // IC — but actual data stays in IOHW order
}
// Weight data is memcpy'd as-is, NOT reordered
```

QNN IR uses **HWIO** format `[KH, KW, IC, OC]` for both Conv2d and TransposeConv2d (confirmed by `lib/python/qti/aisw/converters/common/converter_ir/axis_tracker.py:635,640`):
```python
self.conv2d_weights_format = AxisTracker.AxisFormat.HWIO     # [KH, KW, IC, OC]
self.deconv2d_weights_format = AxisTracker.AxisFormat.HWIO   # same!
```

**Weight conversion formula for Deconv**:
```cpp
// MNN Deconv: [IC, OC, KH, KW] → QNN TransposeConv2d: [KH, KW, IC, OC]
srcOffset = w + kernelW * (h + kernelH * (o + oc * i));  // MNN [IC][OC][KH][KW]
dstOffset = o + oc * (i + ic * (w + kernelW * h));       // QNN [KH][KW][IC][OC]
```

Compare with Conv2d (different!):
```cpp
// MNN Conv: [OC, IC, KH, KW] → QNN Conv2d: [KH, KW, IC, OC]
srcOffset = w + kernelW * (h + kernelH * (i + ic * o));  // MNN [OC][IC][KH][KW]
dstOffset = o + oc * (i + ic * (w + kernelW * h));       // QNN [KH][KW][IC][OC]
```

### Implementation Architecture

`QNNDeconvolution` inherits from `QNNCommonExecution` and follows the same pattern as `QNNConvolution`:

1. `onEncode()`:
   - Reads `Convolution2D` parameters (kernel, stride, pad, dilation, group, outPads)
   - Returns `NOT_SUPPORT` if `dilation != 1`
   - Creates param tensors: `stride`, `pad_amount`, `output_padding`, `group`
   - Calls `createWeightAndBias()` to create reordered weight + bias
   - Optionally adds Relu/Relu6 fusion (two-stage: TransposeConv2d → Relu/Relu6)
   - Calls `addNodeToGraph()` with node type `"TransposeConv2d"`, package `"qti.aisw"`

2. `createWeightAndBias()`:
   - Uses `ConvolutionCommon::getConvParameters()` to extract float weight data
   - Reorders weight via `convertWeight()` template
   - Creates static float tensor for weight in QNN HWIO format
   - Creates bias as static float tensor

3. Creator + Registration:
```cpp
class QNNDeconvolutionCreator : public QnnBackend::Creator {
    virtual QNNCommonExecution* onCreate(...) const override {
        if (inputs.size() > 1) return nullptr;  // only single input supported
        return new QNNDeconvolution(backend, op);
    }
};
REGISTER_QNN_OP_CREATOR(QNNDeconvolutionCreator, OpType_Deconvolution)
```

## QNNFlatten Pre-existing Bug

### Symptom
After Deconvolution registration, segfault at:
```
#3  QNNFlatten::ReshapeTranspose at QNNFlatten.cpp:125
#4  QNNFlatten::onEncode at QNNFlatten.cpp:23
```

### Root Cause
[QNNFlatten.cpp:115-129](source/backend/qnn/execution/QNNFlatten.cpp#L115-L129) — the output transpose block (nchw → nhwc) was NOT wrapped in `if (permuteOutput)`.

When `outputDim <= 2` (e.g., flattening from 4D → 2D after ConvTranspose), `permuteOutput` stays `false` and `outputTempIndex` is **uninitialized**. Line 125 accesses `mTempTensorWrappers[outputTempIndex]` with garbage index → segfault.

### Why It Wasn't Triggered Before
Without Deconvolution registered, MNN's geometry system **decomposed** ConvTranspose into primitive ops. The decomposed tensors had different dimension formats that didn't enter the `ReshapeTranspose` code path. Registering Deconvolution preserved the NC4HW4 format, causing downstream Reshape/Flatten ops to take the buggy path.

### Fix
Added `if (permuteOutput)` guard:
```cpp
// nchw -> nhwc
if (permuteOutput) {  // <-- this guard was missing
    mNodeType = "Transpose";
    ...
    mInputs.push_back(*(mTempTensorWrappers[outputTempIndex]->getNativeTensor()));
    ...
}
```

## Debugging Timeline

1. **Initial error**: `"Not registered type 17"` + `Assertion 'res' failed`
   - Cause: Deconvolution op not registered in QNN backend
   - Fix: Created QNNDeconvolution class and registered it

2. **First segfault**: After initial registration, immediate crash
   - Cause: Included `dilation` parameter which QNN TransposeConv2d doesn't support
   - Fix: Removed dilation param, added `NOT_SUPPORT` check for dilation != 1

3. **Second segfault**: After dilation fix, still crashed
   - Debug: Added MNN_PRINT logs showing both Deconvolution onEncode calls succeeded
   - GDB backtrace: Crash in `QNNFlatten::ReshapeTranspose`, NOT in Deconvolution code
   - Root cause: Pre-existing bug in QNNFlatten exposed by new tensor format flow
   - Fix: Added `if (permuteOutput)` guard

## Known Limitations

- QNN TransposeConv2d does NOT support `dilation` (returns `NOT_SUPPORT` if dilation != 1)
- Only single-input Deconvolution supported (`inputs.size() <= 1`, no `hasOutputShape` support)
- Float weights only (no int8 weight quantization support yet, though can be added later following QNNConvolution's pattern)

## Reference

- QNN SDK op definitions: `QNN_SDK_ROOT/include/QNN/QnnOpDef.h`
- QNN Python translation reference: `QNN_SDK_ROOT/lib/python/qti/aisw/converters/qnn_backend/qnn_translations.py` (class `QnnTransposeConv2dTranslation`)
- Axis formats: `QNN_SDK_ROOT/lib/python/qti/aisw/converters/common/converter_ir/axis_tracker.py` (HWIO = [KH,KW,IC,OC])
- MNN ONNX conversion: `tools/converter/source/optimizer/onnxextra/OnnxConvolutionMerge.cpp` (IOHW → Convolution2D)
