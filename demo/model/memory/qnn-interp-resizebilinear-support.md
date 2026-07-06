---
name: qnn-interp-resizebilinear-support
description: Added QNN support for Interp op (ResizeBilinear/ResizeNearest/ResizeCubic) for DeepLabV3 model
metadata: 
  node_type: memory
  type: project
  originSessionId: 5ee5a1ca-49c0-4e66-bf17-9dcc784be608
---

Added QNN backend support for the `Interp` op (OpType 35), used by DeepLabV3's ResizeBilinear layers.

New files:
- [`source/backend/qnn/execution/QNNInterp.hpp`](../blob/master/source/backend/qnn/execution/QNNInterp.hpp)
- [`source/backend/qnn/execution/QNNInterp.cpp`](../blob/master/source/backend/qnn/execution/QNNInterp.cpp)

Modified files:
- [`source/backend/qnn/backend/QNNUtils.hpp`](../blob/master/source/backend/qnn/backend/QNNUtils.hpp) — extern declaration
- [`source/backend/qnn/backend/QNNUtils.cpp`](../blob/master/source/backend/qnn/backend/QNNUtils.cpp) — registration call
- [`source/backend/qnn/execution/QNNConcat.cpp`](../blob/master/source/backend/qnn/execution/QNNConcat.cpp) — NC4HW4→NHWC axis fix

Mapping: resizeType=2 (bilinear) → QNN_OP_RESIZE_BILINEAR; resizeType=1/4 (nearest) → QNN_OP_RESIZE NEAREST; resizeType=3 (cubic) → QNN_OP_RESIZE CUBIC.

Full documentation at [`skills/add-new-op/notes/qnn-interp-implementation.md`](../blob/master/skills/add-new-op/notes/qnn-interp-implementation.md).

Related: [[qnn-op-registration-pattern]], [[qnn-concat-axis-nc4hw4-to-nhwc]]
