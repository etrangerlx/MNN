---
name: qnn-concat-axis-nc4hw4-to-nhwc
description: "QNN Concat axis must be converted from MNN's NC4HW4 convention to QNN's NHWC convention"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 5ee5a1ca-49c0-4e66-bf17-9dcc784be608
---

MNN internally stores tensors in NC4HW4 format, and the Concat op's axis follows this convention. But QNN uses NHWC format. The axis must be converted before passing to QNN.

NC4HW4 4D → NHWC 4D mapping: `{0→0, 1→3, 2→1, 3→2}`

This is a pre-existing bug in [`source/backend/qnn/execution/QNNConcat.cpp`](../blob/master/source/backend/qnn/execution/QNNConcat.cpp) — it passes the axis directly without conversion. The bug is often hidden because other conversion errors (like missing op registrations) cause earlier failures.

**Why:** MNN's shape inference computes output dimensions using NC4HW4 axis (axis=1 = channels). QNN's graph validation computes output using NHWC axis (axis=1 = height). The mismatch causes "Expected X but got Y" validation errors.

**How to apply:** When using any QNN op that takes an axis parameter, check if the axis needs conversion from NC4HW4 to NHWC. The conversion depends on the tensor's `dimensionFormat` and `dimensions()`. Use `TensorUtils::getDescribe(tensor)->dimensionFormat` to check.
