---
name: qnn-op-registration-pattern
description: "QNN backend op registration requires explicit function calls in QNNUtils.cpp, not auto-registration"
metadata: 
  node_type: memory
  type: reference
  originSessionId: 5ee5a1ca-49c0-4e66-bf17-9dcc784be608
---

QNN backend ops use **explicit** registration, unlike CPU backend:

1. Each op `.cpp` file calls `REGISTER_QNN_OP_CREATOR(Creator, OpType)` which generates `void ___Creator__OpType__()`
2. The function must have an `extern` declaration in [`source/backend/qnn/backend/QNNUtils.hpp`](../blob/master/source/backend/qnn/backend/QNNUtils.hpp)
3. The function must be called explicitly in `registerQNNOps()` in [`source/backend/qnn/backend/QNNUtils.cpp`](../blob/master/source/backend/qnn/backend/QNNUtils.cpp)

Missing any of these three steps = "Not registered type N" error at runtime, even if the code compiles.

**Why:** The QNN runtime is dynamically loaded via dlopen; static initializers don't fire reliably. Explicit registration ensures ops are registered at the right time during QNN backend initialization.

**How to apply:** When adding a new QNN op, always complete all three steps: (1) add REGISTER_QNN_OP_CREATOR in the op's .cpp, (2) add extern declaration in QNNUtils.hpp, (3) add call in QNNUtils.cpp::registerQNNOps().
