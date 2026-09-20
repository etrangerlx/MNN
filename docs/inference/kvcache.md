# KV Cache 工作机制

MNN 的 LLM 推理采用「元数据驱动 + 后端专属管理器」的 KV Cache 架构，由三层协同工作：

```
LLM 引擎层 (llm.cpp)          ←→     Runtime (Interpreter)
    │                                    │
    │ setHintPtr(KVCACHE_INFO, mMeta)    │  在每个 Attention 算子中
    ▼                                    ▼
 KVMeta 结构体  ───────────────────→  CPUAttention
                                         │
                                         ▼
                                CPUKVCacheManager
```

---

## 一、核心结构 KVMeta

`KVMeta`（`source/core/KVMeta.hpp`）是引擎层与底层算子之间的通信桥梁。引擎层通过设置 `KVMeta` 的字段来告诉 Attention 算子每次 forward 时需要增/删多少 KV。

核心字段：

| 字段 | 含义 |
|------|------|
| `previous` | 当前 KVCache 已有的 token 长度 |
| `add` | 本次 forward 要新增的 token 数 |
| `remove` | 本次要丢弃的 token 数（从尾部丢弃） |
| `reserve` | `[begin, size]` 数组，保留部分回退的 KV（用于 `eraseHistory`） |
| `file_flag` | `NoChange` / `PendingWrite` / `PendingRead`，用于 Prefix Cache 磁盘读写 |

每次 `forward()` 后调用 `sync()` 更新状态：
```cpp
previous = previous - remove + add + revertNumber;
```

---

## 二、KVCache 的生命周期

在 `CPUAttention::onExecute`（`source/backend/cpu/CPUAttention.cpp`）中，每次推理都会检查并驱动 KV 管理器：

```cpp
if (!mIsKVShared) {
    if (mMeta->previous == mMeta->remove) {
        mKVCacheManager->onClear();          // 清空旧缓存
        mKVCacheManager->onAlloc(mMeta, seq_len); // 重新分配
    } else {
        mKVCacheManager->onRealloc(mMeta);   // 检查容量并扩容
    }
    // 将本次 key/value 写入 KVCache
    mKVCacheManager->onUpdateKV(key, value, (int)insertLen);
}
```

### 1. onAlloc — 首次分配 / 重置

根据 `seq_len + mExpandChunk`（默认扩容 64）计算容量 `mMaxLength`。支持两种存储模式：

- **内存模式**：使用 `Tensor::createDevice` + `Backend::onAcquireBuffer` 直接申请显存/内存 buffer。
- **磁盘模式**（`kvcache_mmap=true`）：创建 `.k` / `.v` 文件并用 `mmap` 映射到地址空间。当物理内存不足时，OS 自动将不活跃的页换出到磁盘，上层无需关心换页细节。同时支持 Prefix Cache 时把文件保存到 `prefix_cache_path` 目录用于后续复用。

### 2. onRealloc — 动态扩容

如果 `previous + add - remove > mMaxLength`，触发扩容：

- **内存模式**：`expandKVCacheInMem()` 申请更大的 buffer，`memcpy` 旧 KV 后释放旧空间。
- **磁盘模式**：`expandKVCacheInDisk()` 先把旧文件 mmap 读入临时内存，resize 文件后重新 mmap，再将数据拷贝回去。

同时处理 `remove`（丢弃尾部 KV）和 `reserve`（选择性回退保留）。

### 3. onUpdateKV — 写入新 KV

`ProcessKey` / `ProcessValue` 将 forward 出的新 key/value 按照 SIMD packing 格式写入到已有 KV 的尾部（`mPastLength` 偏移处），然后 `mPastLength += seq_len`。

---

## 三、数据布局（Packing 格式）

为了配合 CPU SIMD/GEMM kernel，KV Cache 不是朴素的 `[seq_len, head_dim]`，而是做了 packing 重排：

### Key 布局

`[maxlen/hP, headdim/lP, hP, lP]`

- `hP = hP8` / `lP = lP8`（int8 模式）或普通 float packing 大小
- `seq` 在 `hP` 维度上打包，`head_dim` 在 `lP` 维度上打包

### Value 布局（Flash Attention 模式）

`[UP_DIV(maxlen, blockKV), UP_DIV(headdim, hP), UP_DIV(blockKV, lP), hP, lP]`

- 按 `MNN_FLASH_ATTENTION_BLOCK_SIZE`（默认 512）分块存储，便于分块 Flash Attention 计算。

---

## 四、KV Cache 量化

`attention_mode` 配置控制是否对 K/V 做量化（`source/backend/cpu/CPUAttention.cpp`）：

| attention_mode | K 量化 | V 量化 |
|---------------|-------|-------|
| 0 | None | None |
| 1 / 9 | Int8 | None |
| 2 / 10 | Int8 | Int8 |
| 3 | TQ3 | None |
| 4 | TQ3 | TQ3 |
| 5 | TQ4 | None |
| 6 | TQ4 | TQ4 |

- **Int8**：per-block 非对称量化。`ProcessKey` / `ProcessValue` 调用 `MNNQuantAttentionKey` / `MNNQuantAttentionValue` kernel 完成打包 + 量化，同时存储 scale / bias。
- **TQ3 / TQ4**：TurboQuant，按 block（block_size=16）量化到 3bit/4bit，大幅降低显存占用。

> **注意**：磁盘模式（mmap）目前不支持量化 KV（见 `CPUKVCacheManager::onAlloc` 中的检查）。

---

## 五、Prefix Cache（Prompt Cache）

MNN 支持把对话前缀的 KV Cache 持久化到磁盘，后续对话中复用，跳过前缀部分的重新计算。

### 写入

第一次在对话中处理完整前缀后：
- `.k` / `.v` 数据保存到 `prefix_cache_path` 目录，文件名为 `{name}_{layer_index}.k` / `{name}_{layer_index}.v`。
- `_sync.k` / `_sync.v` 作为完成标记文件。

### 读取

后续再次对话时，`mMeta->file_flag = PendingRead`，`onAlloc` 中直接 `mmap` 已有的前缀 KV 文件，并通过 `seqlen_in_disk` 记录磁盘上已有长度，Prefill 只需计算增量部分即可。

### 回退（eraseHistory）

通过 `reserve` 数组指定要保留的回退段，底层调用 `moveKV()` 做紧凑化后，`mPastLength` 指向保留段末尾。

---

## 六、关键代码路径

| 操作 | 代码位置 |
|------|---------|
| KVMeta 定义 | `source/core/KVMeta.hpp` |
| KVCacheManager 抽象 | `source/core/KVCacheManager.hpp` `.cpp` |
| CPU 后端实现 | `source/backend/cpu/CPUKVCacheManager.hpp` `.cpp` |
| Attention 中调用驱动 | `source/backend/cpu/CPUAttention.cpp:392-408` |
| LLM 引擎层设置 meta | `transformers/llm/engine/src/llm.cpp:523-532` |
| 引擎层同步 meta 状态 | `transformers/llm/engine/src/llm.cpp:643` |

---

## 七、总结

MNN 的 KV Cache 设计兼顾了**性能、内存和灵活性**：

1. **元数据驱动**：`KVMeta` 解耦了引擎策略与底层存储实现。
2. **按需扩容**：内存和磁盘两种模式均支持动态扩容。
3. **量化压缩**：支持 Int8 / TQ3 / TQ4 多种量化方式减少显存占用。
4. **Prefix Cache**：支持对话前缀的磁盘持久化复用，加速多轮对话。
5. **Packing 优化**：配合 CPU SIMD kernel 的数据布局，保证 Attention 计算效率。
