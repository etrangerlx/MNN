# MTP (Multi-Token Prediction) 推测解码

MNN 中的 MTP（Multi-Token Prediction，多 token 预测）推测解码是一种通过轻量级 MTP 头并行生成多个候选 token，再由主模型一次性验证的加速机制。相比自回归逐个解码，MTP 在草稿命中率较高时能显著减少前向推理次数。

---

## 目录

- [整体架构](#整体架构)
- [初始化阶段](#初始化阶段)
- [首轮草稿生成（Prefill）](#首轮草稿生成prefill)
- [主循环：验证 + 草稿再生](#主循环验证--草稿再生)
  - [构造 drafts 数组](#构造-drafts-数组)
  - [主模型并行验证](#主模型并行验证)
  - [draftVerify：逐位校验](#draftverify逐位校验)
  - [下一轮 MTP 草稿生成](#下一轮-mtp-草稿生成)
  - [清理与状态更新](#清理与状态更新)
- [核心数据结构](#核心数据结构)
- [关键设计要点](#关键设计要点)

---

## 整体架构

```
┌─────────────────┐         ┌──────────────────┐
│  Prefill 主模型  │ ──────► │ 生成首轮 mtp_draft│
└─────────────────┘         └──────────────────┘
                                     │
                                     ▼
                    ┌────────────────────────────┐
                    │   While (未达 max_token)    │
                    └────────────────────────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         ▼                           ▼                           ▼
┌─────────────────┐    ┌─────────────────────┐    ┌──────────────────────┐
│ drafts = [cur]  │    │ 主模型 forwardVec     │    │ draftVerify 逐位校验  │
│ + mtp_draft[]   │───►│ (并行验证所有草稿)     │───►│ 返回接受数 i_dft      │
└─────────────────┘    └─────────────────────┘    └──────────────────────┘
                                                           │
                                ┌──────────────────────────┘
                                ▼
                    ┌─────────────────────────┐
                    │ MTP mtpForward 生成下一轮 │
                    │ mtp_draft[]              │
                    │ (基于验证的 hidden_states)│
                    └─────────────────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │ 清理未接受 KV & 更新 context│
                    └─────────────────────────┘
                                │
                                (loop back)
```

MTP 核心文件：

- [`transformers/llm/engine/src/speculative_decoding/mtp.cpp`](../../transformers/llm/engine/src/speculative_decoding/mtp.cpp)
- [`transformers/llm/engine/src/speculative_decoding/generate.cpp`](../../transformers/llm/engine/src/speculative_decoding/generate.cpp)
- [`transformers/llm/engine/src/speculative_decoding/generate.hpp`](../../transformers/llm/engine/src/speculative_decoding/generate.hpp)

主模型相关：

- [`transformers/llm/engine/include/llm/llm.hpp`](../../transformers/llm/engine/include/llm/llm.hpp)

---

## 初始化阶段

**入口**：[`MtpGeneration::load()`](../../transformers/llm/engine/src/speculative_decoding/mtp.cpp#L18-L38)

MTP 模型相对于主模型独立加载，拥有独立的 KV Cache 管理机制：

| 步骤 | 行为 | 代码位置 |
|------|------|----------|
| 创建 KVMeta | 分配独立的 KV Cache 元数据 `mMtpMeta` | `mtp.cpp:19` |
| 加载 MTP 模型 | 从 `config->mtp_model()` 路径加载一次 `mtp.mnn` | `mtp.cpp:23-28` |
| 构建模块池 | 按 `<seq_len, isAllLogits>` 建立克隆模块池，避免推理时反复创建 | `mtp.cpp:30-36` |
| 获取 hidden_states 索引 | 记录主模型 `hidden_states` 在输出列表中的位置 | `mtp.cpp:37` |

**MTP 拓扑定义**（输入 -> 输出）：

```
Inputs:  [input_embeds, hidden_states, attention_mask, position_ids, logits_index]
Outputs: [logits]
```

模块池的 key 设计：
- 推理阶段（decode）：`seq_len = 1..mDraftLength+1`，`isAllLogits = true`
- Prefill 阶段：`seq_len = mPrefillKey`（默认 100），`isAllLogits = false`

---

## 首轮草稿生成（Prefill）

**入口**：[`MtpGeneration::generate()`](../../transformers/llm/engine/src/speculative_decoding/mtp.cpp#L121-L147)

在首次进入主循环前，先构造一组合适的 embedding 喂给 MTP，生成第一批草稿 token。

**输入构造逻辑**：

```cpp
auto cur_embed = mLlm->embedding({mContext->current_token});
auto pre_embeds = _Split(input_embeds, {1, input_embeds->getInfo()->dim[0]-1}, 0);
auto prefill_embeds = _Concat({pre_embeds[1], cur_embed}, 0);
```

- 把 `input_embeds`（prompt 的 embedding）**第一个 token 去掉**
- **末尾追加 `current_token`**（prefill 后采样出的第一个 token）
- 这样 MTP 看到的是与主模型 prefill 对齐的序列

**采样首轮草稿**：

```cpp
auto mtpDraft = mtpForward(prefill_embeds, prev_hidden_states);
for (int i = 0; i < mLlm->mDraftLength; i++) {
    auto sample_offset = i * sample_size;
    mtp_draft[i] = mLlm->sample(mtpDraft[0], sample_offset, sample_size);
}
```

- `prev_hidden_states` 来自主模型 Prefill 阶段的输出
- 从 MTP 输出的 logits 中按位置偏移依次采样 `mDraftLength` 个 token（默认 4 个）
- 结果存入 `mtp_draft` 数组，缓存供下一轮验证使用

---

## 主循环：验证 + 草稿再生

**入口**：[`MtpGeneration::generate()`](../../transformers/llm/engine/src/speculative_decoding/mtp.cpp#L158-L252)

每轮迭代的核心是：把上一轮 MTP 生成的草稿交给主模型并行验证，验证通过后，立即生成下一轮草稿。

---

### 构造 drafts 数组

**代码位置**：[`mtp.cpp:167-180`](../../transformers/llm/engine/src/speculative_decoding/mtp.cpp#L167-L180)

```cpp
std::vector<int> drafts;
drafts.push_back(mContext->current_token);  // 当前已确认的 token
drafts.insert(drafts.end(), mtp_draft.begin(), mtp_draft.end());  // 草稿 token
```

`drafts` 的结构（假设 `mDraftLength = 4`）：

| 索引 | 含义 |
|------|------|
| 0 | `current_token`（上一轮已确认/刚采样的 token） |
| 1 | `mtp_draft[0]`（第一个草稿 token） |
| 2 | `mtp_draft[1]`（第二个草稿 token） |
| 3 | `mtp_draft[2]`（第三个草稿 token） |
| 4 | `mtp_draft[3]`（第四个草稿 token） |

---

### 主模型并行验证

**入口**：[`forwardVec(drafts)`](../../transformers/llm/engine/src/llm.cpp#L667-L672)

```cpp
mLlm->mMeta->add = drafts.size();
auto outputs = mLlm->forwardVec(drafts);
auto logits = outputs[0];
```

这是推测解码的核心优化：**主模型不是逐个 token 自回归推理，而是一次前向并行验证所有草稿**，与逐个解码在数学上完全等价（通过 causal mask 保证），但前向次数从 N 次降为 1 次。

#### 4.1 入口到 embedding

[`llm.cpp:667-688`](../../transformers/llm/engine/src/llm.cpp#L667-L688)

```cpp
std::vector<VARP> Llm::forwardVec(const std::vector<int>& input_ids) {
    auto input_embeds = embedding(input_ids);   // (1, seq_len, hidden_size)
    auto outputs = forwardVec(input_embeds);
    ...
}
```

假设 `drafts = [cur_tok, d_1, d_2, d_3, d_4]`（`mDraftLength + 1 = 5` 个 token），则 `input_embeds` 的 shape 为 `(1, 5, hidden_size)`。

#### 4.2 KV Cache 通知：`mMeta->add`

[`llm.cpp:685`](../../transformers/llm/engine/src/llm.cpp#L685)

```cpp
mMeta->add = seq_len;  // 5
```

`KVMeta`（定义见 [`source/core/KVMeta.hpp`](../../source/core/KVMeta.hpp)）是主模型与 backend KV Cache 之间的通信接口：

| 字段 | 含义 |
|------|------|
| `add` | 本次推理新增 token 数，backend 据此在 KV Cache 中预留/追加 K/V 槽位 |
| `previous` | 历史已缓存 token 的总长度（经过前面多轮累积） |
| `remove` | 草稿验证不通过后，需回滚删除的 KV 槽位数 |
| `sync()` | 每轮结束后，`previous = previous - remove + add + revertNumber`，并清零临时字段 |

当主模型执行 `forwardVec` 时，backend 通过 `KVCACHE_INFO` hint 读取 `mMeta`：
- 新 K/V：`seq_len=5` 个位置的 key/value 写入 KV Cache
- Attention 计算：query 与历史缓存 + 新写入的 K/V 做内积
- 验证失败时：`remove = drafts.size() - i_dft` 抹掉那部分 K/V

#### 4.3 causal attention mask 构造

[`llm.cpp:1470-1533`](../../transformers/llm/engine/src/llm.cpp#L1470-L1533)

```cpp
int kv_seq_len = mContext->all_seq_len + seq_len;  // N + 5
attentionMask = _Input({1, 1, seq_len, kv_seq_len}, ...);
for (int i = 0; i < seq_len; i++) {
    for (int j = 0; j < kv_seq_len; j++) {
        ptr[kv_seq_len * i + j] = (j > i + mContext->all_seq_len)
            ? std::numeric_limits<float>::lowest()
            : 0.0f;
    }
}
```

mask 的 shape 为 `(batch=1, heads=1, query_len=5, kv_len=N+5)`，其含义是**下三角因果约束**：

```
         历史 token (N 个)              新输入 token (5 个)

      | ← 全部可见 ← | |  下三角可见  | ← 上三角 mask → |
q0:   [0, 0, ...,  0,   0, -inf, -inf, -inf, -inf]  cur_tok 只看自己和历史
q1:   [0, 0, ...,  0,   0,  0  , -inf, -inf, -inf]  d_1 能看 cur_tok
q2:   [0, 0, ...,  0,   0,  0  ,  0  , -inf, -inf]  d_2 能看 cur_tok + d_1
q3:   [0, 0, ...,  0,   0,  0  ,  0  ,  0  , -inf]  d_3 能看前 3 个
q4:   [0, 0, ...,  0,   0,  0  ,  0  ,  0  ,  0  ]  d_4 能看全部
```

**为什么这个 mask 能保证正确性**：
- 位置 `i`（对应 `d_i`）只能看到前 `i` 个新输入 + 所有历史 token
- 这意味着 `d_i` 的预测**不会偷看** `d_{i+1}`、`d_{i+2}` 等后续草稿
- 多个位置的计算结果分别等同于逐个自回归推理的结果，只是 batch 在一起并行完成

#### 4.4 `logitsIndex` 与模块池选择

[`llm.cpp:534-562`](../../transformers/llm/engine/src/llm.cpp#L534-L562)

```cpp
bool inDecode = mContext->gen_seq_len > 0;
bool isAllLogists = mConfig->all_logits() ? true : (inDecode ? mInSpec : false);
int seqLenKey = inDecode ? hiddenState->getInfo()->dim[mSeqLenIndex] : mPrefillKey;
isAllLogists = seqLenKey == 1 ? false : isAllLogists;

auto moduleKey = std::make_pair(seqLenKey, isAllLogists);
selectModule = mModulePool[moduleKey];
```

主模型在加载阶段预克隆了三种用途的模块：

| 模块 key | 用途 |
|----------|------|
| `(1, false)` | 自回归单 token 解码（验证失败后逐个补 token） |
| `(mDraftLength+1, true)` | 草稿并行验证（验证 `drafts.size()` 个 token） |
| `(100, all_logits)` | Prefill 阶段 |

`mInSpec` 表示当前处于推测解码模式，`isAllLogists` 在前向时会决定输出哪些位置的 logits：

```cpp
if (isAllLogists) {
    logitsIndex = logitsAllIdx;   // {0}  -> 返回所有位置的 logits
} else {
    logitsIndex = logitsLastIdx;  // {-1} -> 只返回最后一个位置的 logits
}
if (mMeta->add != seqLen) {
    logitsIndex = logitsAllIdx;   // 如果带 pad，必须返回全部
}
```

#### 4.5 模型内部：Attention + KV Cache 行为

单次前向推理中，各层 Transformer 的执行逻辑（backend 层实现，以 KV Cache 后端为例）：

```
1. 取出历史 KV cache -> K_hist (N, d_k), V_hist (N, d_v)
2. 用当前 input_embeds 计算 Q_cur (5, d_k), K_cur (5, d_k), V_cur (5, d_v)
3. 将 K_cur、V_cur 追加到 KV cache -> K_all (N+5, d_k), V_all (N+5, d_v)
4. 执行 Attention: softmax( Q_cur · K_all^T / √d + causal_mask ) · V_all

   - Q_cur[0]（cur_tok）只能看到历史 N 个位置
     -> 输出等价于 "给 cur_tok 做 next-token prediction"
   - Q_cur[1]（d_1）能看到 N+1 个位置（历史 + cur_tok）
     -> 输出等价于 "cur_tok 被接受后，验证 d_1 是否正确"
   - Q_cur[2]（d_2）能看到 N+2 个位置（历史 + cur_tok + d_1）
     -> 输出等价于 "d_1 被接受后，验证 d_2 是否正确"
   ...
5. 逐层前向传递，最终到达 lm_head，输出 logits
```

多个位置的计算结果在 batch 维度上并行生成，kernel 只启动一次（或极少量次），大幅减少了 GPU kernel launch 开销。

#### 4.6 logits 输出结构与 offset 对应

验证通过后的 `logits` shape 近似为 `(batch=1, seq_len=5, vocab_size)`。

在 [`generate.cpp:118-159`](../../transformers/llm/engine/src/speculative_decoding/generate.cpp#L118-L159) 中的 `draftVerify` 逐个采样时，offset 计算如下：

```cpp
auto sample_size = logits->getInfo()->dim[...];  // vocab_size
auto sample_offset = logits->getInfo()->size - (drafts.size() - i_dft + 1) * sample_size;
auto predict = mLlm->sample(logits, sample_offset, sample_size);
```

| `i_dft` | 验证对象 | `sample_offset` | 对应 logits 位置 | 语义 |
|---------|---------|----------------|-----------------|------|
| 1 | `d_1` | `size - 5 * vocab` | logits[0] | cur_tok 的 next-token prediction |
| 2 | `d_2` | `size - 4 * vocab` | logits[1] | cur_tok + d_1 的 next-token prediction |
| 3 | `d_3` | `size - 3 * vocab` | logits[2] | cur_tok + d_1 + d_2 的 next-token prediction |
| 4 | `d_4` | `size - 2 * vocab` | logits[3] | ... |
| 全对时 | 额外采样 | `size - 1 * vocab` | logits[4] | 所有草稿都接受后，再采一个 |

`predict`（你选中的第 127 行）是从指定 offset 的 logits 分布中采样得到的 token ID，然后与对应位置的草稿 token 对比：
- `predict == drafts[i_dft]`：接受该草稿
- `predict != drafts[i_dft]`：拒绝并替换为 `predict`

换算公式 `sample_offset = size - (drafts.size() - i_dft + 1) * sample_size` 表明 logit 在内存中按 `(batch, seq_len, vocab)` 排列，offset 从末尾向前数，每步后退 `sample_size`（即 `vocab_size`）。

---

### draftVerify：逐位校验

**实现位置**：[`generate.cpp:118-159`](../../transformers/llm/engine/src/speculative_decoding/generate.cpp#L118-L159)

```cpp
int i_dft = 1;
for (; i_dft < drafts.size(); i_dft++) {
    // 从 logits 对应位置采样主模型预测
    auto predict = mLlm->sample(logits, sample_offset, sample_size);

    if (mLlm->is_stop(predict)) {
        mContext->current_token = predict;
        stop = true;
        break;
    }

    if (predict != drafts[i_dft]) {
        mContext->current_token = predict;  // 用主模型结果替换草稿
        break;
    }
    // 匹配成功，继续验证下一个
}

if (i_dft == drafts.size()) {
    // 所有草稿都对，再从最后一个位置额外采样一个
    auto predict = mLlm->sample(logits, last_offset, sample_size);
    mContext->current_token = predict;
}
```

**校验规则**：

从左到右（索引 1 开始，跳过 `current_token`），对比主模型采样结果与草稿 token：

| 情况 | 行为 | 接受数 `i_dft` |
|------|------|----------------|
| 全部匹配 | 接受所有草稿，额外从最后一个位置采样新 token | `drafts.size()` |
| 第 k 个不匹配 | 接受前 `k-1` 个草稿，用主模型结果替换第 k 个 | `k` |
| 遇到 stop token | 停止生成 | 当前位置 |

> **为什么跳过索引 0**：`drafts[0]` 是已经确认的 `current_token`，不需要验证。

---

### 下一轮 MTP 草稿生成

**代码位置**：[`mtp.cpp:197-215`](../../transformers/llm/engine/src/speculative_decoding/mtp.cpp#L197-L215)

```cpp
std::vector<int> currentIds;
for (int i = 1; i < i_dft; i++) {
    currentIds.push_back(drafts[i]);  // 本轮被接受的草稿 token
}
currentIds.push_back(mContext->current_token);  // 本轮新确认的 token

auto prev_hidden_states = outputs[1];  // 主模型验证时输出的 hidden_states
auto mtpDraft = mtpForward(currentIds, prev_hidden_states);

mMtpMeta->remove = drafts.size() - i_dft;  // 清理未通过草稿的 KV Cache

for (int i = 0; i < mLlm->mDraftLength; i++) {
    auto offset = (i * dim + i_dft - 1) * sample_size;
    mtp_draft[i] = mLlm->sample(mtpDraft[0], offset, sample_size);
}
```

**输入构造**：

- `currentIds` = 被接受的草稿 token（不含原 `current_token`），长度 = `i_dft - 1`
- 末尾追加本轮验证后新确认的 `current_token`
- `prev_hidden_states` 来自主模型 `forwardVec()` 的第二个输出（`outputs[1]`）

**KV Cache 清理**：

`mMtpMeta->remove = drafts.size() - i_dft`：清理 MTP KV Cache 中未被接受的那部分草稿 token，保证缓存一致性。

**偏移计算说明**：

```cpp
auto offset = (i * mtpDraft[0]->getInfo()->dim[1] + i_dft - 1) * sample_size;
```

- `dim[1]` 是序列长度维度
- `i_dft - 1` 是被接受 token 数量（也是 MTP 输入序列中最后一个位置的索引）
- 对每个未来位置 `i`，采样对应偏移的 logits

---

### 清理与状态更新

**代码位置**：[`mtp.cpp:217-251`](../../transformers/llm/engine/src/speculative_decoding/mtp.cpp#L217-L251)

```cpp
mLlm->mMeta->remove = drafts.size() - i_dft;   // 清理主模型未通过的 KV
len += i_dft;                                     // 累加有效 token 数
mLlm->updateContext(i_dft, i_dft - 1);           // 更新上下文长度

// 追加到历史
mContext->history_tokens.insert(..., drafts.begin(), drafts.begin() + i_dft);
mContext->output_tokens.insert(..., drafts.begin(), drafts.begin() + i_dft);
```

| 动作 | 说明 |
|------|------|
| `mMeta->remove` | 清理主模型 KV Cache 中未验证通过的草稿 |
| `len += i_dft` | 累加本轮实际接受的有效 token 数 |
| `updateContext` | 更新主模型上下文状态（序列长度、生成长度） |
| 追加历史 | 把接受的 token 加入对话历史，用于下一轮 |

**停止条件**：

- 用户取消 / 内部错误
- 超时
- 达到 `max_new_tokens`
- `is_stop()` 返回 true

---

## 核心数据结构

### `MtpGeneration` 类成员

| 成员 | 类型 | 说明 |
|------|------|------|
| `mMtpModules` | `vector<shared_ptr<Module>>` | MTP 模型模块（通常只有 1 个 base） |
| `mMtpModulePool` | `map<pair<int, bool>, shared_ptr<Module>>` | 按 `<seq_len, isAllLogits>` 键值复用/克隆的模块池 |
| `mMtpMeta` | `shared_ptr<KVMeta>` | MTP 独立的 KV Cache 元数据 |
| `mHiddenStateIndex` | `int` | 主模型 `hidden_states` 输出的索引 |

### 模块池 key 含义

```cpp
auto moduleKey = std::make_pair(seqLenKey, isAllLogists);
```

| `seqLenKey` | `isAllLogists` | 场景 |
|-------------|----------------|------|
| `1..mDraftLength+1` | `true` | 解码阶段，取所有位置的 logits |
| `mPrefillKey`（100） | `false` | Prefill 阶段，只取最后一个位置的 logits |

---

## 关键设计要点

| 设计 | 说明 |
|------|------|
| **模块隔离** | MTP 拥有独立的 `ModulePool` 和 `KVMeta`，与主模型不共享 KV Cache，避免相互污染 |
| **一次并行验证** | 主模型 `forwardVec(drafts)` 同时处理所有草稿 token，是性能加速的核心 |
| **逐位拒绝采样** | `draftVerify` 从左到右按顺序验证，遇到第一个错误立即停止，这是推测解码的标准做法 |
| **隐藏状态复用** | MTP 始终复用主模型输出的 `hidden_states` 作为条件输入，而非自己维护深层状态表示 |
| **KV 清理机制** | 每轮结束后 `remove` 掉未通过的草稿 token，保证主模型和 MTP 的缓存一致性 |
| **模块克隆池化** | 按序列长度和 logits 模式预克隆模块，避免推理过程中的动态创建开销 |
| **输入反偏移对齐** | Prefill 首轮草稿生成时，`input_embeds` 去掉首 token、末尾补 `current_token`，与主模型上下文严格对齐 |