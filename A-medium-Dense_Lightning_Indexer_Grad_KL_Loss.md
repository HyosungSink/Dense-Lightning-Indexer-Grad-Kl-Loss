# 【A组中等题】DenseLightningIndexerGradKLLoss 算子

## 1. 算子说明

DenseLightningIndexerGradKLLoss 是 LightningIndexer（稀疏 Top-k 索引选择算子）的**反向梯度算子**，并融合了 KL 散度损失计算。它通过 KL 散度约束 LightningIndexer 的 Top-k 选择分布逼近主注意力（main attention）的 softmax 分布，从而计算 loss 及其对 queryIndex/keyIndex/weights 的梯度，用于训练 LightningIndexer。稠密（dense）场景下 query/key/queryIndex/keyIndex 不做稀疏化处理。

### 背景术语（题目内自解释）

- **LightningIndexer**：为每个 query token 从上下文中选出相关度最高的 Top-k 个 key 位置索引，减少长序列 Attention 计算量。本算子是其训练阶段的反向。
- **主注意力（main attention）与 target 分布 `p`**：用 query/key 做标准注意力 `softmax(Q@K^T/√d)`，对所有头求和后沿上下文方向做 L1 正则化，得到 target 分布 `p`（长度为 key 序列长度）。
- **Indexer 得分 `I` 与预测分布**：用 queryIndex/keyIndex 计算 `I = W @ ReLU(q̃ @ K̃^T)`，再做 `softmax(I)` 得到 LightningIndexer 的预测分布。
- **KL 散度**：衡量两个分布差异，`D_KL(a||b) = Σ_i a_i log(a_i/b_i)`。loss 使预测分布 `softmax(I)` 逼近 target 分布 `p`。
- **GQA/MQA**：Grouped/Multi-Query Attention。Indexer 的 keyIndex 头数 `Nidx2=1`（MQA），所有 queryIndex 头共享一组 keyIndex，group size = Nidx1。
- **RoPE**：旋转位置编码，query/key 各带一份 RoPE 向量（维度 `Dr`）。
- **维度符号**：`B`=Batch Size，`S1`=query 序列长度，`S2`=key 序列长度，`N1`=query 头数，`N2`=key 头数（等于 N1），`Nidx1`=queryIndex 头数，`Nidx2`=keyIndex 头数（恒为 1），`D`=每个头维度（=128），`Dr`=RoPE 维度（=64），`G=N1/N2`。
- **数据排布**：本算子仅支持 `BSND` 排布，即 `(B, S, N, D)`。
- **变长序列**：`actual_seq_lengths` 给出每个 batch 有效 token 数（超出为 padding，不参与计算）。

### 计算公式

1. **Indexer 得分**（Top-k value）：

```
I[t,:] = W[t,:] @ ReLU( q̃[t,:] @ K̃[:,t,:]^T )
```

其中 `W` 为 weights，`q̃` 为 queryIndex，`K̃` 为 keyIndex。

2. **target 分布**（主注意力 softmax，各头求和后 L1 正则化）：

```
p[t,:] = L1norm( Σ_head softmax( q[t,:] @ K[:,t,:]^T / √d ) )
```

3. **KL Loss**：

```
Loss = Σ_t D_KL( p[t,:] || softmax(I[t,:]) ) = Σ_t Σ_i p_i log( p_i / softmax(I)_i )
```

4. **梯度**（由 KL 散度对 I 求导，再链式传播）：

```
dI[t,:]   = softmax(I[t,:]) - p[t,:]
dW[t,:]   = dI[t,:] @ (ReLU(S[t,:]))^T        # S = q̃ @ K̃^T
dq̃[t,:]  = dS[t,:] @ K̃[:,t,:]
dK̃[:,t,:] = (dS[t,:])^T @ q̃[:,t,:]
```

其中 `S = q̃ @ K̃^T` 为 indexer 相似度，`dS` 由 `dI` 经 `W` 与 ReLU 反向得到。

---

## 2. 输入输出说明

### 输入规格

| 输入 | 必选/可选 | 类型 | 形状 | 数据类型 | 含义 |
|---|---|---|---|---|---|
| query | 必选 | 张量 | (B, S1, N1, D) | float16/bfloat16 | 主注意力输入 Q。N1∈{32,64,128}，D=128 |
| key | 必选 | 张量 | (B, S2, N2, D) | float16/bfloat16 | 主注意力输入 K。N2=N1（标准 MHA，每 query 头对应一个独立 key 头），D=128 |
| queryIndex | 必选 | 张量 | (B, S1, Nidx1, D) | float16/bfloat16 | LightningIndexer 的 query。Nidx1∈{8,16,32,64}，D=128 |
| keyIndex | 必选 | 张量 | (B, S2, Nidx2, D) | float16/bfloat16 | LightningIndexer 的 key。Nidx2=1（MQA），D=128 |
| weights | 必选 | 张量 | (B, S1, Nidx1) | float16/bfloat16/float32 | group 内每个头的权重 W |
| softmaxMax | 必选 | 张量 | (B, N2, S1, G) | float32 | 主注意力前向 pass 的 softmax 行最大值，由外部传入（避免重算）。G=N1/N2 |
| softmaxSum | 必选 | 张量 | (B, N2, S1, G) | float32 | 主注意力前向 pass 的 softmax 行求和，由外部传入 |
| softmaxMaxIndex | 必选 | 张量 | (B, Nidx2, S1) | float32 | indexer 前向 pass 的 softmax 行最大值，由外部传入。Nidx2=1 |
| softmaxSumIndex | 必选 | 张量 | (B, Nidx2, S1) | float32 | indexer 前向 pass 的 softmax 行求和，由外部传入 |
| queryRope | 必选 | 张量 | (B, S1, N1, Dr) | float16/bfloat16 | query 的 RoPE 信息，Dr=64 |
| keyRope | 必选 | 张量 | (B, S2, N2, Dr) | float16/bfloat16 | key 的 RoPE 信息，Dr=64 |
| actual_seq_lengths_query | 可选 | 数组 | (B,) | int64 | 每个 batch query 的有效 token 数。传 None 表示与 S1 相同 |
| actual_seq_lengths_key | 可选 | 数组 | (B,) | int64 | 每个 batch key 的有效 token 数。传 None 表示与 S2 相同 |
| scaleValue | 必选 | 属性 | 标量 | double | 注意力缩放系数，对应公式中的 1/√d |
| layout | 必选 | 属性 | 标量 | string | 数据排布格式，默认 "BSND"，本算子仅支持 BSND |
| sparseMode | 必选 | 属性 | 标量 | int64 | 掩码模式，仅支持 3（rightDownCausal 下三角掩码） |

> 说明：属性 `pre_tokens`、`next_tokens` 用于稀疏计算的滑窗位置（仅 `sparseMode=0/4` 时生效，本算子仅支持 3 故不生效），仅支持默认最大值 2^63-1，参赛者无需修改。

### 输出规格

| 输出类型 | 必选/可选 | 形状 | 数据类型 | 含义 |
|---|---|---|---|---|
| dQueryIndex | 必选 | (B, S1, Nidx1, D) | float16/bfloat16 | queryIndex 的梯度 dq̃ |
| dKeyIndex | 必选 | (B, S2, Nidx2, D) | float16/bfloat16 | keyIndex 的梯度 dK̃ |
| dWeights | 必选 | (B, S1, Nidx1) | float16/bfloat16/float32 | weights 的梯度 dW |
| loss | 必选 | (1,) | float32 | KL 散度损失值 |

### 形状约束

- 仅支持训练场景。仅 Atlas A2/A3 系列产品支持，**Ascend 950PR/950DT 不支持**。
- HeadDim `D=128`（query 与 queryIndex 的 D 相同），RoPE 维度 `Dr=64`。
- query/key/queryIndex/keyIndex 数据类型必须一致；weights 不为 float32 时，五者数据类型必须一致。
- query/key/queryIndex/keyIndex/weights 不支持空 tensor。
- `sparseMode` 仅支持 3（rightDownCausal）。
- query 头数 `N1`∈{32,64,128}，key 头数 `N2=N1`（主注意力为标准 MHA）；queryIndex 头数 `Nidx1`∈{8,16,32,64}，keyIndex 头数 `Nidx2=1`（MQA）。
- `B` 取值 1~256；`S1、S2` 取值 1~128K，支持不等长。
- 仅支持 ND 数据格式（任意多维非结构化连续格式）。
- 默认非确定性实现，可通过 `aclrtCtxSetSysParamOpt` 开启确定性。

### 输出形状计算公式

- `dQueryIndex`：与 queryIndex 形状一致 `(B, S1, Nidx1, D)`。
- `dKeyIndex`：与 keyIndex 形状一致 `(B, S2, Nidx2, D)`。
- `dWeights`：与 weights 形状一致 `(B, S1, Nidx1)`。
- `loss`：标量 `(1,)`。

---

## 3. 算子逻辑说明

### 3.1 target 分布 p（主注意力）

对每个 query token，用 query/key 计算主注意力 score `q@K^T/√d`（每头独立），做 softmax，对所有头求和后沿 key 方向 L1 正则化，得到 target 分布 `p`（长度 S2）。实际算子用输入 `softmaxMax/softmaxSum` 重建 softmax 以避免重算。

### 3.2 预测分布 softmax(I)（Indexer）

计算 indexer 相似度 `S = q̃ @ K̃^T`，ReLU 激活，再用 weights 加权得 indexer 得分 `I = W @ ReLU(S)`，做 `softmax(I)` 得预测分布。实际算子用 `softmaxMaxIndex/softmaxSumIndex` 重建。

### 3.3 KL Loss

`Loss = Σ_t D_KL(p || softmax(I)) = Σ_t Σ_i p_i log(p_i / softmax(I)_i)`，约束预测分布逼近 target 分布。

### 3.4 反向梯度（链式求导）

- `dI = softmax(I) - p`（KL 散度对 I 的梯度）
- `dW = dI @ ReLU(S)^T`（weights 梯度）
- `dS = outer(W, dI) ⊙ (S>0)`，即 `dS[i,j] = W[i]·dI[j]·(S[i,j]>0)`（经 W 反传与 ReLU 反向）
- `dq̃ = dS @ K̃`（queryIndex 梯度）
- `dK̃ = dS^T @ q̃`（keyIndex 梯度）

### 3.5 掩码与 RoPE

- `sparseMode=3`（rightDownCausal）：query 序列右端对齐 key 序列右端的下三角掩码，query 位置 `i` 仅可见 key 位置 `j`，当 `j ≤ i + (S2 - S1)`（位置索引从 0 开始，假设 S2≥S1）；不满足条件的位置在 softmax 前屏蔽（不参与归一化）。该掩码同时应用于主注意力与 indexer score。
- RoPE（queryRope/keyRope）按旋转位置编码对 query/key 施加旋转并融合到主注意力计算。

---

## 4. 决赛任务要求

- **精度保障**：针对不同 `N1/Nidx1`、变长序列与 `sparseMode=3` 掩码，设计算子逻辑，保证 KL loss 及 dQueryIndex/dKeyIndex/dWeights 梯度精度正确（可用数值梯度校验）。
- **性能优化**：利用输入的 softmaxMax/Sum/Index 中间结果避免重算，优化双注意力（main + indexer）矩阵乘与 KL/反向的并行度，算子性能更优。
- **切分最优**：探索 query 序列维度、头维度与 key 上下文维度在多 batch/变长场景下的切分方式，找到不同输入 shape 场景下的最优解。

---

## 功能示例

```python
import numpy as np

def dense_lightning_indexer_grad_kl_loss(query, key, query_index, key_index, weights, scale_value):
    # query:(B,S1,N1,D) key:(B,S2,N2,D) N2=N1  (main attention, 每头独立 K)
    # query_index:(B,S1,Nidx1,D) key_index:(B,S2,Nidx2,D) Nidx2=1  weights:(B,S1,Nidx1)
    # 输出 dQueryIndex:(B,S1,Nidx1,D) dKeyIndex:(B,S2,Nidx2,D) dWeights:(B,S1,Nidx1) loss:标量
    # 聚焦核心 KL loss + indexer 反向；实际算子用 softmaxMax/Sum/Index 输入重建 softmax 避免重算，
    # 此处为简洁内部重算；省略 mask(sparseMode=3) 与 RoPE 融合(见 3.5 节)
    q = query.astype(np.float32); k = key.astype(np.float32)
    qi = query_index.astype(np.float32); ki = key_index.astype(np.float32); w = weights.astype(np.float32)
    B, S1, N1, D = q.shape; _, S2, N2, _ = k.shape; _, _, Nidx1, _ = qi.shape
    dQi = np.zeros_like(qi); dKi = np.zeros_like(ki); dW = np.zeros_like(w); loss = 0.0
    for b in range(B):
        for s in range(S1):
            # target 分布 p：主注意力各头 softmax 求和后 L1 正则化
            S_main = np.einsum('hd,jhd->hj', q[b, s], k[b]) * scale_value   # (N1, S2) 每头独立 score
            sm = np.exp(S_main - S_main.max(-1, keepdims=True)); sm /= sm.sum(-1, keepdims=True)
            p_raw = sm.sum(0); p = p_raw / p_raw.sum()                       # target 分布
            # 预测分布 softmax(I)
            S_idx = qi[b, s] @ ki[b, :, 0, :].T                              # (Nidx1, S2) indexer 相似度
            relu = np.maximum(S_idx, 0); I = w[b, s] @ relu                  # (S2,) indexer 得分
            softmax_I = np.exp(I - I.max()); softmax_I /= softmax_I.sum()
            # KL loss: D_KL(p || softmax_I)
            eps = 1e-12
            loss += np.sum(p * np.log((p + eps) / (softmax_I + eps)))
            # 反向：dI = softmax_I - p，链式求 dW/dq̃/dK̃
            dI = softmax_I - p
            dW[b, s] = dI @ relu.T                                           # (Nidx1,)
            dS = np.outer(w[b, s], dI) * (S_idx > 0)                        # (Nidx1, S2) ReLU 反向
            dQi[b, s] = dS @ ki[b, :, 0, :]                                   # (Nidx1, D)
            dKi[b, :, 0, :] += dS.T @ qi[b, s]                               # (S2, D) 累加（同一 key 位置可能被多 query 访问）
    return dQi.astype(np.float16), dKi.astype(np.float16), dW.astype(np.float16), loss

np.set_printoptions(precision=4, suppress=True)

# 示例1：KL loss 与反向梯度（BSND，B=1, S1=1, N1=32, N2=32, Nidx1=8, Nidx2=1, D=128, S2=4）
# query/queryIndex/weights 全 0.1；key/keyIndex 位置 j 全为 (j+1)*0.1（随位置递增）
D = 128; scale = 1.0 / np.sqrt(D)
B, S1, N1, N2, Nidx1, Nidx2, S2 = 1, 1, 32, 32, 8, 1, 4
query       = np.full((B, S1, N1, D), 0.1, dtype=np.float16)
key         = np.tile((np.arange(1, S2 + 1) * 0.1).reshape(1, S2, 1, 1), (1, 1, N2, D)).astype(np.float16)
query_index = np.full((B, S1, Nidx1, D), 0.1, dtype=np.float16)
key_index   = np.tile((np.arange(1, S2 + 1) * 0.1).reshape(1, S2, 1, 1), (1, 1, Nidx2, D)).astype(np.float16)
weights     = np.full((B, S1, Nidx1), 0.1, dtype=np.float16)
dQi, dKi, dW, loss = dense_lightning_indexer_grad_kl_loss(query, key, query_index, key_index, weights, scale)
# 输出：loss=0.4409；dQueryIndex/dKeyIndex/dWeights shape 分别为 (1,1,8,128)/(1,4,1,128)/(1,1,8)
# target 分布 p 偏向 key 位置 3（[0.209,0.234,0.263,0.294]），预测分布 softmax_I 更偏位置3（[0.030,0.084,0.234,0.651]）
# 因各头 queryIndex 相同，dWeights 各头相同（均≈1.107）
print('loss:', round(loss, 4))
print('dWeights(8 维):', dW[0, 0])

# 示例2：多 batch 场景（B=2，两个 batch 结构相同，各自独立计算梯度与 loss）
B = 2
query       = np.full((B, S1, N1, D), 0.1, dtype=np.float16)
key         = np.tile((np.arange(1, S2 + 1) * 0.1).reshape(1, S2, 1, 1), (B, 1, N2, D)).astype(np.float16)
query_index = np.full((B, S1, Nidx1, D), 0.1, dtype=np.float16)
key_index   = np.tile((np.arange(1, S2 + 1) * 0.1).reshape(1, S2, 1, 1), (B, 1, Nidx2, D)).astype(np.float16)
weights     = np.full((B, S1, Nidx1), 0.1, dtype=np.float16)
dQi, dKi, dW, loss = dense_lightning_indexer_grad_kl_loss(query, key, query_index, key_index, weights, scale)
# 输出形状：dQueryIndex=(2,1,8,128) dKeyIndex=(2,4,1,128) dWeights=(2,1,8)
# 两 batch 结构相同 => loss 为示例1的 2 倍（0.8818），各 batch 梯度一致
print('多 batch loss:', round(loss, 4))
```

