# 【CANN社区任务2026】+CANN开源开放+reshape_and_cache_kernel_flash算子优化方法

- topicId: 02195219910905760181
- source: https://www.hiascend.com/developer/blog/details/02195219910905760181
- section: CANN
- createTime: 20260720062146

# 数据类型调整
矢量运算单元不支持部分特定数据类型，计算时会退化为标量运算，影响性能，在确定不影响精度的情况下，建议使用支持的数据类型，提升性能。

案例：triton.language.where(condition, x, y)

i64实现：
xbar = tl.where(cols < N, X - mean, 0.0)

fp32实现：
cols_cmp = cols.to(tl.float32)
xbar = tl.where(cols_cmp < N, x - mean, 0.0)

# 离散访存优化
原生Triton作为面向GPU设计的语言，支持SIMT写法，通过自由设置mask实现离散访存，但是昇腾作为SIMD架构，在连续访存场景下可达到更高性能。

案例：load离散数据 y = x[idx]，先完整load到片上再通过gather选择

原实现：
idx = tl.load(idx_ptr + rn * stride_idx)
mask = idx < M
val = tl.load(x_ptr + idx * stride_x, mask=mask)

优化实现：
idx = tl.load(idx_ptr + rn * stride_idx)
x_shared = tl.load(x_ptr + rm * stride_x)
val = tl.gather(x_shared, idx, 0)

# 访存调度优化
1. GM是多核共享内存，其中有部分空间（192MB）与AICore之间存在高带宽搬运通路，就是L2 Cache。提升L2 Cache的命中率可以优化算子/模型的性能。

同一时间访存位置避免离散导致L2 Cache命中率低。

比如以下这个例子：
4核，每个核内tiling成3块sub block

**连续sub block**

| 1 | 2 | 3 | 1 |
|---|---|---|---|
| 2 | 3 | 1 | 2 |
| 3 | 1 | 2 | 3 |

→

| 1 | 2 | 3 | 1 |
|---|---|---|---|
| 2 | 3 | 1 | 2 |
| 3 | 1 | 2 | 3 |

L2 Cache命中率低

---

**非连续sub block**

| 1 | 1 | 1 | 1 |
|---|---|---|---|
| 2 | 2 | 2 | 2 |
| 3 | 3 | 3 | 3 |

→

| 1 | 1 | 1 | 1 |
|---|---|---|---|
| 2 | 2 | 2 | 2 |
| 3 | 3 | 3 | 3 |

L2 Cache命中率高

2. 避免同一时间需要数据过大，超出L2 Cache的部分访存速度较慢。
3. 避免同一时间大量核访问同一块内存导致读冲突。
