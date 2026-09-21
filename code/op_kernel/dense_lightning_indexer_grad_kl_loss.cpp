// Kernel侧核函数实现：DenseLightningIndexerGradKLLoss
//
// 数据布局采用 key-major（每行一个 key 位置）：
//   SC[j][h] = scale * q[h]·k[j][h]      主注意力分数
//   U[j][i]  = relu(qi[i]·ki[j])         indexer 相似度
// 每个 key 只做 1 次 GM 载入；一次向量指令覆盖“多头 × 多迭代”，
// 大幅减少小指令数量与核内同步；低精度输入用 T 精度乘、fp32 累加。
//
// 每个 query 行的流程：
//   1) 主注意力：逐 key 求 SC -> 每头最大值（按 key 行累积 Max）-> 减最大值取 exp ->
//      按头求和 -> 全部 key 求和后 L1 归一化得到 target p
//   2) indexer：逐 key 求 U（含 ReLU）-> logits = W·U -> softmax 得到预测分布 pred
//   3) loss 与梯度：dI = pred - p，ds = w*dI*(S>0)，dq = ds@K̃，dk = ds^T@Q̃，dW = ReLU(S)@dI
#include "kernel_operator.h"

#include "dense_lightning_indexer_grad_kl_loss_tiling.h"
#include "tiling_key_dense_lightning_indexer_grad_kl_loss.h"

using namespace AscendC;

namespace {
// 计算 tile 类型：fp32 输入直接用 float；fp16/bf16 用 half（bf16 先转成 half）
template <typename T>
struct ComputeTile {
    using Type = typename std::conditional<sizeof(T) == 4u, float, half>::type;
};
constexpr uint32_t FP32_PER_BLOCK = 8;      // 32B / 4B
constexpr uint32_t FP32_PER_REPEAT = 64;    // 256B / 4B
constexpr float INDICATOR_SCALE = 1.0e30f;
constexpr int8_t EV_MTE2_V = EVENT_ID0;
constexpr int8_t EV_V_MTE3 = EVENT_ID1;
constexpr int8_t EV_V_S = EVENT_ID2;
constexpr int8_t EV_S_V = EVENT_ID3;
constexpr int8_t EV_S_MTE3 = EVENT_ID4;
constexpr int8_t EV_MTE3_V = EVENT_ID5;
constexpr int8_t EV_V_MTE2 = EVENT_ID6;
}  // namespace

template <typename T, typename TW>
class KernelDenseLightningIndexerGradKlLoss {
public:
    using CT = typename ComputeTile<T>::Type;

    __aicore__ inline KernelDenseLightningIndexerGradKlLoss() {}

    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR query_index, GM_ADDR key_index,
                                GM_ADDR weights, GM_ADDR d_query_index, GM_ADDR d_key_index,
                                GM_ADDR d_weights, GM_ADDR loss, GM_ADDR workspace,
                                const DenseLightningIndexerGradKlLossTilingData &td) {
        batch_ = td.batch;
        s1_ = td.s1;
        s2_ = td.s2;
        n1_ = td.n1;
        nidx_ = td.nidx;
        dim_ = td.dim;
        dimPad_ = td.dimPad;
        visPad_ = td.visPad;
        n1Pad_ = td.n1Pad;
        nidxPad_ = td.nidxPad;
        causal_ = td.causal;
        rowsPerTask_ = td.rowsPerTask;
        blocksPerBatch_ = td.blocksPerBatch;
        taskCount_ = td.taskCount;
        headBlock_ = td.headBlock;
        dkRows_ = td.dkRows;
        splitDk_ = td.splitDk;
        dkPartialElems_ = td.dkPartialElems;
        dkOffset_ = td.dkOffset;
        lossOffset_ = td.lossOffset;
        scale_ = td.scale;
        stageElems_ = td.stageElems;
        kiCacheRows_ = td.kiCacheRows;
        kChunkRows_ = td.kChunkRows > 0u ? td.kChunkRows : 1u;
        inputBytes_ = static_cast<uint32_t>(sizeof(T));
        weightBytes_ = static_cast<uint32_t>(sizeof(TW));

        const uint64_t qRows = static_cast<uint64_t>(batch_) * s1_ * n1_;
        const uint64_t kRows = static_cast<uint64_t>(batch_) * s2_ * n1_;
        const uint64_t qiRows = static_cast<uint64_t>(batch_) * s1_ * nidx_;
        const uint64_t kiRows = static_cast<uint64_t>(batch_) * s2_;
        const uint64_t wRows = static_cast<uint64_t>(batch_) * s1_ * nidx_;
        queryGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(query), qRows * dim_);
        keyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(key), kRows * dim_);
        queryIndexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(query_index), qiRows * dim_);
        keyIndexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(key_index), kiRows * dim_);
        weightsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ TW *>(weights), wRows);
        dQueryIndexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(d_query_index), qiRows * dim_);
        dKeyIndexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(d_key_index), kiRows * dim_);
        dWeightsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ TW *>(d_weights), wRows);
        lossGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(loss), 1);
        const uint64_t wsElems = static_cast<uint64_t>(dkOffset_) +
                                 static_cast<uint64_t>(dkPartialElems_) * taskCount_ +
                                 static_cast<uint64_t>(lossOffset_) + 64u;
        workspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace), wsElems);

        layout_ = DliglComputeUbLayout(headBlock_, n1_, nidx_, dimPad_, visPad_, n1Pad_, nidxPad_,
                                       dkRows_, inputBytes_, weightBytes_, stageElems_,
                                       kiCacheRows_, kChunkRows_);
        BuildTensors();
    }

    __aicore__ inline void Process() {
        const uint32_t coreIdx = static_cast<uint32_t>(GetBlockIdx());
        coreNum_ = static_cast<uint32_t>(GetBlockNum());
        totalLoss_ = 0.0f;
        Duplicate(scratch_[128], 1.0f, 64);
        Duplicate(lz_, 1.0f, 64);
        Duplicate(scratch_[384], 0.0f, 8);
        for (uint32_t task = coreIdx; task < taskCount_; task += coreNum_) {
            ProcessTask(task);
        }
        if (coreNum_ > 1u) {
            SyncAll();
        }
        FinishLoss(coreIdx);
        if (dkPartialElems_ > 0u) {
            ReduceDk(coreIdx);
        }
    }

private:
    __aicore__ inline void BuildTensors() {
        qT_ = LocalTensor<CT>(TPosition::VECCALC, layout_.qTOff, n1_ * dimPad_);
        kT_ = LocalTensor<CT>(TPosition::VECCALC, layout_.kTOff,
                              kChunkRows_ * n1_ * dimPad_);
        prodT_ = LocalTensor<CT>(TPosition::VECCALC, layout_.prodTOff, headBlock_ * dimPad_);
        prod_ = LocalTensor<float>(TPosition::VECCALC, layout_.prodOff, headBlock_ * dimPad_);
        outT_ = LocalTensor<T>(TPosition::VECCALC, layout_.outTOff, dimPad_);
        stageRaw_ = LocalTensor<T>(TPosition::VECCALC, layout_.stageOff, stageElems_);
        qiStage_ = LocalTensor<T>(TPosition::VECCALC, layout_.qiStageOff,
                                  QI_STAGE_ROWS * dimPad_);
        dqStage_ = LocalTensor<T>(TPosition::VECCALC, layout_.qiStageOff, QI_STAGE_ROWS * dimPad_);
        qiF_ = LocalTensor<float>(TPosition::VECCALC, layout_.qiFOff, nidx_ * dimPad_);
        kiT_ = LocalTensor<CT>(TPosition::VECCALC, layout_.kiTOff, dimPad_);
        if (kiCacheRows_ > 0u) {
            kiCache_ = LocalTensor<float>(TPosition::VECCALC, layout_.kiCacheOff, kiCacheRows_ * dimPad_);
            kiCacheT_ = LocalTensor<T>(TPosition::VECCALC, layout_.kiCacheTOff, kiCacheRows_ * dimPad_);
        }
        kiF_ = LocalTensor<float>(TPosition::VECCALC, layout_.kiFOff, dimPad_);
        sc_ = LocalTensor<float>(TPosition::VECCALC, layout_.scOff, visPad_ * n1Pad_);
        u_ = LocalTensor<float>(TPosition::VECCALC, layout_.uOff, visPad_ * nidxPad_);
        tgt_ = LocalTensor<float>(TPosition::VECCALC, layout_.tgtOff, visPad_);
        sh_ = LocalTensor<float>(TPosition::VECCALC, layout_.shOff, visPad_);
        pred_ = LocalTensor<float>(TPosition::VECCALC, layout_.predOff, visPad_);
        lossAcc_ = LocalTensor<float>(TPosition::VECCALC, layout_.lossAccOff, visPad_);
        lz_ = LocalTensor<float>(TPosition::VECCALC, layout_.lzOff, 64u);
        del_ = LocalTensor<float>(TPosition::VECCALC, layout_.delOff, visPad_);
        dkAcc_ = LocalTensor<float>(TPosition::VECCALC, layout_.dkOff, dkRows_ * dimPad_);
        dkRow_ = LocalTensor<float>(TPosition::VECCALC, layout_.dkRowOff, dimPad_);
        dkOutT_ = LocalTensor<T>(TPosition::VECCALC, layout_.dkOutTOff, dkRows_ * dimPad_);
        dq_ = LocalTensor<float>(TPosition::VECCALC, layout_.dqOff, nidx_ * dimPad_);
        maxVec_ = LocalTensor<float>(TPosition::VECCALC, layout_.maxOff, n1Pad_);
        dblk_ = LocalTensor<float>(TPosition::VECCALC, layout_.dblkOff, visPad_ * 8u);
        mblk_ = LocalTensor<float>(TPosition::VECCALC, layout_.mblkOff, nidx_ * 8u);
        part_ = LocalTensor<float>(TPosition::VECCALC, layout_.partOff, 2048u);
        wRaw_ = LocalTensor<TW>(TPosition::VECCALC, layout_.wRawOff, nidx_);
        w_ = LocalTensor<float>(TPosition::VECCALC, layout_.wOff, nidx_);
        wb_ = LocalTensor<float>(TPosition::VECCALC, layout_.wbOff, nidx_ * 8u);
        dw_ = LocalTensor<float>(TPosition::VECCALC, layout_.dwOff, nidx_);
        dwRow_ = LocalTensor<float>(TPosition::VECCALC, layout_.dwRowOff, nidx_);
        scratch_ = LocalTensor<float>(TPosition::VECCALC, layout_.tmpOff, 8u * 64u);
    }

    __aicore__ inline uint32_t MinU32(uint32_t a, uint32_t b) const { return a < b ? a : b; }
    __aicore__ inline uint32_t Align8(uint32_t value) const { return (value + 7u) & ~7u; }
    __aicore__ inline uint32_t PiecesOf(uint32_t width) const { return (width + 63u) / 64u; }
    __aicore__ inline uint32_t MaskOf(uint32_t width, uint32_t piece) const {
        const uint32_t left = width - piece * 64u;
        return left > 64u ? 64u : left;
    }
    // 低精度 tile 的 repeat/block 元素数
    static constexpr uint32_t TLanes() { return 256u / sizeof(CT); }
    static constexpr uint32_t TBlocks() { return 32u / sizeof(CT); }

    __aicore__ inline void SyncMte2ToVec() {
        SetFlag<HardEvent::MTE2_V>(EV_MTE2_V);
        WaitFlag<HardEvent::MTE2_V>(EV_MTE2_V);
    }
    __aicore__ inline void SyncVecToMte3() {
        SetFlag<HardEvent::V_MTE3>(EV_V_MTE3);
        WaitFlag<HardEvent::V_MTE3>(EV_V_MTE3);
    }
    __aicore__ inline void SyncMte3Done() {
        SetFlag<HardEvent::MTE3_V>(EV_MTE3_V);
        WaitFlag<HardEvent::MTE3_V>(EV_MTE3_V);
    }
    __aicore__ inline void SyncVecDone() {
        SetFlag<HardEvent::V_MTE2>(EV_V_MTE2);
        WaitFlag<HardEvent::V_MTE2>(EV_V_MTE2);
    }
    __aicore__ inline float ReadScalar(const LocalTensor<float> &src) {
        SetFlag<HardEvent::V_S>(EV_V_S);
        WaitFlag<HardEvent::V_S>(EV_V_S);
        return src.GetValue(0);
    }

    __aicore__ inline float VecSum(const LocalTensor<float> &src, uint32_t count) {
        const uint32_t pieces = PiecesOf(count);
        for (uint32_t p = 0; p < pieces; ++p) {
            WholeReduceSum<float>(part_[p * 8], src[p * 64u], MaskOf(count, p), 1, 1, 1, 8);
        }
        if (pieces > 1u) {
            WholeReduceSum<float>(part_, part_, pieces, 1, 1, 1, 1);
        }
        return ReadScalar(part_);
    }

    __aicore__ inline float VecMax(const LocalTensor<float> &src, uint32_t count) {
        const uint32_t pieces = PiecesOf(count);
        WholeReduceMax<float>(part_, src, MaskOf(count, 0), 1, 1, 1, 8,
                              ReduceOrder::ORDER_ONLY_VALUE);
        for (uint32_t p = 1; p < pieces; ++p) {
            WholeReduceMax<float>(part_[p * 8], src[p * 64u], MaskOf(count, p), 1, 1, 1, 8,
                                  ReduceOrder::ORDER_ONLY_VALUE);
            Max(part_, part_, part_[p * 8], 8);
        }
        return ReadScalar(part_);
    }

    __aicore__ inline float ScalarLn(float value) {
        scratch_[384].SetValue(0, value);
        SetFlag<HardEvent::S_V>(EV_S_V);
        WaitFlag<HardEvent::S_V>(EV_S_V);
        Ln(scratch_[384], scratch_[384], 8);
        return ReadScalar(scratch_[384]);
    }

    // ------------------------------------------------------------- 载入
    // dimPad == dim 时源/目的都连续且长度按 32B 对齐，可用单条 DataCopy 代替按行
    // DataCopyPad：后者每行一个 256B 小 burst，MTE2 实测被 burst 数量而非带宽限制。
    __aicore__ inline bool BulkOk(uint64_t base, uint64_t elems) const {
        return dimPad_ == dim_ && elems != 0ull && (elems * sizeof(CT)) % 32u == 0u &&
               (base * sizeof(T)) % 32u == 0u;
    }

    // 只发起 GM->UB 搬运、不等待：同一行内多次搬运共享一次 MTE2 等待，
    // 把“逐块 DMA 往返”变成一次往返（实测 loads 阶段是本算子最大可控开销）。
    __aicore__ inline void IssueRows(const GlobalTensor<T> &src, uint64_t base, uint64_t elems,
                                     LocalTensor<CT> dst) {
        if (BulkOk(base, elems)) {
            DataCopy(dst, src[base], static_cast<uint32_t>(elems));
            return;
        }
        const uint32_t rows = static_cast<uint32_t>(elems / dim_);
        DataCopyExtParams params{static_cast<uint16_t>(rows),
                                 static_cast<uint32_t>(dim_ * inputBytes_),
                                 static_cast<uint32_t>((dimPad_ - dim_) * inputBytes_), 0, 0};
        DataCopyPad(dst, src[base], params, DataCopyPadExtParams<CT>{false, 0, 0, 0});
    }

    __aicore__ inline void LoadRows(const GlobalTensor<T> &src, uint64_t base, uint32_t rows,
                                    LocalTensor<CT> dst) {
        // 覆盖目的缓冲前，先等上一轮矢量读和 MTE3 读结束
        SyncVecDone();
        SyncMte3Done();
        const uint64_t bulkElems = static_cast<uint64_t>(rows) * dim_;
        if (BulkOk(base, bulkElems)) {
            DataCopy(dst, src[base], static_cast<uint32_t>(bulkElems));
            SyncMte2ToVec();
            return;
        }
        DataCopyExtParams params{static_cast<uint16_t>(rows),
                                 static_cast<uint32_t>(dim_ * inputBytes_),
                                 static_cast<uint32_t>((dimPad_ - dim_) * inputBytes_), 0, 0};
        if constexpr (sizeof(T) == 2u && !std::is_same<T, half>::value) {
            // bf16：向量指令不支持 bf16 乘加，先落 T 暂存再转 half
            DataCopyPad(stageRaw_, src[base], params, DataCopyPadExtParams<T>{false, 0, 0, 0});
            SyncMte2ToVec();
            Cast(dst, stageRaw_, RoundMode::CAST_NONE, rows * dimPad_);
        } else {
            DataCopyPad(dst, src[base], params, DataCopyPadExtParams<CT>{false, 0, 0, 0});
            SyncMte2ToVec();
        }
    }

    __aicore__ inline void LoadQRow(uint32_t b, uint32_t row) {
        const uint64_t base = (static_cast<uint64_t>(b) * s1_ + row) * n1_ * dim_;
        LoadRows(queryGm_, base, n1_, qT_);
    }

    __aicore__ inline void LoadKRow(uint32_t b, uint32_t j) {
        const uint64_t base = (static_cast<uint64_t>(b) * s2_ + j) * n1_ * dim_;
        LoadRows(keyGm_, base, n1_, kT_);
    }

    // 一次连续载入 rows 个 key 的整块 key（GM 上相邻 key 的 (n1, dim) 连续）
    __aicore__ inline void LoadKChunk(uint32_t b, uint32_t j0, uint32_t rows) {
        const uint64_t base = (static_cast<uint64_t>(b) * s2_ + j0) * n1_ * dim_;
        const uint64_t elems = static_cast<uint64_t>(rows) * n1_ * dim_;
        SyncVecDone();
        SyncMte3Done();
        if (BulkOk(base, elems)) {
            DataCopy(kT_, keyGm_[base], static_cast<uint32_t>(elems));
            SyncMte2ToVec();
            return;
        }
        DataCopyExtParams params{static_cast<uint16_t>(rows * n1_),
                                 static_cast<uint32_t>(dim_ * inputBytes_),
                                 static_cast<uint32_t>((dimPad_ - dim_) * inputBytes_), 0, 0};
        if constexpr (sizeof(T) == 2u && !std::is_same<T, half>::value) {
            DataCopyPad(stageRaw_, keyGm_[base], params, DataCopyPadExtParams<T>{false, 0, 0, 0});
            SyncMte2ToVec();
            Cast(kT_, stageRaw_, RoundMode::CAST_NONE, rows * n1_ * dimPad_);
        } else {
            DataCopyPad(kT_, keyGm_[base], params, DataCopyPadExtParams<CT>{false, 0, 0, 0});
            SyncMte2ToVec();
        }
    }

    // 前奏中只发起、不等待（与其它搬运共享一次 MTE2 等待）
    __aicore__ inline void IssueQiFirst(uint32_t b, uint32_t row) {
        if (nidx_ > QI_STAGE_ROWS) {
            return;
        }
        const uint64_t base = (static_cast<uint64_t>(b) * s1_ + row) * nidx_ * dim_;
        const uint64_t elems = static_cast<uint64_t>(nidx_) * dim_;
        if constexpr (sizeof(T) == 4u) {
            IssueRows(queryIndexGm_, base, elems, qiF_);
        } else {
            IssueRows(queryIndexGm_, base, elems, qiStage_);
        }
    }

    __aicore__ inline void LoadQiAll(uint32_t b, uint32_t row, bool firstIssued = false) {
        const uint64_t base = (static_cast<uint64_t>(b) * s1_ + row) * nidx_ * dim_;
        if constexpr (sizeof(T) == 4u) {
            if (firstIssued) {
                return;
            }
            LoadRows(queryIndexGm_, base, nidx_, qiF_);
        } else {
            if (firstIssued) {
                Cast(qiF_, qiStage_, RoundMode::CAST_NONE, nidx_ * dimPad_);
                return;
            }
            // 分块载入并转 fp32，避免常驻一块 (nidx, dimPad) 的低精度缓冲
            for (uint32_t i0 = 0; i0 < nidx_; i0 += QI_STAGE_ROWS) {
                const uint32_t rows = MinU32(QI_STAGE_ROWS, nidx_ - i0);
                const uint64_t chunkBase = base + static_cast<uint64_t>(i0) * dim_;
                SyncVecDone();
                SyncMte3Done();
                if (BulkOk(chunkBase, static_cast<uint64_t>(rows) * dim_)) {
                    DataCopy(qiStage_, queryIndexGm_[chunkBase],
                             static_cast<uint32_t>(rows) * dim_);
                } else {
                    DataCopyExtParams params{static_cast<uint16_t>(rows),
                                             static_cast<uint32_t>(dim_ * inputBytes_), 0, 0, 0};
                    DataCopyPad(qiStage_, queryIndexGm_[chunkBase], params,
                                DataCopyPadExtParams<T>{false, 0, 0, 0});
                }
                SyncMte2ToVec();
                Cast(qiF_[i0 * dimPad_], qiStage_, RoundMode::CAST_NONE, rows * dimPad_);
            }
        }
    }

    __aicore__ inline void LoadKiRow(uint32_t b, uint32_t j) {
        const uint64_t base = (static_cast<uint64_t>(b) * s2_ + j) * dim_;
        LoadRows(keyIndexGm_, base, 1u, kiT_);
    }

    // 一次载入该 batch 全部 key 的 keyIndex（仅当缓存能装下 s2 行时）
    __aicore__ inline void IssueKiAll(uint32_t b) {
        if (kiCacheRows_ < s2_) {
            return;
        }
        const uint64_t base = static_cast<uint64_t>(b) * s2_ * dim_;
        const uint64_t elems = static_cast<uint64_t>(s2_) * dim_;
        if constexpr (sizeof(T) == 4u) {
            IssueRows(keyIndexGm_, base, elems, kiCache_);
        } else {
            IssueRows(keyIndexGm_, base, elems, kiCacheT_);
        }
    }

    __aicore__ inline void FinishKiAll() {
        if (kiCacheRows_ < s2_) {
            return;
        }
        if constexpr (sizeof(T) != 4u) {
            Cast(kiCache_, kiCacheT_, RoundMode::CAST_NONE, s2_ * dimPad_);
        }
    }

    // 取第 j 个 key 的 ki（fp32）：优先走缓存，否则逐 key 载入
    __aicore__ inline LocalTensor<float> KiRow(uint32_t b, uint32_t j) {
        if (kiCacheRows_ >= s2_) {
            return kiCache_[j * dimPad_];
        }
        LoadKiRow(b, j);
        if constexpr (sizeof(CT) == 4u) {
            Adds(kiF_, kiT_, 0.0f, dimPad_);
        } else {
            SyncVecDone();
            Cast(kiF_, kiT_, RoundMode::CAST_NONE, dimPad_);
        }
        return kiF_;
    }

    __aicore__ inline void IssueWeights(uint32_t b, uint32_t row) {
        const uint64_t base = (static_cast<uint64_t>(b) * s1_ + row) * nidx_;
        DataCopyExtParams params{1, static_cast<uint32_t>(nidx_ * weightBytes_), 0, 0, 0};
        DataCopyPad(wRaw_, weightsGm_[base], params, DataCopyPadExtParams<TW>{false, 0, 0, 0});
    }

    __aicore__ inline void FinishWeights() {
        if constexpr (sizeof(TW) != 4u) {
            Cast(w_, wRaw_, RoundMode::CAST_NONE, nidx_);
        } else {
            Adds(w_, wRaw_, 0.0f, nidx_);
        }
        Brcb(wb_, w_, static_cast<uint8_t>((nidx_ + 7u) / 8u), {1, 8});
    }

    // 逐行 fp32 归约：dst[r] = sum_{d<width} src[r*pitch + d]，rows 行
    __aicore__ inline void ReduceRowsFp32(const LocalTensor<float> &dst, const LocalTensor<float> &src,
                                          uint32_t rows, uint32_t width, uint32_t pitch) {
        const uint32_t pieces = (width + FP32_PER_REPEAT - 1u) / FP32_PER_REPEAT;
        const uint8_t stride = static_cast<uint8_t>(pitch / FP32_PER_BLOCK);
        for (uint32_t p = 0; p < pieces; ++p) {
            const uint32_t left = width - p * FP32_PER_REPEAT;
            const uint32_t mask = left > FP32_PER_REPEAT ? FP32_PER_REPEAT : left;
            // 每个 piece 的逐行结果依次存放在 part_[p*rows .. p*rows+rows)
            WholeReduceSum<float>(part_[p * rows], src[p * FP32_PER_REPEAT], mask,
                                  static_cast<int32_t>(rows), 1, 1, static_cast<int32_t>(stride));
        }
        if (pieces > 1u) {
            for (uint32_t p = 1; p < pieces; ++p) {
                Add(part_, part_, part_[p * rows], rows);
            }
        }
        Adds(dst, part_, 0.0f, static_cast<int32_t>(rows));
    }

    __aicore__ inline uint32_t VisibleOf(uint32_t row) const {
        if (causal_ == 0u) {
            return s2_;
        }
        const int64_t visible =
            static_cast<int64_t>(row) + static_cast<int64_t>(s2_) - static_cast<int64_t>(s1_) + 1;
        if (visible <= 0) {
            return 0u;
        }
        return visible > static_cast<int64_t>(s2_) ? s2_ : static_cast<uint32_t>(visible);
    }

    // ------------------------------------------------------------- 主注意力分数
    __aicore__ inline void ComputeScoreBlock(uint32_t j, uint32_t kLocal, uint32_t h0,
                                              uint32_t hc) {
        constexpr uint32_t LANES = 256u / sizeof(CT);
        const uint32_t kBase = kLocal * n1_ * dimPad_;
        constexpr uint8_t TSTRIDE_DIV = TBlocks();
        const uint32_t pieces = (dim_ + LANES - 1u) / LANES;
        const uint32_t redWidth = (pieces > 1u) ? (dim_ < LANES ? dim_ : LANES) : dim_;
        const uint8_t tStride = static_cast<uint8_t>(dimPad_ / TSTRIDE_DIV);
        const uint8_t fStride = static_cast<uint8_t>(dimPad_ / FP32_PER_BLOCK);
        for (uint32_t p = 0; p < pieces; ++p) {
            const uint32_t left = dim_ - p * LANES;
            const uint32_t mask = left > LANES ? LANES : left;
            if constexpr (sizeof(CT) == 4u) {
                BinaryRepeatParams params{1, 1, 1, fStride, fStride, fStride};
                Mul(prod_[p * LANES], kT_[kBase + h0 * dimPad_ + p * LANES],
                    qT_[h0 * dimPad_ + p * LANES], mask, static_cast<uint8_t>(hc), params);
            } else {
                BinaryRepeatParams params{1, 1, 1, tStride, tStride, tStride};
                Mul(prodT_[p * LANES], kT_[kBase + h0 * dimPad_ + p * LANES],
                    qT_[h0 * dimPad_ + p * LANES], mask, static_cast<uint8_t>(hc), params);
            }
        }
        if constexpr (sizeof(CT) == 4u) {
            if (pieces > 1u) {
                BinaryRepeatParams addParams{1, 1, 1, fStride, fStride, fStride};
                for (uint32_t p = 1; p < pieces; ++p) {
                    Add(prod_, prod_, prod_[p * LANES], LANES, static_cast<uint8_t>(hc), addParams);
                }
            }
            ReduceRowsFp32(sc_[j * n1Pad_ + h0], prod_, hc, redWidth, dimPad_);
        } else {
            if (pieces > 1u) {
                BinaryRepeatParams addParams{1, 1, 1, tStride, tStride, tStride};
                for (uint32_t p = 1; p < pieces; ++p) {
                    Add(prodT_, prodT_, prodT_[p * LANES], LANES, static_cast<uint8_t>(hc),
                        addParams);
                }
            }
            Cast(prod_, prodT_, RoundMode::CAST_NONE, hc * dimPad_);
            ReduceRowsFp32(sc_[j * n1Pad_ + h0], prod_, hc, redWidth, dimPad_);
        }
    }

    __aicore__ inline void ComputeScores(uint32_t b, uint32_t vis) {
        for (uint32_t c = 0; c < vis; c += kChunkRows_) {
            const uint32_t rows = MinU32(kChunkRows_, vis - c);
            if (c != 0u) {
                LoadKChunk(b, c, rows);
            }
            for (uint32_t j = c; j < c + rows; ++j) {
                for (uint32_t h0 = 0; h0 < n1_; h0 += headBlock_) {
                    const uint32_t hc = MinU32(headBlock_, n1_ - h0);
                    ComputeScoreBlock(j, j - c, h0, hc);
                }
            }
        }
        Muls(sc_, sc_, scale_, visPad_ * n1Pad_);
    }

    // 每头最大值（跨 key）-> target_j = sum_h exp(s_hj - m_h)，最后统一 L1 归一化
    __aicore__ inline void ComputeTarget(uint32_t vis) {
        Duplicate(tgt_, 0.0f, visPad_);   // padding 保持 0，便于按 visPad 做运算
        Duplicate(maxVec_, -3.0e38f, n1Pad_);
        const uint32_t nPieces = PiecesOf(n1_);
        for (uint32_t j = 0; j < vis; ++j) {
            for (uint32_t p = 0; p < nPieces; ++p) {
                Max(maxVec_[p * 64u], maxVec_[p * 64u], sc_[j * n1Pad_ + p * 64u], MaskOf(n1_, p));
            }
        }
        for (uint32_t j = 0; j < vis; ++j) {
            for (uint32_t p = 0; p < nPieces; ++p) {
                Sub(sc_[j * n1Pad_ + p * 64u], sc_[j * n1Pad_ + p * 64u], maxVec_[p * 64u],
                    MaskOf(n1_, p));
            }
            Exp(sc_[j * n1Pad_], sc_[j * n1Pad_], n1_);
            if (nPieces == 1u) {
                WholeReduceSum<float>(tgt_[j], sc_[j * n1Pad_], MaskOf(n1_, 0), 1, 1, 1, 1);
                continue;
            }
            // n1 > 64：各 64 头分片分别归约（结果间隔 8 个槽位），再用向量相加、
            // 只做一次标量回读，避免每个分片一次 V->S 同步。
            for (uint32_t p = 0; p < nPieces; ++p) {
                WholeReduceSum<float>(part_[p * 8], sc_[j * n1Pad_ + p * 64u], MaskOf(n1_, p), 1,
                                      1, 1, 1);
            }
            for (uint32_t p = 1; p < nPieces; ++p) {
                Add(part_, part_, part_[p * 8], 8);
            }
            tgt_.SetValue(j, ReadScalar(part_));
            SetFlag<HardEvent::S_V>(EV_S_V);
            WaitFlag<HardEvent::S_V>(EV_S_V);
        }
        const float total = VecSum(tgt_, vis);
        Muls(tgt_, tgt_, 1.0f / total, vis);
    }

    // ------------------------------------------------------------- indexer 相似度
    // U[j][i] = relu(qi[i] · ki[j])：qi 常驻 fp32，ki 逐 key 转 fp32
    __aicore__ inline void ComputeSimilarity(uint32_t b, uint32_t vis) {
        const uint32_t pieces = (dim_ + FP32_PER_REPEAT - 1u) / FP32_PER_REPEAT;
        const uint32_t redWidth =
            (pieces > 1u) ? (dim_ < FP32_PER_REPEAT ? dim_ : FP32_PER_REPEAT) : dim_;
        const uint8_t fStride = static_cast<uint8_t>(dimPad_ / FP32_PER_BLOCK);
        // prod_ 只有 headBlock 行容量，索引头按 headBlock 分块，避免 nidx > headBlock 越界
        const uint32_t chunk = headBlock_ == 0u ? 1u : headBlock_;
        for (uint32_t j = 0; j < vis; ++j) {
            const LocalTensor<float> ki = KiRow(b, j);
            for (uint32_t i0 = 0; i0 < nidx_; i0 += chunk) {
                const uint32_t hc = MinU32(chunk, nidx_ - i0);
                for (uint32_t p = 0; p < pieces; ++p) {
                    const uint32_t left = dim_ - p * FP32_PER_REPEAT;
                    const uint32_t mask = left > FP32_PER_REPEAT ? FP32_PER_REPEAT : left;
                    BinaryRepeatParams params{1, 1, 1, fStride, fStride, 0};
                    Mul(prod_[p * FP32_PER_REPEAT], qiF_[i0 * dimPad_ + p * FP32_PER_REPEAT],
                        ki[p * FP32_PER_REPEAT], mask, static_cast<uint8_t>(hc), params);
                }
                if (pieces > 1u) {
                    BinaryRepeatParams addParams{1, 1, 1, fStride, fStride, fStride};
                    for (uint32_t p = 1; p < pieces; ++p) {
                        Add(prod_, prod_, prod_[p * FP32_PER_REPEAT], FP32_PER_REPEAT,
                            static_cast<uint8_t>(hc), addParams);
                    }
                }
                ReduceRowsFp32(u_[j * nidxPad_ + i0], prod_, hc, redWidth, dimPad_);
            }
            Maxs(u_[j * nidxPad_], u_[j * nidxPad_], 0.0f, nidx_);  // ReLU
        }
    }

    // logits_j = sum_i w_i * relu(u_ij)
    __aicore__ inline void ComputeLogits(uint32_t vis) {
        const uint32_t pieces = PiecesOf(nidx_);
        // padding 位置置为 -1e30：后续按 visPad 宽度做 softmax 时贡献恒为 0
        Duplicate(sh_, -1.0e30f, visPad_);
        for (uint32_t j = 0; j < vis; ++j) {
            for (uint32_t p = 0; p < pieces; ++p) {
                Mul(dwRow_[p * 64u], u_[j * nidxPad_ + p * 64u], w_[p * 64u], MaskOf(nidx_, p));
            }
            if (pieces == 1u) {
                WholeReduceSum<float>(sh_[j], dwRow_, MaskOf(nidx_, 0), 1, 1, 1, 1);
            } else {
                WholeReduceSum<float>(part_, dwRow_, MaskOf(nidx_, 0), 1, 1, 1, 1);
                for (uint32_t p = 1; p < pieces; ++p) {
                    WholeReduceSum<float>(part_[p * 8], dwRow_[p * 64u], MaskOf(nidx_, p), 1, 1, 1,
                                          1);
                    Add(part_, part_, part_[p * 8], 8);
                }
                Adds(sh_[j], part_, 0.0f, 1);
            }
        }
    }

    __aicore__ inline void SoftmaxAndLoss(uint32_t vis) {
        // 统一按 visPad（8 的倍数）计算：sh_ 的 padding 已置 -1e30，exp 后为 0
        const uint32_t width = visPad_;
        const float logitMax = VecMax(sh_, width);
        Adds(sh_, sh_, -logitMax, width);
        Exp(pred_, sh_, width);
        const float normalizer = VecSum(pred_, width);
        Muls(pred_, pred_, 1.0f / normalizer, width);
        Sub(del_, pred_, tgt_, width);
        // loss 项：sum p*(ln p - shifted) + ln Z（tgt padding 为 0）
        Maxs(scratch_, tgt_, 1.0e-30f, width);
        Ln(scratch_, scratch_, width);
        Mul(scratch_, scratch_, tgt_, width);
        Mul(scratch_[64], tgt_, sh_, width);
        Sub(scratch_, scratch_, scratch_[64], width);
        // 逐 key 项按向量累积，lnZ 只写第 0 个元素：整个任务只在末尾做一次归约
        Add(lossAcc_, lossAcc_, scratch_, width);
        lz_.SetValue(0, normalizer);
        SetFlag<HardEvent::S_V>(EV_S_V);
        WaitFlag<HardEvent::S_V>(EV_S_V);
        Ln(lz_, lz_, 8);
        Add(lossAcc_, lossAcc_, lz_, 8);
        (void)vis;
    }

    // ------------------------------------------------------------- 梯度
    __aicore__ inline void ComputeGradients(uint32_t b, uint32_t vis) {
        Duplicate(dq_, 0.0f, nidx_ * dimPad_);
        Duplicate(dw_, 0.0f, nidx_);
        Brcb(dblk_, del_, static_cast<uint8_t>((vis + 7u) / 8u), {1, 8});
        const uint32_t dimPieces = PiecesOf(dim_);
        const uint32_t nidxPieces = PiecesOf(nidx_);
        const uint8_t dimStride = static_cast<uint8_t>(dimPad_ / FP32_PER_BLOCK);
        for (uint32_t j = 0; j < vis; ++j) {
            const LocalTensor<float> ki = KiRow(b, j);
            // m = w * delta_j * (u > 0)，u 已含 ReLU
            Muls(dwRow_, u_[j * nidxPad_], INDICATOR_SCALE, nidx_);
            Mins(dwRow_, dwRow_, 1.0f, nidx_);
            for (uint32_t p = 0; p < nidxPieces; ++p) {
                Mul(dwRow_[p * 64u], dwRow_[p * 64u], dblk_[j * 8], MaskOf(nidx_, p),
                    static_cast<uint8_t>(1), {1, 1, 0, 1, 1, 1});
            }
            Mul(dwRow_, dwRow_, w_, nidx_);
            // dw += relu(u_j) * delta_j
            for (uint32_t p = 0; p < nidxPieces; ++p) {
                MulAddDst(dw_[p * 64u], u_[j * nidxPad_ + p * 64u], dblk_[j * 8], MaskOf(nidx_, p),
                          static_cast<uint8_t>(1), {1, 1, 0, 1, 1, 1});
            }
            Brcb(mblk_, dwRow_, static_cast<uint8_t>((nidx_ + 7u) / 8u), {1, 8});
            // dq += ki_j ⊗ m_j
            for (uint32_t p = 0; p < dimPieces; ++p) {
                BinaryRepeatParams dqParams{1, 1, 0, dimStride, 0, 1};
                MulAddDst(dq_[p * 64u], ki[p * 64u], mblk_, MaskOf(dim_, p),
                          static_cast<uint8_t>(nidx_), dqParams);
            }
            // dk[j] += sum_i qi[i] * m_ij
            // 用 dstRepStride=0 让一条指令跨 repeat 累加：repeat 对应 index head，
            // src0 按 qiF_ 行距前进，src1 走 mblk_ 的每头 1 个 block。
            Duplicate(dkRow_, 0.0f, dimPad_);
            {
                const uint8_t qRepStride = static_cast<uint8_t>(dimPad_ / FP32_PER_BLOCK);
                const uint8_t mRepStride = 1;
                for (uint32_t p = 0; p < dimPieces; ++p) {
                    BinaryRepeatParams dkParams{1, 1, 0, 0, qRepStride, mRepStride};
                    MulAddDst(dkRow_[p * 64u], qiF_[p * 64u], mblk_, MaskOf(dim_, p),
                              static_cast<uint8_t>(nidx_), dkParams);
                }
            }
            if (dkRows_ >= s2_) {
                Add(dkAcc_[j * dimPad_], dkAcc_[j * dimPad_], dkRow_, dimPad_);
            } else if (dkPartialElems_ > 0u) {
                const uint64_t offset = static_cast<uint64_t>(dkOffset_) + static_cast<uint64_t>(taskBase_) * dkPartialElems_ +
                                        static_cast<uint64_t>(j) * dimPad_;
                SyncVecToMte3();
                SyncVecDone();
                DataCopy(prod_, workspaceGm_[offset], dimPad_);
                SyncMte2ToVec();
                Add(dkRow_, dkRow_, prod_, dimPad_);
                SyncVecToMte3();
                DataCopy(workspaceGm_[offset], dkRow_, dimPad_);
                SyncMte3Done();
            }
        }
    }

    __aicore__ inline void StoreRowOut(uint32_t b, uint32_t row) {
        const uint64_t dqBase = (static_cast<uint64_t>(b) * s1_ + row) * nidx_ * dim_;
        const uint64_t dwBase = (static_cast<uint64_t>(b) * s1_ + row) * nidx_;
        SyncVecToMte3();
        if constexpr (sizeof(CT) == 4u) {
            DataCopyExtParams params{static_cast<uint16_t>(nidx_),
                                     static_cast<uint32_t>(dim_ * sizeof(T)),
                                     static_cast<uint32_t>((dimPad_ - dim_) * 4u), 0, 0};
            DataCopyPad(dQueryIndexGm_[dqBase], dq_, params);
        } else {
            // 低精度输出：按 STAGE 行分块 Cast，显著减少每 key 一次的 V/MTE3 同步
            for (uint32_t i0 = 0; i0 < nidx_; i0 += QI_STAGE_ROWS) {
                const uint32_t rows = MinU32(QI_STAGE_ROWS, nidx_ - i0);
                Cast(dqStage_, dq_[i0 * dimPad_], RoundMode::CAST_NONE, rows * dimPad_);
                SyncVecToMte3();
                DataCopyExtParams params{static_cast<uint16_t>(rows),
                                         static_cast<uint32_t>(dim_ * sizeof(T)),
                                         static_cast<uint32_t>((dimPad_ - dim_) * sizeof(T)), 0, 0};
                DataCopyPad(dQueryIndexGm_[dqBase + static_cast<uint64_t>(i0) * dim_],
                            dqStage_, params);
                SyncMte3Done();
            }
        }
        DataCopyExtParams wParams{1, static_cast<uint32_t>(nidx_ * sizeof(TW)), 0, 0, 0};
        if constexpr (sizeof(TW) == 4u) {
            DataCopyPad(dWeightsGm_[dwBase], dw_, wParams);
        } else {
            Cast(wRaw_, dw_, RoundMode::CAST_NONE, nidx_);
            SyncVecToMte3();
            DataCopyPad(dWeightsGm_[dwBase], wRaw_, wParams);
        }
    }

    __aicore__ inline void ProcessTask(uint32_t task) {
        taskBase_ = task;
        const uint32_t b = task / blocksPerBatch_;
        const uint32_t blk = task % blocksPerBatch_;
        const uint32_t rowStart = blk * rowsPerTask_;
        const uint32_t rowEnd = MinU32(s1_, rowStart + rowsPerTask_);
        rowLoss_ = 0.0f;
        Duplicate(lossAcc_, 0.0f, visPad_);
        if (coreNum_ > 1u) {
            SyncVecToMte3();
            DataCopy(workspaceGm_[lossOffset_ + static_cast<uint64_t>(task) * 8u], scratch_[384], 8);
            SyncMte3Done();
        }
        if (dkRows_ >= s2_) {
            Duplicate(dkAcc_, 0.0f, MinU32(dkRows_, s2_) * dimPad_);
        } else if (dkPartialElems_ > 0u) {
            Duplicate(dkRow_, 0.0f, dimPad_);
            SyncVecToMte3();
            for (uint32_t j = 0; j < s2_; ++j) {
                DataCopy(workspaceGm_[static_cast<uint64_t>(dkOffset_) + static_cast<uint64_t>(taskBase_) * dkPartialElems_ +
                                      static_cast<uint64_t>(j) * dimPad_],
                         dkRow_, dimPad_);
            }
            SyncMte3Done();
        }
        for (uint32_t row = rowStart; row < rowEnd; ++row) {
            ProcessRow(b, row, VisibleOf(row));
        }
        rowLoss_ = VecSum(lossAcc_, visPad_);
        if (coreNum_ == 1u) {
            totalLoss_ += rowLoss_;
        } else {
            scratch_[384].SetValue(0, rowLoss_);
            SetFlag<HardEvent::S_MTE3>(EV_S_MTE3);
            WaitFlag<HardEvent::S_MTE3>(EV_S_MTE3);
            DataCopy(workspaceGm_[lossOffset_ + static_cast<uint64_t>(task) * 8u], scratch_[384], 8);
            SyncMte3Done();
        }
        WriteDk(b);
    }

    __aicore__ inline void ProcessRow(uint32_t b, uint32_t row, uint32_t vis) {
        if (vis == 0u) {
            Duplicate(dq_, 0.0f, nidx_ * dimPad_);
            Duplicate(dw_, 0.0f, nidx_);
            StoreRowOut(b, row);
            return;
        }
        // 行前奏：等上一行的向量/MTE3 读完，一次性发起全部 GM->UB 搬运，只等待一次
        SyncVecDone();
        SyncMte3Done();
        IssueWeights(b, row);
        IssueRows(queryGm_, (static_cast<uint64_t>(b) * s1_ + row) * n1_ * dim_,
                  static_cast<uint64_t>(n1_) * dim_, qT_);
        const uint32_t firstRows = MinU32(kChunkRows_, vis);
        IssueRows(keyGm_, (static_cast<uint64_t>(b) * s2_) * n1_ * dim_,
                  static_cast<uint64_t>(firstRows) * n1_ * dim_, kT_);
        IssueKiAll(b);
        const bool qiFirst = (nidx_ <= QI_STAGE_ROWS);
        if (qiFirst) {
            IssueQiFirst(b, row);
        }
        SyncMte2ToVec();
        FinishWeights();
        // 主注意力只需要 q/k：先算分数，再做 qi/ki 的精度转换，缩短关键路径
        ComputeScores(b, vis);
        LoadQiAll(b, row, qiFirst);
        FinishKiAll();
        ComputeTarget(vis);
        ComputeSimilarity(b, vis);
        ComputeLogits(vis);
        SoftmaxAndLoss(vis);
        ComputeGradients(b, vis);
        StoreRowOut(b, row);

    }

    // dKeyIndex 写回：整块 Cast + 一次同步（原先每行一次 Cast/同步，是核内同步数最多的路径）。
    __aicore__ inline void WriteDk(uint32_t b) {
        if (dkRows_ < s2_) {
            return;
        }
        if (dkPartialElems_ > 0u) {
            SyncVecToMte3();
            for (uint32_t j = 0; j < s2_; ++j) {
                DataCopy(workspaceGm_[static_cast<uint64_t>(dkOffset_) +
                                      static_cast<uint64_t>(taskBase_) * dkPartialElems_ +
                                      static_cast<uint64_t>(j) * dimPad_],
                         dkAcc_[j * dimPad_], dimPad_);
            }
            SyncMte3Done();
            return;
        }
        const uint64_t offset = static_cast<uint64_t>(b) * s2_ * dim_;
        const uint64_t elems = static_cast<uint64_t>(s2_) * dim_;
        if constexpr (sizeof(CT) == 4u) {
            SyncVecToMte3();
            for (uint32_t j = 0; j < s2_; ++j) {
                DataCopyExtParams params{1, static_cast<uint32_t>(dim_ * sizeof(T)), 0, 0, 0};
                DataCopyPad(dKeyIndexGm_[offset + static_cast<uint64_t>(j) * dim_],
                            dkAcc_[j * dimPad_], params);
            }
            SyncMte3Done();
            return;
        }
        Cast(dkOutT_, dkAcc_, RoundMode::CAST_NONE, s2_ * dimPad_);
        SyncVecToMte3();
        if (BulkOk(offset, elems)) {
            DataCopy(dKeyIndexGm_[offset], dkOutT_, static_cast<uint32_t>(elems));
        } else {
            for (uint32_t j = 0; j < s2_; ++j) {
                DataCopyExtParams params{1, static_cast<uint32_t>(dim_ * sizeof(T)), 0, 0, 0};
                DataCopyPad(dKeyIndexGm_[offset + static_cast<uint64_t>(j) * dim_],
                            dkOutT_[j * dimPad_], params);
            }
        }
        SyncMte3Done();
    }

    __aicore__ inline void FinishLoss(uint32_t coreIdx) {
        if (coreNum_ == 1u) {
            lossGm_.SetValue(0, totalLoss_);
            return;
        }
        if (coreIdx == 0u) {
            float total = 0.0f;
            for (uint32_t t = 0; t < taskCount_; ++t) {
                SyncVecDone();
                DataCopy(prod_, workspaceGm_[lossOffset_ + static_cast<uint64_t>(t) * 8u], 8);
                SyncMte2ToVec();
                total += VecSum(prod_, 8);
            }
            lossGm_.SetValue(0, total);
        }
    }

    __aicore__ inline void ReduceDk(uint32_t coreIdx) {
        const uint64_t rowCount = static_cast<uint64_t>(batch_) * s2_;
        for (uint64_t index = coreIdx; index < rowCount; index += coreNum_) {
            const uint32_t b = static_cast<uint32_t>(index / s2_);
            const uint32_t j = static_cast<uint32_t>(index % s2_);
            Duplicate(scratch_, 0.0f, dimPad_);
            for (uint32_t blk = 0; blk < blocksPerBatch_; ++blk) {
                const uint32_t t = b * blocksPerBatch_ + blk;
                SyncVecDone();   // prod_ 可能仍在被上一轮 Add 读取
                DataCopy(prod_, workspaceGm_[static_cast<uint64_t>(dkOffset_) + static_cast<uint64_t>(t) * dkPartialElems_ +
                                             static_cast<uint64_t>(j) * dimPad_],
                         dimPad_);
                SyncMte2ToVec();
                Add(scratch_, scratch_, prod_, dimPad_);
            }
            const uint64_t offset =
                static_cast<uint64_t>(b) * s2_ * dim_ + static_cast<uint64_t>(j) * dim_;
            SyncVecToMte3();
            DataCopyExtParams params{1, static_cast<uint32_t>(dim_ * sizeof(T)), 0, 0, 0};
            if constexpr (sizeof(CT) == 4u) {
                DataCopyPad(dKeyIndexGm_[offset], scratch_, params);
            } else {
                Cast(outT_, scratch_, RoundMode::CAST_NONE, dimPad_);
                SyncVecToMte3();
                DataCopyPad(dKeyIndexGm_[offset], outT_, params);
                SyncMte3Done();
            }
        }
    }

private:
    uint32_t batch_ = 1, s1_ = 1, s2_ = 1, n1_ = 1, nidx_ = 1, dim_ = 1, dimPad_ = 1, visPad_ = 1;
    uint32_t n1Pad_ = 1, nidxPad_ = 1;
    uint32_t causal_ = 1, rowsPerTask_ = 1, blocksPerBatch_ = 1, taskCount_ = 1;
    uint32_t headBlock_ = 1, dkRows_ = 0, splitDk_ = 0;
    uint32_t dkPartialElems_ = 0, dkOffset_ = 0, lossOffset_ = 0, inputBytes_ = 2, weightBytes_ = 2;
    uint32_t stageElems_ = 0, kiCacheRows_ = 0, kChunkRows_ = 1;
    uint32_t coreNum_ = 1;
    float scale_ = 1.0f;
    float rowLoss_ = 0.0f;
    float totalLoss_ = 0.0f;
    uint32_t taskBase_ = 0;
    DliglUbLayout layout_;

    LocalTensor<CT> qT_, kT_, prodT_, kiT_;
    LocalTensor<T> qiStage_, dqStage_, kiCacheT_;
    LocalTensor<T> stageRaw_, outT_, dkOutT_;
    LocalTensor<float> prod_, qiF_, kiF_, kiCache_, sc_, u_, tgt_, sh_, pred_, del_, dkAcc_, dkRow_, dq_;
    LocalTensor<float> maxVec_, dblk_, mblk_, part_, w_, wb_, dw_, dwRow_, scratch_;
    LocalTensor<float> lossAcc_, lz_;
    LocalTensor<TW> wRaw_;

    GlobalTensor<T> queryGm_, keyGm_, queryIndexGm_, keyIndexGm_;
    GlobalTensor<T> dQueryIndexGm_, dKeyIndexGm_;
    GlobalTensor<TW> weightsGm_, dWeightsGm_;
    GlobalTensor<float> lossGm_, workspaceGm_;
};

template <typename DT_QUERY>
__global__ __aicore__ void dense_lightning_indexer_grad_kl_loss(
    GM_ADDR query, GM_ADDR key, GM_ADDR query_index, GM_ADDR key_index, GM_ADDR weights,
    GM_ADDR d_query_index, GM_ADDR d_key_index, GM_ADDR d_weights, GM_ADDR loss, GM_ADDR workspace,
    GM_ADDR tiling) {
    // 仅启动 Vector 核，避免为纯 Vector 算子额外拉起 Cube 核的头开销；
    // 多核路径使用 SyncAll（硬同步），因此必须用带硬同步的 AIV 1:0 类型。
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    REGISTER_TILING_DEFAULT(DenseLightningIndexerGradKlLossTilingData);
    GET_TILING_DATA_WITH_STRUCT(DenseLightningIndexerGradKlLossTilingData, tiling_data, tiling);
    if (tiling_data.weightsFp32 != 0u) {
        KernelDenseLightningIndexerGradKlLoss<DT_QUERY, float> op;
        op.Init(query, key, query_index, key_index, weights, d_query_index, d_key_index, d_weights,
                loss, workspace, tiling_data);
        op.Process();
    } else {
        KernelDenseLightningIndexerGradKlLoss<DT_QUERY, DT_QUERY> op;
        op.Init(query, key, query_index, key_index, weights, d_query_index, d_key_index, d_weights,
                loss, workspace, tiling_data);
        op.Process();
    }
}
