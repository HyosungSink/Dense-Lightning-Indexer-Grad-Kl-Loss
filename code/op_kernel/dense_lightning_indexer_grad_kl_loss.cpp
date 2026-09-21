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
        lossOffset_ = td.lossOffset;
        scale_ = td.scale;
        stageElems_ = td.stageElems;
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
        const uint64_t wsElems =
            static_cast<uint64_t>(dkPartialElems_) * taskCount_ + taskCount_ + 64u;
        workspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace), wsElems);

        layout_ = DliglComputeUbLayout(headBlock_, n1_, nidx_, dimPad_, visPad_, n1Pad_, nidxPad_,
                                       dkRows_, inputBytes_, weightBytes_, stageElems_);
        BuildTensors();
    }

    __aicore__ inline void Process() {
        const uint32_t coreIdx = static_cast<uint32_t>(GetBlockIdx());
        coreNum_ = static_cast<uint32_t>(GetBlockNum());
        totalLoss_ = 0.0f;
        Duplicate(scratch_[128], 1.0f, 64);
        PipeBarrier<PIPE_V>();
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
        kT_ = LocalTensor<CT>(TPosition::VECCALC, layout_.kTOff, n1_ * dimPad_);
        prodT_ = LocalTensor<CT>(TPosition::VECCALC, layout_.prodTOff, headBlock_ * dimPad_);
        prod_ = LocalTensor<float>(TPosition::VECCALC, layout_.prodOff, headBlock_ * dimPad_);
        outT_ = LocalTensor<T>(TPosition::VECCALC, layout_.outTOff, dimPad_);
        stageRaw_ = LocalTensor<T>(TPosition::VECCALC, layout_.stageOff, stageElems_);
        qiStage_ = LocalTensor<T>(TPosition::VECCALC, layout_.qiStageOff,
                                  QI_STAGE_ROWS * dimPad_);
        qiF_ = LocalTensor<float>(TPosition::VECCALC, layout_.qiFOff, nidx_ * dimPad_);
        kiT_ = LocalTensor<CT>(TPosition::VECCALC, layout_.kiTOff, dimPad_);
        kiF_ = LocalTensor<float>(TPosition::VECCALC, layout_.kiFOff, dimPad_);
        sc_ = LocalTensor<float>(TPosition::VECCALC, layout_.scOff, visPad_ * n1Pad_);
        u_ = LocalTensor<float>(TPosition::VECCALC, layout_.uOff, visPad_ * nidxPad_);
        tgt_ = LocalTensor<float>(TPosition::VECCALC, layout_.tgtOff, visPad_);
        sh_ = LocalTensor<float>(TPosition::VECCALC, layout_.shOff, visPad_);
        pred_ = LocalTensor<float>(TPosition::VECCALC, layout_.predOff, visPad_);
        del_ = LocalTensor<float>(TPosition::VECCALC, layout_.delOff, visPad_);
        dkAcc_ = LocalTensor<float>(TPosition::VECCALC, layout_.dkOff, dkRows_ * dimPad_);
        dkRow_ = LocalTensor<float>(TPosition::VECCALC, layout_.dkRowOff, dimPad_);
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
            PipeBarrier<PIPE_V>();
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
            PipeBarrier<PIPE_V>();
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
    __aicore__ inline void LoadRows(const GlobalTensor<T> &src, uint64_t base, uint32_t rows,
                                    LocalTensor<CT> dst) {
        // 覆盖目的缓冲前，先等上一轮矢量读和 MTE3 读结束
        SyncVecDone();
        SyncMte3Done();
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

    __aicore__ inline void LoadQiAll(uint32_t b, uint32_t row) {
        const uint64_t base = (static_cast<uint64_t>(b) * s1_ + row) * nidx_ * dim_;
        if constexpr (sizeof(T) == 4u) {
            LoadRows(queryIndexGm_, base, nidx_, qiF_);
        } else {
            // 分块载入并转 fp32，避免常驻一块 (nidx, dimPad) 的低精度缓冲
            for (uint32_t i0 = 0; i0 < nidx_; i0 += QI_STAGE_ROWS) {
                const uint32_t rows = MinU32(QI_STAGE_ROWS, nidx_ - i0);
                SyncVecDone();
                SyncMte3Done();
                DataCopyExtParams params{static_cast<uint16_t>(rows),
                                         static_cast<uint32_t>(dim_ * inputBytes_), 0, 0, 0};
                DataCopyPad(qiStage_, queryIndexGm_[base + static_cast<uint64_t>(i0) * dim_], params,
                            DataCopyPadExtParams<T>{false, 0, 0, 0});
                SyncMte2ToVec();
                Cast(qiF_[i0 * dimPad_], qiStage_, RoundMode::CAST_NONE, rows * dimPad_);
                PipeBarrier<PIPE_V>();
            }
        }
    }

    __aicore__ inline void LoadKiRow(uint32_t b, uint32_t j) {
        const uint64_t base = (static_cast<uint64_t>(b) * s2_ + j) * dim_;
        LoadRows(keyIndexGm_, base, 1u, kiT_);
    }

    __aicore__ inline void LoadWeights(uint32_t b, uint32_t row) {
        const uint64_t base = (static_cast<uint64_t>(b) * s1_ + row) * nidx_;
        DataCopyExtParams params{1, static_cast<uint32_t>(nidx_ * weightBytes_), 0, 0, 0};
        DataCopyPad(wRaw_, weightsGm_[base], params, DataCopyPadExtParams<TW>{false, 0, 0, 0});
        SyncMte2ToVec();
        if constexpr (sizeof(TW) != 4u) {
            Cast(w_, wRaw_, RoundMode::CAST_NONE, nidx_);
        } else {
            Adds(w_, wRaw_, 0.0f, nidx_);
        }
        PipeBarrier<PIPE_V>();
        Brcb(wb_, w_, static_cast<uint8_t>((nidx_ + 7u) / 8u), {1, 8});
        PipeBarrier<PIPE_V>();
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
            PipeBarrier<PIPE_V>();
            for (uint32_t p = 1; p < pieces; ++p) {
                Add(part_, part_, part_[p * rows], rows);
            }
        }
        PipeBarrier<PIPE_V>();
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
    __aicore__ inline void ComputeScoreBlock(uint32_t j, uint32_t h0, uint32_t hc) {
        constexpr uint32_t LANES = 256u / sizeof(CT);
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
                Mul(prod_[p * LANES], kT_[h0 * dimPad_ + p * LANES], qT_[h0 * dimPad_ + p * LANES],
                    mask, static_cast<uint8_t>(hc), params);
            } else {
                BinaryRepeatParams params{1, 1, 1, tStride, tStride, tStride};
                Mul(prodT_[p * LANES], kT_[h0 * dimPad_ + p * LANES],
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
            PipeBarrier<PIPE_V>();
            Cast(prod_, prodT_, RoundMode::CAST_NONE, hc * dimPad_);
            PipeBarrier<PIPE_V>();
            ReduceRowsFp32(sc_[j * n1Pad_ + h0], prod_, hc, redWidth, dimPad_);
        }
    }

    __aicore__ inline void ComputeScores(uint32_t b, uint32_t vis) {
        for (uint32_t j = 0; j < vis; ++j) {
            LoadKRow(b, j);
            for (uint32_t h0 = 0; h0 < n1_; h0 += headBlock_) {
                const uint32_t hc = MinU32(headBlock_, n1_ - h0);
                ComputeScoreBlock(j, h0, hc);
                PipeBarrier<PIPE_V>();
            }
        }
        Muls(sc_, sc_, scale_, visPad_ * n1Pad_);
        PipeBarrier<PIPE_V>();
    }

    // 每头最大值（跨 key）-> target_j = sum_h exp(s_hj - m_h)，最后统一 L1 归一化
    __aicore__ inline void ComputeTarget(uint32_t vis) {
        Duplicate(maxVec_, -3.0e38f, n1Pad_);
        PipeBarrier<PIPE_V>();
        const uint32_t nPieces = PiecesOf(n1_);
        for (uint32_t j = 0; j < vis; ++j) {
            for (uint32_t p = 0; p < nPieces; ++p) {
                Max(maxVec_[p * 64u], maxVec_[p * 64u], sc_[j * n1Pad_ + p * 64u], MaskOf(n1_, p));
            }
        }
        PipeBarrier<PIPE_V>();
        for (uint32_t j = 0; j < vis; ++j) {
            for (uint32_t p = 0; p < nPieces; ++p) {
                Sub(sc_[j * n1Pad_ + p * 64u], sc_[j * n1Pad_ + p * 64u], maxVec_[p * 64u],
                    MaskOf(n1_, p));
            }
            PipeBarrier<PIPE_V>();
            Exp(sc_[j * n1Pad_], sc_[j * n1Pad_], n1_);
            PipeBarrier<PIPE_V>();
            if (nPieces == 1u) {
                WholeReduceSum<float>(tgt_[j], sc_[j * n1Pad_], MaskOf(n1_, 0), 1, 1, 1, 1);
                continue;
            }
            {
                for (uint32_t p = 0; p < nPieces; ++p) {
                    WholeReduceSum<float>(part_[p * 8], sc_[j * n1Pad_ + p * 64u], MaskOf(n1_, p), 1,
                                          1, 1, 1);
                }
                PipeBarrier<PIPE_V>();
                WholeReduceSum<float>(part_, part_, nPieces, 1, 1, 1, 1);
                PipeBarrier<PIPE_V>();
                tgt_.SetValue(j, ReadScalar(part_));
                SetFlag<HardEvent::S_V>(EV_S_V);
                WaitFlag<HardEvent::S_V>(EV_S_V);
            }
        }
        PipeBarrier<PIPE_V>();
        const float total = VecSum(tgt_, vis);
        Muls(tgt_, tgt_, 1.0f / total, vis);
        PipeBarrier<PIPE_V>();
    }

    // ------------------------------------------------------------- indexer 相似度
    // U[j][i] = relu(qi[i] · ki[j])：qi 常驻 fp32，ki 逐 key 转 fp32
    __aicore__ inline void ComputeSimilarity(uint32_t b, uint32_t vis) {
        const uint32_t pieces = (dim_ + FP32_PER_REPEAT - 1u) / FP32_PER_REPEAT;
        const uint32_t redWidth =
            (pieces > 1u) ? (dim_ < FP32_PER_REPEAT ? dim_ : FP32_PER_REPEAT) : dim_;
        const uint8_t fStride = static_cast<uint8_t>(dimPad_ / FP32_PER_BLOCK);
        for (uint32_t j = 0; j < vis; ++j) {
            LoadKiRow(b, j);
            if constexpr (sizeof(T) == 4u) {
                Adds(kiF_, kiT_, 0.0f, dimPad_);
            } else {
                SyncVecDone();
                Cast(kiF_, kiT_, RoundMode::CAST_NONE, dimPad_);
            }
            PipeBarrier<PIPE_V>();
            for (uint32_t p = 0; p < pieces; ++p) {
                const uint32_t left = dim_ - p * FP32_PER_REPEAT;
                const uint32_t mask = left > FP32_PER_REPEAT ? FP32_PER_REPEAT : left;
                BinaryRepeatParams params{1, 1, 1, fStride, fStride, 0};
                Mul(prod_[p * FP32_PER_REPEAT], qiF_[p * FP32_PER_REPEAT],
                    kiF_[p * FP32_PER_REPEAT], mask, static_cast<uint8_t>(nidx_), params);
            }
            if (pieces > 1u) {
                BinaryRepeatParams addParams{1, 1, 1, fStride, fStride, fStride};
                for (uint32_t p = 1; p < pieces; ++p) {
                    Add(prod_, prod_, prod_[p * FP32_PER_REPEAT], FP32_PER_REPEAT,
                        static_cast<uint8_t>(nidx_), addParams);
                }
            }
            PipeBarrier<PIPE_V>();
            ReduceRowsFp32(u_[j * nidxPad_], prod_, nidx_, redWidth, dimPad_);
            PipeBarrier<PIPE_V>();
            Maxs(u_[j * nidxPad_], u_[j * nidxPad_], 0.0f, nidx_);  // ReLU
        }
        PipeBarrier<PIPE_V>();
    }

    // logits_j = sum_i w_i * relu(u_ij)
    __aicore__ inline void ComputeLogits(uint32_t vis) {
        const uint32_t pieces = PiecesOf(nidx_);
        for (uint32_t j = 0; j < vis; ++j) {
            for (uint32_t p = 0; p < pieces; ++p) {
                Mul(dwRow_[p * 64u], u_[j * nidxPad_ + p * 64u], w_[p * 64u], MaskOf(nidx_, p));
            }
            PipeBarrier<PIPE_V>();
            if (pieces == 1u) {
                WholeReduceSum<float>(sh_[j], dwRow_, MaskOf(nidx_, 0), 1, 1, 1, 1);
            } else {
                WholeReduceSum<float>(part_, dwRow_, MaskOf(nidx_, 0), 1, 1, 1, 1);
                for (uint32_t p = 1; p < pieces; ++p) {
                    WholeReduceSum<float>(part_[p * 8], dwRow_[p * 64u], MaskOf(nidx_, p), 1, 1, 1,
                                          1);
                    PipeBarrier<PIPE_V>();
                    Add(part_, part_, part_[p * 8], 8);
                }
                PipeBarrier<PIPE_V>();
                Adds(sh_[j], part_, 0.0f, 1);
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void SoftmaxAndLoss(uint32_t vis) {
        const float logitMax = VecMax(sh_, vis);
        Adds(sh_, sh_, -logitMax, vis);
        PipeBarrier<PIPE_V>();
        Exp(pred_, sh_, vis);
        PipeBarrier<PIPE_V>();
        const float normalizer = VecSum(pred_, vis);
        Muls(pred_, pred_, 1.0f / normalizer, vis);
        PipeBarrier<PIPE_V>();
        Sub(del_, pred_, tgt_, vis);
        PipeBarrier<PIPE_V>();
        // loss 项：sum p*(ln p - shifted) + ln Z
        Maxs(scratch_, tgt_, 1.0e-30f, vis);
        PipeBarrier<PIPE_V>();
        Ln(scratch_, scratch_, vis);
        PipeBarrier<PIPE_V>();
        Mul(scratch_, scratch_, tgt_, vis);
        PipeBarrier<PIPE_V>();
        Mul(scratch_[64], tgt_, sh_, vis);
        PipeBarrier<PIPE_V>();
        Sub(scratch_, scratch_, scratch_[64], vis);
        PipeBarrier<PIPE_V>();
        rowLoss_ += VecSum(scratch_, vis) + ScalarLn(normalizer);
    }

    // ------------------------------------------------------------- 梯度
    __aicore__ inline void ComputeGradients(uint32_t b, uint32_t vis) {
        Duplicate(dq_, 0.0f, nidx_ * dimPad_);
        Duplicate(dw_, 0.0f, nidx_);
        PipeBarrier<PIPE_V>();
        Brcb(dblk_, del_, static_cast<uint8_t>((vis + 7u) / 8u), {1, 8});
        PipeBarrier<PIPE_V>();
        const uint32_t dimPieces = PiecesOf(dim_);
        const uint32_t nidxPieces = PiecesOf(nidx_);
        const uint8_t dimStride = static_cast<uint8_t>(dimPad_ / FP32_PER_BLOCK);
        for (uint32_t j = 0; j < vis; ++j) {
            LoadKiRow(b, j);
            if constexpr (sizeof(CT) == 4u) {
                Adds(kiF_, kiT_, 0.0f, dimPad_);
            } else {
                SyncVecDone();
                Cast(kiF_, kiT_, RoundMode::CAST_NONE, dimPad_);
            }
            PipeBarrier<PIPE_V>();
            // m = w * delta_j * (u > 0)，u 已含 ReLU
            Muls(dwRow_, u_[j * nidxPad_], INDICATOR_SCALE, nidx_);
            PipeBarrier<PIPE_V>();
            Mins(dwRow_, dwRow_, 1.0f, nidx_);
            PipeBarrier<PIPE_V>();
            for (uint32_t p = 0; p < nidxPieces; ++p) {
                Mul(dwRow_[p * 64u], dwRow_[p * 64u], dblk_[j * 8], MaskOf(nidx_, p),
                    static_cast<uint8_t>(1), {1, 1, 0, 1, 1, 1});
            }
            PipeBarrier<PIPE_V>();
            Mul(dwRow_, dwRow_, w_, nidx_);
            PipeBarrier<PIPE_V>();
            // dw += relu(u_j) * delta_j
            for (uint32_t p = 0; p < nidxPieces; ++p) {
                MulAddDst(dw_[p * 64u], u_[j * nidxPad_ + p * 64u], dblk_[j * 8], MaskOf(nidx_, p),
                          static_cast<uint8_t>(1), {1, 1, 0, 1, 1, 1});
            }
            PipeBarrier<PIPE_V>();
            Brcb(mblk_, dwRow_, static_cast<uint8_t>((nidx_ + 7u) / 8u), {1, 8});
            PipeBarrier<PIPE_V>();
            // dq += ki_j ⊗ m_j
            for (uint32_t p = 0; p < dimPieces; ++p) {
                BinaryRepeatParams dqParams{1, 1, 0, dimStride, 0, 1};
                MulAddDst(dq_[p * 64u], kiF_[p * 64u], mblk_, MaskOf(dim_, p),
                          static_cast<uint8_t>(nidx_), dqParams);
            }
            PipeBarrier<PIPE_V>();
            // dk[j] += sum_i qi[i] * m_ij
            Duplicate(dkRow_, 0.0f, dimPad_);
            PipeBarrier<PIPE_V>();
            for (uint32_t p = 0; p < dimPieces; ++p) {
                Duplicate(scratch_, 0.0f, 64);
                PipeBarrier<PIPE_V>();
                BinaryRepeatParams dkParams{1, 1, 0, 0, dimStride, 1};
                MulAddDst(scratch_, qiF_[p * 64u], mblk_, MaskOf(dim_, p),
                          static_cast<uint8_t>(nidx_), dkParams);
                PipeBarrier<PIPE_V>();
                Add(dkRow_[p * 64u], dkRow_[p * 64u], scratch_, MaskOf(dim_, p));
            }
            PipeBarrier<PIPE_V>();
            if (dkRows_ >= s2_) {
                Add(dkAcc_[j * dimPad_], dkAcc_[j * dimPad_], dkRow_, dimPad_);
                PipeBarrier<PIPE_V>();
            } else if (dkPartialElems_ > 0u) {
                const uint64_t offset = static_cast<uint64_t>(taskBase_) * dkPartialElems_ +
                                        static_cast<uint64_t>(j) * dimPad_;
                SyncVecToMte3();
                SyncVecDone();
                DataCopy(prod_, workspaceGm_[offset], dimPad_);
                SyncMte2ToVec();
                Add(dkRow_, dkRow_, prod_, dimPad_);
                PipeBarrier<PIPE_V>();
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
        for (uint32_t i = 0; i < nidx_; ++i) {
            const uint64_t offset = dqBase + static_cast<uint64_t>(i) * dim_;
            DataCopyExtParams params{1, static_cast<uint32_t>(dim_ * sizeof(T)), 0, 0, 0};
            if constexpr (sizeof(CT) == 4u) {
                DataCopyPad(dQueryIndexGm_[offset], dq_[i * dimPad_], params);
            } else {
                Cast(outT_, dq_[i * dimPad_], RoundMode::CAST_NONE, dimPad_);
                SyncVecToMte3();
                DataCopyPad(dQueryIndexGm_[offset], outT_, params);
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
            SyncMte3Done();
        }
    }

    __aicore__ inline void ProcessTask(uint32_t task) {
        taskBase_ = task;
        const uint32_t b = task / blocksPerBatch_;
        const uint32_t blk = task % blocksPerBatch_;
        const uint32_t rowStart = blk * rowsPerTask_;
        const uint32_t rowEnd = MinU32(s1_, rowStart + rowsPerTask_);
        rowLoss_ = 0.0f;
        if (coreNum_ > 1u) {
            Duplicate(scratch_[384], 0.0f, 8);
            PipeBarrier<PIPE_V>();
            SyncVecToMte3();
            DataCopy(workspaceGm_[lossOffset_ + task], scratch_[384], 8);
            SyncMte3Done();
        }
        if (dkRows_ >= s2_) {
            Duplicate(dkAcc_, 0.0f, MinU32(dkRows_, s2_) * dimPad_);
            PipeBarrier<PIPE_V>();
        } else if (dkPartialElems_ > 0u) {
            Duplicate(dkRow_, 0.0f, dimPad_);
            PipeBarrier<PIPE_V>();
            SyncVecToMte3();
            for (uint32_t j = 0; j < s2_; ++j) {
                DataCopy(workspaceGm_[static_cast<uint64_t>(taskBase_) * dkPartialElems_ +
                                      static_cast<uint64_t>(j) * dimPad_],
                         dkRow_, dimPad_);
            }
            SyncMte3Done();
        }
        for (uint32_t row = rowStart; row < rowEnd; ++row) {
            ProcessRow(b, row, VisibleOf(row));
        }
        if (coreNum_ == 1u) {
            totalLoss_ += rowLoss_;
        } else {
            scratch_[384].SetValue(0, rowLoss_);
            SetFlag<HardEvent::S_MTE3>(EV_S_MTE3);
            WaitFlag<HardEvent::S_MTE3>(EV_S_MTE3);
            DataCopy(workspaceGm_[lossOffset_ + task], scratch_[384], 8);
            SyncMte3Done();
        }
        WriteDk(b);
    }

    __aicore__ inline void ProcessRow(uint32_t b, uint32_t row, uint32_t vis) {
        if (vis == 0u) {
            Duplicate(dq_, 0.0f, nidx_ * dimPad_);
            Duplicate(dw_, 0.0f, nidx_);
            PipeBarrier<PIPE_V>();
            StoreRowOut(b, row);
            return;
        }
        LoadWeights(b, row);
        LoadQRow(b, row);
        LoadQiAll(b, row);
        ComputeScores(b, vis);
        ComputeTarget(vis);
        ComputeSimilarity(b, vis);
        ComputeLogits(vis);
        SoftmaxAndLoss(vis);
        ComputeGradients(b, vis);
        StoreRowOut(b, row);

    }

    __aicore__ inline void WriteDk(uint32_t b) {
        if (dkRows_ < s2_) {
            return;
        }
        SyncVecToMte3();
        for (uint32_t j = 0; j < s2_; ++j) {
            if (dkPartialElems_ > 0u) {
                DataCopy(workspaceGm_[static_cast<uint64_t>(taskBase_) * dkPartialElems_ +
                                      static_cast<uint64_t>(j) * dimPad_],
                         dkAcc_[j * dimPad_], dimPad_);
            } else {
                const uint64_t offset =
                    static_cast<uint64_t>(b) * s2_ * dim_ + static_cast<uint64_t>(j) * dim_;
                DataCopyExtParams params{1, static_cast<uint32_t>(dim_ * sizeof(T)), 0, 0, 0};
                if constexpr (sizeof(CT) == 4u) {
                    DataCopyPad(dKeyIndexGm_[offset], dkAcc_[j * dimPad_], params);
                } else {
                    Cast(outT_, dkAcc_[j * dimPad_], RoundMode::CAST_NONE, dimPad_);
                    SyncVecToMte3();
                    DataCopyPad(dKeyIndexGm_[offset], outT_, params);
                    SyncMte3Done();
                }
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
            for (uint32_t t = 0; t < taskCount_; t += 64u) {
                const uint32_t count = MinU32(64u, taskCount_ - t);
                DataCopy(prod_, workspaceGm_[lossOffset_ + t], Align8(count));
                SyncMte2ToVec();
                total += VecSum(prod_, count);
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
            PipeBarrier<PIPE_V>();
            for (uint32_t blk = 0; blk < blocksPerBatch_; ++blk) {
                const uint32_t t = b * blocksPerBatch_ + blk;
                DataCopy(prod_, workspaceGm_[static_cast<uint64_t>(t) * dkPartialElems_ +
                                             static_cast<uint64_t>(j) * dimPad_],
                         dimPad_);
                SyncMte2ToVec();
                Add(scratch_, scratch_, prod_, dimPad_);
                PipeBarrier<PIPE_V>();
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
    uint32_t dkPartialElems_ = 0, lossOffset_ = 0, inputBytes_ = 2, weightBytes_ = 2;
    uint32_t stageElems_ = 0;
    uint32_t coreNum_ = 1;
    float scale_ = 1.0f;
    float rowLoss_ = 0.0f;
    float totalLoss_ = 0.0f;
    uint32_t taskBase_ = 0;
    DliglUbLayout layout_;

    LocalTensor<CT> qT_, kT_, prodT_, kiT_;
    LocalTensor<T> qiStage_;
    LocalTensor<T> stageRaw_, outT_;
    LocalTensor<float> prod_, qiF_, kiF_, sc_, u_, tgt_, sh_, pred_, del_, dkAcc_, dkRow_, dq_;
    LocalTensor<float> maxVec_, dblk_, mblk_, part_, w_, wb_, dw_, dwRow_, scratch_;
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
