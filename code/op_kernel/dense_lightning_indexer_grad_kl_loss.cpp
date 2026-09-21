// Kernel侧核函数实现：DenseLightningIndexerGradKLLoss
//
// 每行独立计算（dKeyIndex 需要在行间累加）：
//   1) 主注意力：逐头点积 -> 每头 softmax（仅减最大值）-> 各头求和 -> L1 归一化得到 target p
//   2) indexer：逐头点积得到相似度 S，ReLU 后按 weights 加权得到 logits，softmax 得到预测分布
//   3) loss 与梯度：dI = pred - p，ds = w*dI*(S>0)，dq = ds@K̃，dk = ds^T@Q̃，dW = ReLU(S)@dI
#include "kernel_operator.h"

#include "dense_lightning_indexer_grad_kl_loss_tiling.h"
#include "tiling_key_dense_lightning_indexer_grad_kl_loss.h"

using namespace AscendC;

#define DLIGL_DEBUG 0

namespace {
constexpr uint32_t FLOATS_PER_BLOCK = 8;    // 32B / 4B
constexpr float INDICATOR_SCALE = 1.0e30f;  // ReLU 正值放大到 >=1 后截断为 1
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
        causal_ = td.causal;
        rowsPerTask_ = td.rowsPerTask;
        blocksPerBatch_ = td.blocksPerBatch;
        taskCount_ = td.taskCount;
        headBlock_ = td.headBlock;
        keyChunk_ = td.keyChunk;
        simPitch_ = td.simPitch;
        dkRows_ = td.dkRows;
        storeSim_ = td.storeSim;
        splitDk_ = td.splitDk;
        dkPartialElems_ = td.dkPartialElems;
        lossOffset_ = td.lossOffset;
        scale_ = td.scale;
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

        layout_ = DliglComputeUbLayout(headBlock_, keyChunk_, nidx_, dimPad_, visPad_,
                                       storeSim_ != 0u ? visPad_ : simPitch_, dkRows_, inputBytes_,
                                       weightBytes_);
        BuildTensors();
    }

    __aicore__ inline void Process() {
        const uint32_t coreIdx = static_cast<uint32_t>(GetBlockIdx());
        coreNum_ = static_cast<uint32_t>(GetBlockNum());
        totalLoss_ = 0.0f;
        Duplicate(scratch_[128], 1.0f, 64);  // ones
        Duplicate(scratch_[192], 0.0f, 64);  // zeros
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
        q_ = LocalTensor<float>(TPosition::VECCALC, layout_.qOff, headBlock_ * dimPad_);
        kT_ = LocalTensor<T>(TPosition::VECCALC, layout_.kTOff, keyChunk_ * dimPad_);
        kF_ = LocalTensor<float>(TPosition::VECCALC, layout_.kFOff, keyChunk_ * dimPad_);
        prod_ = LocalTensor<float>(TPosition::VECCALC, layout_.prodOff, keyChunk_ * dimPad_);
        score_ = LocalTensor<float>(TPosition::VECCALC, layout_.scoreOff, headBlock_ * visPad_);
        sim_ = LocalTensor<float>(TPosition::VECCALC, layout_.simOff,
                                  nidx_ * (storeSim_ != 0u ? visPad_ : simPitch_));
        tgt_ = LocalTensor<float>(TPosition::VECCALC, layout_.tgtOff, visPad_);
        sh_ = LocalTensor<float>(TPosition::VECCALC, layout_.shOff, visPad_);
        pred_ = LocalTensor<float>(TPosition::VECCALC, layout_.predOff, visPad_);
        del_ = LocalTensor<float>(TPosition::VECCALC, layout_.delOff, visPad_);
        dkAcc_ = LocalTensor<float>(TPosition::VECCALC, layout_.dkOff, dkRows_ * dimPad_);
        dkChunk_ = LocalTensor<float>(TPosition::VECCALC, layout_.dkcOff, keyChunk_ * dimPad_);
        qi_ = LocalTensor<float>(TPosition::VECCALC, layout_.qiOff, dimPad_);
        qiTTmp_ = LocalTensor<T>(TPosition::VECCALC, layout_.qiTOff, dimPad_);
        dq_ = LocalTensor<float>(TPosition::VECCALC, layout_.dqOff, nidx_ * dimPad_);
        part_ = LocalTensor<float>(TPosition::VECCALC, layout_.partOff, 64u * 8u);
        bcast_ = LocalTensor<float>(TPosition::VECCALC, layout_.bcastOff,
                                    (headBlock_ > keyChunk_ ? headBlock_ : keyChunk_) * 8u);
        max_ = LocalTensor<float>(TPosition::VECCALC, layout_.maxOff, 64u * 8u);
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

    __aicore__ inline void SyncMte2ToVec() {
        SetFlag<HardEvent::MTE2_V>(EV_MTE2_V);
        WaitFlag<HardEvent::MTE2_V>(EV_MTE2_V);
    }
    __aicore__ inline void SyncVecToMte3() {
        SetFlag<HardEvent::V_MTE3>(EV_V_MTE3);
        WaitFlag<HardEvent::V_MTE3>(EV_V_MTE3);
    }
    // 等待 MTE3（UB->GM）全部完成，之后才能覆盖其源缓冲区
    __aicore__ inline void SyncMte3Done() {
        SetFlag<HardEvent::MTE3_V>(EV_MTE3_V);
        WaitFlag<HardEvent::MTE3_V>(EV_MTE3_V);
    }
    // 等待矢量流水完成，之后 MTE2 才能覆盖矢量读过的缓冲区
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

    // dst[r] = sum_d a[r][d] * b[d]，b 通过 src1RepStride=0 广播到所有行
    __aicore__ inline void DotRows(const LocalTensor<float> &dst, const LocalTensor<float> &a,
                                   const LocalTensor<float> &b, uint32_t rows, uint32_t width,
                                   uint32_t pitch) {
        const uint32_t pieces = PiecesOf(width);
        const uint8_t pitchStride = static_cast<uint8_t>(pitch / FLOATS_PER_BLOCK);
        for (uint32_t p = 0; p < pieces; ++p) {
            BinaryRepeatParams params{1, 1, 1, pitchStride, pitchStride, 0};
            Mul(prod_[p * 64u], a[p * 64u], b[p * 64u], MaskOf(width, p),
                static_cast<uint8_t>(rows), params);
        }
        if (pieces > 1u) {
            BinaryRepeatParams addParams{1, 1, 1, pitchStride, pitchStride, pitchStride};
            for (uint32_t p = 1; p < pieces; ++p) {
                Add(prod_, prod_, prod_[p * 64u], MaskOf(width, p), static_cast<uint8_t>(rows),
                    addParams);
            }
        }
        WholeReduceSum<float>(dst, prod_, MaskOf(width, 0), static_cast<int32_t>(rows), 1, 1,
                              static_cast<int32_t>(pitch / FLOATS_PER_BLOCK));
    }

    __aicore__ inline void LoadKeyRows(uint32_t b, uint32_t k0, uint32_t kcCur, uint32_t head) {
        SyncMte3Done();
        SyncVecDone();
        const uint64_t base = (static_cast<uint64_t>(b) * s2_ + k0) * n1_ * dim_ +
                              static_cast<uint64_t>(head) * dim_;
        DataCopyExtParams params{static_cast<uint16_t>(kcCur),
                                 static_cast<uint32_t>(dim_ * inputBytes_),
                                 static_cast<uint32_t>((n1_ - 1u) * dim_ * inputBytes_), 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        if constexpr (sizeof(T) == 4u) {
            DataCopyPad(kF_, keyGm_[base], params, pad);
            SyncMte2ToVec();
        } else {
            DataCopyPad(kT_, keyGm_[base], params, pad);
            SyncMte2ToVec();
            Cast(kF_, kT_, RoundMode::CAST_NONE, kcCur * dimPad_);
        }
    }

    __aicore__ inline void LoadKiRows(uint32_t b, uint32_t k0, uint32_t kcCur) {
        SyncMte3Done();
        SyncVecDone();
        const uint64_t base = (static_cast<uint64_t>(b) * s2_ + k0) * dim_;
        DataCopyExtParams params{static_cast<uint16_t>(kcCur),
                                 static_cast<uint32_t>(dim_ * inputBytes_), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        if constexpr (sizeof(T) == 4u) {
            DataCopyPad(kF_, keyIndexGm_[base], params, pad);
            SyncMte2ToVec();
        } else {
            DataCopyPad(kT_, keyIndexGm_[base], params, pad);
            SyncMte2ToVec();
            Cast(kF_, kT_, RoundMode::CAST_NONE, kcCur * dimPad_);
        }
    }

    __aicore__ inline void LoadQBlock(uint32_t b, uint32_t row, uint32_t h0, uint32_t hc) {
        SyncMte3Done();
        SyncVecDone();
        const uint64_t base = (static_cast<uint64_t>(b) * s1_ + row) * n1_ * dim_ +
                              static_cast<uint64_t>(h0) * dim_;
        DataCopyExtParams params{static_cast<uint16_t>(hc),
                                 static_cast<uint32_t>(dim_ * inputBytes_), 0, 0, 0};
        if constexpr (sizeof(T) == 4u) {
            DataCopyPad(q_, queryGm_[base], params, DataCopyPadExtParams<T>{false, 0, 0, 0});
            SyncMte2ToVec();
        } else {
            DataCopyPad(kT_, queryGm_[base], params, DataCopyPadExtParams<T>{false, 0, 0, 0});
            SyncMte2ToVec();
            Cast(q_, kT_, RoundMode::CAST_NONE, hc * dimPad_);
        }
        // 注意力缩放：直接把 scale 乘到 query 上，主注意力 score = scale * (q·k)
        Muls(q_, q_, scale_, hc * dimPad_);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void LoadQiRow(uint32_t b, uint32_t row, uint32_t head) {
        SyncMte3Done();
        SyncVecDone();
        const uint64_t base = ((static_cast<uint64_t>(b) * s1_ + row) * nidx_ + head) * dim_;
        DataCopyExtParams params{1, static_cast<uint32_t>(dim_ * inputBytes_), 0, 0, 0};
        if constexpr (sizeof(T) == 4u) {
            DataCopyPad(qi_, queryIndexGm_[base], params, DataCopyPadExtParams<T>{false, 0, 0, 0});
            SyncMte2ToVec();
        } else {
            DataCopyPad(qiTTmp_, queryIndexGm_[base], params, DataCopyPadExtParams<T>{false, 0, 0, 0});
            SyncMte2ToVec();
            Cast(qi_, qiTTmp_, RoundMode::CAST_NONE, dimPad_);
        }
    }

    __aicore__ inline void LoadWeights(uint32_t b, uint32_t row) {
        SyncMte3Done();
        SyncVecDone();
        const uint64_t base = (static_cast<uint64_t>(b) * s1_ + row) * nidx_;
        DataCopyExtParams wParams{1, static_cast<uint32_t>(nidx_ * weightBytes_), 0, 0, 0};
        DataCopyPad(wRaw_, weightsGm_[base], wParams, DataCopyPadExtParams<TW>{false, 0, 0, 0});
        SyncMte2ToVec();
        if constexpr (sizeof(TW) != 4u) {
            Cast(w_, wRaw_, RoundMode::CAST_NONE, nidx_);
            PipeBarrier<PIPE_V>();
        } else {
            Adds(w_, wRaw_, 0.0f, nidx_);
            PipeBarrier<PIPE_V>();
        }
        Brcb(wb_, w_, static_cast<uint8_t>((nidx_ + 7u) / 8u), {1, 8});
        PipeBarrier<PIPE_V>();
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

    // ------------------------------------------------------------- 主注意力 -> target
    __aicore__ inline void ComputeTarget(uint32_t b, uint32_t row, uint32_t vis) {
        Duplicate(tgt_, 0.0f, visPad_);
        PipeBarrier<PIPE_V>();
        for (uint32_t h0 = 0; h0 < n1_; h0 += headBlock_) {
            const uint32_t hc = MinU32(headBlock_, n1_ - h0);
            LoadQBlock(b, row, h0, hc);
            Duplicate(score_, 0.0f, hc * visPad_);
            PipeBarrier<PIPE_V>();
            for (uint32_t h = 0; h < hc; ++h) {
                for (uint32_t k0 = 0; k0 < vis; k0 += keyChunk_) {
                    const uint32_t kcCur = MinU32(keyChunk_, vis - k0);
                    LoadKeyRows(b, k0, kcCur, h0 + h);
                    DotRows(score_[h * visPad_ + k0], kF_, q_[h * dimPad_], kcCur, dim_, dimPad_);
                }
            }
            PipeBarrier<PIPE_V>();
            WholeReduceMax<float>(max_, score_, MaskOf(vis, 0), static_cast<int32_t>(hc), 1, 1,
                                  static_cast<int32_t>(visPad_ / FLOATS_PER_BLOCK),
                                  ReduceOrder::ORDER_ONLY_VALUE);
            for (uint32_t p = 1; p < PiecesOf(vis); ++p) {
                WholeReduceMax<float>(max_[64], score_[p * 64u], MaskOf(vis, p),
                                      static_cast<int32_t>(hc), 1, 1,
                                      static_cast<int32_t>(visPad_ / FLOATS_PER_BLOCK),
                                      ReduceOrder::ORDER_ONLY_VALUE);
                PipeBarrier<PIPE_V>();
                Max(max_, max_, max_[64], hc);
            }
            PipeBarrier<PIPE_V>();
            Brcb(bcast_, max_, static_cast<uint8_t>((hc + 7u) / 8u), {1, 8});
            PipeBarrier<PIPE_V>();
            for (uint32_t p = 0; p < PiecesOf(vis); ++p) {
                BinaryRepeatParams subParams{1, 1, 0,
                                             static_cast<uint8_t>(visPad_ / FLOATS_PER_BLOCK),
                                             static_cast<uint8_t>(visPad_ / FLOATS_PER_BLOCK), 1};
                Sub(score_[p * 64u], score_[p * 64u], bcast_, MaskOf(vis, p),
                    static_cast<uint8_t>(hc), subParams);
            }
            PipeBarrier<PIPE_V>();
            Exp(score_, score_, hc * visPad_);
            PipeBarrier<PIPE_V>();
            for (uint32_t p = 0; p < PiecesOf(vis); ++p) {
                BinaryRepeatParams accParams{1, 1, 0, 0,
                                             static_cast<uint8_t>(visPad_ / FLOATS_PER_BLOCK), 0};
                MulAddDst(tgt_[p * 64u], score_[p * 64u], scratch_[128], MaskOf(vis, p),
                          static_cast<uint8_t>(hc), accParams);
            }
            PipeBarrier<PIPE_V>();
        }
        const float total = VecSum(tgt_, vis);
        Muls(tgt_, tgt_, 1.0f / total, vis);
        PipeBarrier<PIPE_V>();
    }

    // ------------------------------------------------------------- indexer logits
    __aicore__ inline void ComputeLogits(uint32_t b, uint32_t row, uint32_t vis) {
        Duplicate(sh_, 0.0f, visPad_);
        PipeBarrier<PIPE_V>();
        for (uint32_t k0 = 0; k0 < vis; k0 += keyChunk_) {
            const uint32_t kcCur = MinU32(keyChunk_, vis - k0);
            LoadKiRows(b, k0, kcCur);
            for (uint32_t i = 0; i < nidx_; ++i) {
                LoadQiRow(b, row, i);
                DotRows(sim_[i * simPitch_], kF_, qi_, kcCur, dim_, dimPad_);
                if (storeSim_ != 0u) {
                    DataCopy(sim_[i * visPad_ + k0], sim_[i * simPitch_], Align8(kcCur));
                }
            }
            PipeBarrier<PIPE_V>();
            Maxs(sim_, sim_, 0.0f, nidx_ * simPitch_);
            PipeBarrier<PIPE_V>();
            for (uint32_t p = 0; p < PiecesOf(kcCur); ++p) {
                BinaryRepeatParams accParams{1, 1, 0, 0,
                                             static_cast<uint8_t>(simPitch_ / FLOATS_PER_BLOCK), 1};
                MulAddDst(sh_[k0 + p * 64u], sim_[p * 64u], wb_, MaskOf(kcCur, p),
                          static_cast<uint8_t>(nidx_), accParams);
            }
            PipeBarrier<PIPE_V>();
        }
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
    __aicore__ inline void ComputeGradients(uint32_t b, uint32_t row, uint32_t vis) {
        Duplicate(dq_, 0.0f, nidx_ * dimPad_);
        Duplicate(dw_, 0.0f, nidx_);
        PipeBarrier<PIPE_V>();
        for (uint32_t k0 = 0; k0 < vis; k0 += keyChunk_) {
            const uint32_t kcCur = MinU32(keyChunk_, vis - k0);
            if (storeSim_ != 0u) {
                for (uint32_t i = 0; i < nidx_; ++i) {
                    DataCopy(sim_[i * simPitch_], sim_[i * visPad_ + k0], Align8(kcCur));
                }
                PipeBarrier<PIPE_V>();
            } else {
                LoadKiRows(b, k0, kcCur);
                for (uint32_t i = 0; i < nidx_; ++i) {
                    LoadQiRow(b, row, i);
                    DotRows(sim_[i * simPitch_], kF_, qi_, kcCur, dim_, dimPad_);
                }
                PipeBarrier<PIPE_V>();
            }
            Maxs(sim_, sim_, 0.0f, nidx_ * simPitch_);
            PipeBarrier<PIPE_V>();
            // dw += sum_j relu(S_ij) * delta_j
            for (uint32_t p = 0; p < PiecesOf(kcCur); ++p) {
                BinaryRepeatParams mulParams{1, 1, 1,
                                             static_cast<uint8_t>(simPitch_ / FLOATS_PER_BLOCK),
                                             static_cast<uint8_t>(simPitch_ / FLOATS_PER_BLOCK), 0};
                Mul(prod_[p * 64u], sim_[p * 64u], del_[k0 + p * 64u], MaskOf(kcCur, p),
                    static_cast<uint8_t>(nidx_), mulParams);
            }
            PipeBarrier<PIPE_V>();
            WholeReduceSum<float>(dwRow_, prod_, MaskOf(kcCur, 0), static_cast<int32_t>(nidx_), 1, 1,
                                  static_cast<int32_t>(simPitch_ / FLOATS_PER_BLOCK));
            for (uint32_t p = 1; p < PiecesOf(kcCur); ++p) {
                WholeReduceSum<float>(dwRow_[64], prod_[p * 64u], MaskOf(kcCur, p),
                                      static_cast<int32_t>(nidx_), 1, 1,
                                      static_cast<int32_t>(simPitch_ / FLOATS_PER_BLOCK));
                PipeBarrier<PIPE_V>();
                Add(dwRow_, dwRow_, dwRow_[64], nidx_);
            }
            PipeBarrier<PIPE_V>();
            Add(dw_, dw_, dwRow_, nidx_);
            PipeBarrier<PIPE_V>();
            // m = w * delta * (S > 0)
            Muls(sim_, sim_, INDICATOR_SCALE, nidx_ * simPitch_);
            PipeBarrier<PIPE_V>();
            Mins(sim_, sim_, 1.0f, nidx_ * simPitch_);
            PipeBarrier<PIPE_V>();
            for (uint32_t p = 0; p < PiecesOf(kcCur); ++p) {
                BinaryRepeatParams dParams{1, 1, 1,
                                           static_cast<uint8_t>(simPitch_ / FLOATS_PER_BLOCK),
                                           static_cast<uint8_t>(simPitch_ / FLOATS_PER_BLOCK), 0};
                Mul(sim_[p * 64u], sim_[p * 64u], del_[k0 + p * 64u], MaskOf(kcCur, p),
                    static_cast<uint8_t>(nidx_), dParams);
                PipeBarrier<PIPE_V>();
                BinaryRepeatParams wParams{1, 1, 0,
                                           static_cast<uint8_t>(simPitch_ / FLOATS_PER_BLOCK),
                                           static_cast<uint8_t>(simPitch_ / FLOATS_PER_BLOCK), 1};
                Mul(sim_[p * 64u], sim_[p * 64u], wb_, MaskOf(kcCur, p), static_cast<uint8_t>(nidx_),
                    wParams);
            }
            PipeBarrier<PIPE_V>();
            Duplicate(dkChunk_, 0.0f, kcCur * dimPad_);
            PipeBarrier<PIPE_V>();
            for (uint32_t i = 0; i < nidx_; ++i) {
                // dk 需要 queryIndex 的第 i 行；storeSim 路径下这里必须重新加载
                LoadQiRow(b, row, i);
                Brcb(bcast_, sim_[i * simPitch_], static_cast<uint8_t>((kcCur + 7u) / 8u), {1, 8});
                PipeBarrier<PIPE_V>();
                for (uint32_t p = 0; p < PiecesOf(dim_); ++p) {
                    const uint32_t mask = MaskOf(dim_, p);
                    Duplicate(scratch_, 0.0f, 64);
                    PipeBarrier<PIPE_V>();
                    BinaryRepeatParams dqParams{1, 1, 0, 0,
                                                static_cast<uint8_t>(dimPad_ / FLOATS_PER_BLOCK), 1};
                    MulAddDst(scratch_, kF_[p * 64u], bcast_, mask, static_cast<uint8_t>(kcCur),
                              dqParams);
                    PipeBarrier<PIPE_V>();
                    Add(dq_[i * dimPad_ + p * 64u], dq_[i * dimPad_ + p * 64u], scratch_, mask);
                    BinaryRepeatParams dkParams{1, 1, 0,
                                                static_cast<uint8_t>(dimPad_ / FLOATS_PER_BLOCK), 0, 1};
                    MulAddDst(dkChunk_[p * 64u], qi_[p * 64u], bcast_, mask,
                              static_cast<uint8_t>(kcCur), dkParams);
                }
                PipeBarrier<PIPE_V>();
            }
            PipeBarrier<PIPE_V>();
            StoreDkChunk(k0, kcCur);
        }
    }

    __aicore__ inline void StoreDkChunk(uint32_t k0, uint32_t kcCur) {
        if (dkRows_ >= s2_) {
            Add(dkAcc_[k0 * dimPad_], dkAcc_[k0 * dimPad_], dkChunk_, kcCur * dimPad_);
            PipeBarrier<PIPE_V>();
            return;
        }
        if (dkPartialElems_ == 0u) {
            return;  // 资源不足时不落盘（Host 保证不会走到这里）
        }
        // 读-加-写回 workspace 部分和
        DataCopy(prod_, workspaceGm_[static_cast<uint64_t>(taskBase_) * dkPartialElems_ +
                                     static_cast<uint64_t>(k0) * dimPad_],
                 kcCur * dimPad_);
        SyncMte2ToVec();
        Add(dkChunk_, dkChunk_, prod_, kcCur * dimPad_);
        PipeBarrier<PIPE_V>();
        SyncVecToMte3();
        DataCopy(workspaceGm_[static_cast<uint64_t>(taskBase_) * dkPartialElems_ +
                              static_cast<uint64_t>(k0) * dimPad_],
                 dkChunk_, kcCur * dimPad_);
    }

    __aicore__ inline void WriteRowOutputs(uint32_t b, uint32_t row) {
        const uint64_t dqBase = (static_cast<uint64_t>(b) * s1_ + row) * nidx_ * dim_;
        const uint64_t dwBase = (static_cast<uint64_t>(b) * s1_ + row) * nidx_;
        SyncVecToMte3();
        for (uint32_t i = 0; i < nidx_; ++i) {
            const uint64_t offset = dqBase + static_cast<uint64_t>(i) * dim_;
            DataCopyExtParams params{1, static_cast<uint32_t>(dim_ * sizeof(T)), 0, 0, 0};
            if constexpr (sizeof(T) == 4u) {
                DataCopyPad(dQueryIndexGm_[offset], dq_[i * dimPad_], params);
            } else {
                Cast(kT_, dq_[i * dimPad_], RoundMode::CAST_NONE, dimPad_);
                SyncVecToMte3();
                DataCopyPad(dQueryIndexGm_[offset], kT_, params);
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
        if (dkPartialElems_ == 0u || dkRows_ >= s2_) {
            Duplicate(dkAcc_, 0.0f, MinU32(dkRows_, s2_) * dimPad_);
        } else {
            // 分块回写路径：先把本任务的部分和清零
            Duplicate(prod_, 0.0f, dimPad_);
            PipeBarrier<PIPE_V>();
            SyncVecToMte3();
            for (uint32_t j = 0; j < s2_; ++j) {
                DataCopy(workspaceGm_[static_cast<uint64_t>(taskBase_) * dkPartialElems_ +
                                      static_cast<uint64_t>(j) * dimPad_],
                         prod_, dimPad_);
            }
        }
        PipeBarrier<PIPE_V>();
        for (uint32_t row = rowStart; row < rowEnd; ++row) {
            ProcessRow(b, row, VisibleOf(row));
        }
        WriteDk(b);
        if (coreNum_ > 1u) {
            scratch_[384].SetValue(0, rowLoss_);
            SetFlag<HardEvent::S_MTE3>(EV_S_MTE3);
            WaitFlag<HardEvent::S_MTE3>(EV_S_MTE3);
            DataCopy(workspaceGm_[lossOffset_ + task], scratch_[384], 8);
        } else {
            totalLoss_ += rowLoss_;
        }
    }

    __aicore__ inline void DebugDump(uint32_t b, uint32_t row, uint32_t vis) {
        // 打包到 dq_ 缓冲，一次 32B 对齐搬出
        uint32_t off = 0;
        Adds(dq_[off], tgt_, 0.0f, visPad_); off += visPad_;
        Adds(dq_[off], pred_, 0.0f, visPad_); off += visPad_;
        Adds(dq_[off], del_, 0.0f, visPad_); off += visPad_;
        Adds(dq_[off], sh_, 0.0f, visPad_); off += visPad_;
        for (uint32_t h = 0; h < headBlock_ && h < n1_; ++h) {
            Adds(dq_[off], score_[h * visPad_], 0.0f, visPad_); off += visPad_;
        }
        const uint32_t sp = Align8(simPitch_);
        for (uint32_t i = 0; i < nidx_; ++i) {
            Adds(dq_[off], sim_[i * simPitch_], 0.0f, sp); off += sp;
        }
        for (uint32_t j = 0; j < keyChunk_ * dimPad_; ++j) {
            (void)j;
            break;
        }
        PipeBarrier<PIPE_V>();
        Cast(kT_, dq_, RoundMode::CAST_NONE, off);
        SyncVecToMte3();
        DataCopyPad(dQueryIndexGm_[0], kT_,
                    DataCopyExtParams{1, static_cast<uint32_t>(off * sizeof(T)), 0, 0, 0});
        SyncMte3Done();
        (void)b;
        (void)row;
        (void)vis;
    }

    __aicore__ inline void ProcessRow(uint32_t b, uint32_t row, uint32_t vis) {
        if (vis == 0u) {
            Duplicate(dq_, 0.0f, nidx_ * dimPad_);
            Duplicate(dw_, 0.0f, nidx_);
            PipeBarrier<PIPE_V>();
            WriteRowOutputs(b, row);
            return;
        }
        ComputeTarget(b, row, vis);
        LoadWeights(b, row);
        ComputeLogits(b, row, vis);
        SoftmaxAndLoss(vis);
        ComputeGradients(b, row, vis);
        WriteRowOutputs(b, row);
#if DLIGL_DEBUG
        if (b == 0u && row == 0u) {
            DebugDump(b, row, vis);
        }
#endif
    }

    __aicore__ inline void WriteDk(uint32_t b) {
        if (dkPartialElems_ > 0u && dkRows_ < s2_) {
            return;  // 已分块写入 workspace
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
                if constexpr (sizeof(T) == 4u) {
                    DataCopyPad(dKeyIndexGm_[offset], dkAcc_[j * dimPad_], params);
                } else {
                    Cast(kT_, dkAcc_[j * dimPad_], RoundMode::CAST_NONE, dimPad_);
                    SyncVecToMte3();
                    DataCopyPad(dKeyIndexGm_[offset], kT_, params);
                    SyncMte3Done();
                }
            }
        }
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
                DataCopy(kF_, workspaceGm_[lossOffset_ + t], Align8(count));
                SyncMte2ToVec();
                total += VecSum(kF_, count);
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
                DataCopy(kF_, workspaceGm_[static_cast<uint64_t>(t) * dkPartialElems_ +
                                           static_cast<uint64_t>(j) * dimPad_],
                         dimPad_);
                SyncMte2ToVec();
                Add(scratch_, scratch_, kF_, dimPad_);
                PipeBarrier<PIPE_V>();
            }
            const uint64_t offset =
                static_cast<uint64_t>(b) * s2_ * dim_ + static_cast<uint64_t>(j) * dim_;
            SyncVecToMte3();
            DataCopyExtParams params{1, static_cast<uint32_t>(dim_ * sizeof(T)), 0, 0, 0};
            if constexpr (sizeof(T) == 4u) {
                DataCopyPad(dKeyIndexGm_[offset], scratch_, params);
            } else {
                Cast(kT_, scratch_, RoundMode::CAST_NONE, dimPad_);
                SyncVecToMte3();
                DataCopyPad(dKeyIndexGm_[offset], kT_, params);
                SyncMte3Done();
            }
        }
    }

private:
    uint32_t batch_ = 1, s1_ = 1, s2_ = 1, n1_ = 1, nidx_ = 1, dim_ = 1, dimPad_ = 1, visPad_ = 1;
    uint32_t causal_ = 1, rowsPerTask_ = 1, blocksPerBatch_ = 1, taskCount_ = 1;
    uint32_t headBlock_ = 1, keyChunk_ = 1, simPitch_ = 1, dkRows_ = 0, storeSim_ = 0, splitDk_ = 0;
    uint32_t dkPartialElems_ = 0, lossOffset_ = 0, inputBytes_ = 2, weightBytes_ = 2;
    uint32_t coreNum_ = 1;
    float scale_ = 1.0f;
    float rowLoss_ = 0.0f;
    float totalLoss_ = 0.0f;
    uint32_t taskBase_ = 0;
    DliglUbLayout layout_;

    LocalTensor<float> q_, kF_, prod_, score_, sim_, tgt_, sh_, pred_, del_, dkAcc_, dkChunk_;
    LocalTensor<float> qi_, dq_, part_, bcast_, max_, w_, wb_, dw_, dwRow_, scratch_;
    LocalTensor<T> kT_, qiTTmp_;
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
