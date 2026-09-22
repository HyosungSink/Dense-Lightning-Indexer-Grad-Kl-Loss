// Kernel侧核函数实现：DenseLightningIndexerGradKlLoss
//
// 实现路线：单核（blockDim=1）+ 纯 Vector（AIV_ONLY），Host 预计算 UB 布局。
//
// 每个 query 行的流程（全部在 UB 内、按 key 组织，避免逐 key 的 GM 往返与标量同步）：
//   1) 一次性发起 q / weights / queryIndex / keyIndex / key 首块的搬运，只等一次 MTE2；
//   2) 逐 key 求主注意力分数 SC[j][h]（一次 Mul 覆盖 hb 个头）并维护每头最大值；
//   3) 逐 key 求 indexer 相似度 U[j][i]=relu(qi·ki) 与 logits 投影 sh[j]；
//   4) 第二遍遍历 UB 内的 SC 得到 target 分布 p（exp(SC-m) 后按头求和再 L1 归一化）；
//   5) softmax(sh) 得预测分布 pred，dI = pred - p；
//   6) 逐 key 回传：ds = w*dI*(U>0)，dq += ki⊗ds，dk += qi⊗ds，dW += U*dI。
//
// 关键点：整行只在“归一化系数”上做一次 V->S 同步，loss 在最后做一次归约，
// 其余全部是向量指令；单核下无需 SyncAll 与跨核归约。
#include "kernel_operator.h"

#include "dense_lightning_indexer_grad_kl_loss_tiling.h"
#include "tiling_key_dense_lightning_indexer_grad_kl_loss.h"

using namespace AscendC;

namespace {
constexpr float NEG_INF = -3.0e38f;
constexpr float IND_SCALE = 1.0e30f;
constexpr uint32_t FP32_PER_BLK = 8;
constexpr uint32_t FP32_PER_REP = 64;
constexpr int8_t EV_MTE2_V = EVENT_ID0;
constexpr int8_t EV_V_MTE3 = EVENT_ID1;
constexpr int8_t EV_V_MTE2 = EVENT_ID2;
constexpr int8_t EV_MTE3_V = EVENT_ID3;
constexpr int8_t EV_V_S = EVENT_ID4;
}  // namespace

template <typename T, typename TW>
class KernelDenseLightningIndexerGradKlLoss {
public:
    using CT = typename std::conditional<sizeof(T) == 4u, float, half>::type;

    __aicore__ inline KernelDenseLightningIndexerGradKlLoss() {}

    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR query_index, GM_ADDR key_index,
                                GM_ADDR weights, GM_ADDR d_query_index, GM_ADDR d_key_index,
                                GM_ADDR d_weights, GM_ADDR loss, GM_ADDR workspace,
                                const DenseLightningIndexerGradKlLossTilingData &td) {
        (void)workspace;
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
        hb_ = td.hb;
        ib_ = td.ib;
        kcRows_ = td.kcRows > 0u ? td.kcRows : 1u;
        scale_ = td.scale;
        inputBytes_ = static_cast<uint32_t>(sizeof(T));
        weightBytes_ = static_cast<uint32_t>(sizeof(TW));
        weightsFp32_ = td.weightsFp32;

        queryGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(query),
                                 static_cast<uint64_t>(batch_) * s1_ * n1_ * dim_);
        keyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(key),
                               static_cast<uint64_t>(batch_) * s2_ * n1_ * dim_);
        queryIndexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(query_index),
                                      static_cast<uint64_t>(batch_) * s1_ * nidx_ * dim_);
        keyIndexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(key_index),
                                    static_cast<uint64_t>(batch_) * s2_ * dim_);
        weightsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ TW *>(weights),
                                   static_cast<uint64_t>(batch_) * s1_ * nidx_);
        dQueryIndexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(d_query_index),
                                       static_cast<uint64_t>(batch_) * s1_ * nidx_ * dim_);
        dKeyIndexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(d_key_index),
                                     static_cast<uint64_t>(batch_) * s2_ * dim_);
        dWeightsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ TW *>(d_weights),
                                    static_cast<uint64_t>(batch_) * s1_ * nidx_);
        lossGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(loss), 1);

        qT_ = LocalTensor<CT>(TPosition::VECCALC, td.offQ, n1_ * dimPad_);
        kT_ = LocalTensor<CT>(TPosition::VECCALC, td.offK, kcRows_ * n1_ * dimPad_);
        prodT_ = LocalTensor<CT>(TPosition::VECCALC, td.offProdT, prodRows_() * dimPad_);
        prodF_ = LocalTensor<float>(TPosition::VECCALC, td.offProdF, prodRows_() * dimPad_);
        qiT_ = LocalTensor<CT>(TPosition::VECCALC, td.offQi, nidxPad_ * dimPad_);
        qiF_ = LocalTensor<float>(TPosition::VECCALC, td.offQiF, nidxPad_ * dimPad_);
        kiT_ = LocalTensor<CT>(TPosition::VECCALC, td.offKi, visPad_ * dimPad_);
        kiF_ = LocalTensor<float>(TPosition::VECCALC, td.offKiF, visPad_ * dimPad_);
        sc_ = LocalTensor<float>(TPosition::VECCALC, td.offSc, visPad_ * n1Pad_);
        u_ = LocalTensor<float>(TPosition::VECCALC, td.offU, visPad_ * nidxPad_);
        sh_ = LocalTensor<float>(TPosition::VECCALC, td.offSh, visPad_);
        tgt_ = LocalTensor<float>(TPosition::VECCALC, td.offTgt, visPad_);
        pred_ = LocalTensor<float>(TPosition::VECCALC, td.offPred, visPad_);
        del_ = LocalTensor<float>(TPosition::VECCALC, td.offDel, visPad_);
        db_ = LocalTensor<float>(TPosition::VECCALC, td.offDb, visPad_ * 8u);
        w_ = LocalTensor<float>(TPosition::VECCALC, td.offW, nidxPad_);
        dw_ = LocalTensor<float>(TPosition::VECCALC, td.offDw, nidxPad_);
        dsb_ = LocalTensor<float>(TPosition::VECCALC, td.offDsb, ib_ * 8u);
        dq_ = LocalTensor<float>(TPosition::VECCALC, td.offDq, ib_ * dimPad_);
        dk_ = LocalTensor<float>(TPosition::VECCALC, td.offDk, visPad_ * dimPad_);
        mv_ = LocalTensor<float>(TPosition::VECCALC, td.offMv, n1Pad_);
        stageT_ = LocalTensor<CT>(TPosition::VECCALC, td.offStage,
                                  (ib_ > visPad_ ? ib_ : visPad_) * dimPad_);
        tmp_ = LocalTensor<float>(TPosition::VECCALC, td.offTmp, td.tmpElems);
        wRaw_ = LocalTensor<TW>(TPosition::VECCALC, td.offWRaw, nidxPad_);

        // tmp_ 内部划分：分片归约区 / logits / 逐元素临时 / 广播 / 标量 / 常量1 / loss
        part_ = tmp_[0];
        logits_ = tmp_[td.tmpPart];
        tmpA_ = tmp_[td.tmpPart + td.tmpLogit];
        brc_ = tmpA_[visPad_];
        lz_ = brc_[64u];
        scalarA_ = lz_[8u];
        scalarB_ = scalarA_[8u];
        lossAcc_ = scalarB_[8u];
    }

    __aicore__ inline void Process() {
        if (GetBlockIdx() != 0u) {
            return;
        }
        Duplicate(lossAcc_, 0.0f, visPad_);
        for (uint32_t b = 0; b < batch_; ++b) {
            Duplicate(dk_, 0.0f, visPad_ * dimPad_);
            for (uint32_t row = 0; row < s1_; ++row) {
                ProcessRow(b, row);
            }
            StoreDk(b);
        }
        SumToPoint(scalarA_, lossAcc_, visPad_);
        SetFlag<HardEvent::V_S>(EV_V_S);
        WaitFlag<HardEvent::V_S>(EV_V_S);
        lossGm_.SetValue(0, scalarA_.GetValue(0));
    }

private:
    __aicore__ inline uint32_t MinU32(uint32_t a, uint32_t b) const { return a < b ? a : b; }
    __aicore__ inline uint32_t Align8(uint32_t v) const { return (v + 7u) & ~7u; }
    __aicore__ inline uint32_t PiecesDim() const { return (dim_ + FP32_PER_REP - 1u) / FP32_PER_REP; }
    // 主注意力与相似度共用的乘积缓冲行数
    __aicore__ inline uint32_t prodRows_() const { return hb_ > ib_ ? hb_ : ib_; }

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

    // ---------------------------------------------------------------- 归约
    // count 个连续元素求和（或最大值），结果写在 dst[0]
    __aicore__ inline void SumToPoint(const LocalTensor<float> &dst, const LocalTensor<float> &src,
                                      uint32_t count) {
        const uint32_t pieces = (count + FP32_PER_REP - 1u) / FP32_PER_REP;
        if (pieces <= 1u) {
            WholeReduceSum<float>(dst, src, count, 1, 1, 1, 1);
            return;
        }
        for (uint32_t p = 0; p < pieces; ++p) {
            const uint32_t left = count - p * FP32_PER_REP;
            const uint32_t mask = left > FP32_PER_REP ? FP32_PER_REP : left;
            WholeReduceSum<float>(part_[p * 8], src[p * FP32_PER_REP], mask, 1, 1, 1, 1);
        }
        for (uint32_t p = 1; p < pieces; ++p) {
            Add(part_, part_, part_[p * 8], 8);
        }
        // 用归约指令回写：dst 往往是 tgt_[j]/sh_[j] 这类非 32B 对齐地址，
        // 普通向量写会触发非法地址，而归约指令每个 repeat 只写一个元素。
        WholeReduceSum<float>(dst, part_, pieces, 1, 1, 1, 1);
    }

    __aicore__ inline void MaxToPoint(const LocalTensor<float> &dst, const LocalTensor<float> &src,
                                      uint32_t count) {
        const uint32_t pieces = (count + FP32_PER_REP - 1u) / FP32_PER_REP;
        WholeReduceMax<float>(part_, src, MinU32(count, FP32_PER_REP), 1, 1, 1, 1,
                              ReduceOrder::ORDER_ONLY_VALUE);
        for (uint32_t p = 1; p < pieces; ++p) {
            const uint32_t left = count - p * FP32_PER_REP;
            const uint32_t mask = left > FP32_PER_REP ? FP32_PER_REP : left;
            WholeReduceMax<float>(part_[p * 8], src[p * FP32_PER_REP], mask, 1, 1, 1, 1,
                                  ReduceOrder::ORDER_ONLY_VALUE);
            Max(part_, part_, part_[p * 8], 8);
        }
        Adds(dst, part_, 0.0f, 1);
    }

    // rows 行、每行 width 个元素（行距 pitch），逐行求和到 dst[0..rows)
    __aicore__ inline void ReduceRowsInto(const LocalTensor<float> &dst, const LocalTensor<float> &src,
                                          uint32_t rows, uint32_t width, uint32_t pitch) {
        const uint32_t pieces = (width + FP32_PER_REP - 1u) / FP32_PER_REP;
        const uint8_t stride = static_cast<uint8_t>(pitch / FP32_PER_BLK);
        for (uint32_t p = 0; p < pieces; ++p) {
            const uint32_t left = width - p * FP32_PER_REP;
            const uint32_t mask = left > FP32_PER_REP ? FP32_PER_REP : left;
            WholeReduceSum<float>(part_[p * rows], src[p * FP32_PER_REP], mask,
                                  static_cast<int32_t>(rows), 1, 1, static_cast<int32_t>(stride));
        }
        if (pieces > 1u) {
            for (uint32_t p = 1; p < pieces; ++p) {
                Add(part_, part_, part_[p * rows], static_cast<int32_t>(rows));
            }
        }
        Adds(dst, part_, 0.0f, static_cast<int32_t>(rows));
    }

    __aicore__ inline void VecMaxInto(const LocalTensor<float> &dst, const LocalTensor<float> &src,
                                      uint32_t count) {
        const uint32_t pieces = (count + FP32_PER_REP - 1u) / FP32_PER_REP;
        for (uint32_t p = 0; p < pieces; ++p) {
            const uint32_t left = count - p * FP32_PER_REP;
            const uint32_t mask = left > FP32_PER_REP ? FP32_PER_REP : left;
            Max(dst[p * FP32_PER_REP], dst[p * FP32_PER_REP], src[p * FP32_PER_REP], mask);
        }
    }

    // ---------------------------------------------------------------- DMA
    __aicore__ inline bool BulkOk(uint64_t base, uint64_t elems, uint32_t elemBytes) const {
        return elems != 0ull && (elems * elemBytes) % 32u == 0u && (base * elemBytes) % 32u == 0u;
    }

    __aicore__ inline void LoadRows(const GlobalTensor<T> &src, uint64_t base, uint32_t rows,
                                    LocalTensor<CT> dst) {
        if (dimPad_ == dim_ && BulkOk(base, static_cast<uint64_t>(rows) * dim_, inputBytes_)) {
            DataCopy(dst, src[base], rows * dim_);
            return;
        }
        DataCopyExtParams params{static_cast<uint16_t>(rows),
                                 static_cast<uint32_t>(dim_ * inputBytes_), 0,
                                 static_cast<uint32_t>((dimPad_ - dim_) * inputBytes_ / 32u), 0};
        DataCopyPad(dst, src[base], params, DataCopyPadExtParams<CT>{false, 0, 0, 0});
    }

    __aicore__ inline void StoreRows(const GlobalTensor<T> &dst, uint64_t base, uint32_t rows,
                                     const LocalTensor<CT> &src) {
        if (dimPad_ == dim_ && BulkOk(base, static_cast<uint64_t>(rows) * dim_, sizeof(T))) {
            DataCopy(dst[base], src, rows * dim_);
            return;
        }
        DataCopyExtParams params{static_cast<uint16_t>(rows),
                                 static_cast<uint32_t>(dim_ * sizeof(T)),
                                 static_cast<uint32_t>((dimPad_ - dim_) * sizeof(T) / 32u), 0, 0};
        DataCopyPad(dst[base], src, params);
    }

    __aicore__ inline void IssueWeights(uint32_t b, uint32_t row) {
        const uint64_t base = (static_cast<uint64_t>(b) * s1_ + row) * nidx_;
        DataCopyExtParams params{1, static_cast<uint32_t>(nidx_ * weightBytes_), 0, 0, 0};
        DataCopyPad(wRaw_, weightsGm_[base], params, DataCopyPadExtParams<TW>{false, 0, 0, 0});
    }

    // 载入完成后：整块清零再只转换真实元素，尾部天然为 0
    // （避免对 w_[nidx] 这种非 32B 对齐地址做向量写）
    __aicore__ inline void FinishWeights() {
        if (nidxPad_ > nidx_) {
            Duplicate(w_, 0.0f, static_cast<int32_t>(nidxPad_));
        }
        if constexpr (sizeof(TW) != 4u) {
            Cast(w_, wRaw_, RoundMode::CAST_NONE, static_cast<int32_t>(nidx_));
        }
    }

    __aicore__ inline void IssueKChunk(uint32_t b, uint32_t j0, uint32_t rows) {
        const uint64_t base = (static_cast<uint64_t>(b) * s2_ + j0) * n1_ * dim_;
        LoadRows(keyGm_, base, rows * n1_, kT_);
    }

    // ---------------------------------------------------------------- 逐 key 计算
    // SC[j][h] 与每头最大值 mv[h]
    __aicore__ inline void ScoresKey(uint32_t jj, uint32_t j) {
        constexpr uint32_t LANES = 256u / sizeof(CT);
        constexpr uint8_t TSTRIDE_DIV = 32u / sizeof(CT);
        const uint32_t pieces = (dim_ + LANES - 1u) / LANES;
        const uint8_t tStride = static_cast<uint8_t>(dimPad_ / TSTRIDE_DIV);
        const uint32_t kBase = jj * n1_ * dimPad_;
        for (uint32_t h0 = 0; h0 < n1_; h0 += hb_) {
            const uint32_t hc = MinU32(hb_, n1_ - h0);
            for (uint32_t p = 0; p < pieces; ++p) {
                const uint32_t left = dim_ - p * LANES;
                const uint32_t mask = left > LANES ? LANES : left;
                BinaryRepeatParams params{1, 1, 1, tStride, tStride, tStride};
                if constexpr (sizeof(CT) == 4u) {
                    Mul(prodF_[p * LANES], kT_[kBase + h0 * dimPad_ + p * LANES],
                        qT_[h0 * dimPad_ + p * LANES], mask, static_cast<uint8_t>(hc), params);
                } else {
                    Mul(prodT_[p * LANES], kT_[kBase + h0 * dimPad_ + p * LANES],
                        qT_[h0 * dimPad_ + p * LANES], mask, static_cast<uint8_t>(hc), params);
                }
            }
            if constexpr (sizeof(CT) != 4u) {
                Cast(prodF_, prodT_, RoundMode::CAST_NONE, hc * dimPad_);
            }
            ReduceRowsInto(sc_[j * n1Pad_ + h0], prodF_, hc, dim_, dimPad_);
            // 注意：必须在维护每头最大值之前完成 scale，保证 exp(SC-m) 与 SC 同尺度
            Muls(sc_[j * n1Pad_ + h0], sc_[j * n1Pad_ + h0], scale_, static_cast<int32_t>(hc));
            VecMaxInto(mv_[h0], sc_[j * n1Pad_ + h0], hc);
        }
    }

    // U[j][i] = relu(qi_i · ki_j)
    __aicore__ inline void SimKey(uint32_t j) {
        const uint32_t pieces = PiecesDim();
        const uint8_t fStride = static_cast<uint8_t>(dimPad_ / FP32_PER_BLK);
        for (uint32_t i0 = 0; i0 < nidx_; i0 += ib_) {
            const uint32_t ic = MinU32(ib_, nidx_ - i0);
            for (uint32_t p = 0; p < pieces; ++p) {
                const uint32_t left = dim_ - p * FP32_PER_REP;
                const uint32_t mask = left > FP32_PER_REP ? FP32_PER_REP : left;
                BinaryRepeatParams params{1, 1, 1, fStride, fStride, 0};
                Mul(prodF_[p * FP32_PER_REP], qiF_[i0 * dimPad_ + p * FP32_PER_REP],
                    kiF_[j * dimPad_ + p * FP32_PER_REP], mask, static_cast<uint8_t>(ic), params);
            }
            if (pieces > 1u) {
                BinaryRepeatParams addParams{1, 1, 1, fStride, fStride, fStride};
                for (uint32_t p = 1; p < pieces; ++p) {
                    Add(prodF_, prodF_, prodF_[p * FP32_PER_REP], FP32_PER_REP,
                        static_cast<uint8_t>(ic), addParams);
                }
            }
            ReduceRowsInto(u_[j * nidxPad_ + i0], prodF_, ic,
                           pieces > 1u ? FP32_PER_REP : dim_, dimPad_);
            Maxs(u_[j * nidxPad_ + i0], u_[j * nidxPad_ + i0], 0.0f, static_cast<int32_t>(ic));
        }
    }

    // logits_j = Σ_i w_i * U[j][i]
    __aicore__ inline void LogitKey(uint32_t j) {
        Mul(logits_, u_[j * nidxPad_], w_, static_cast<int32_t>(nidx_));
        SumToPoint(sh_[j], logits_, nidx_);
    }

    // ---------------------------------------------------------------- target / softmax
    __aicore__ inline void ComputeTargetAndDelta() {
        // p_raw[j] = Σ_h exp(SC[j][h] - mv[h])
        for (uint32_t j = 0; j < s2_; ++j) {
            for (uint32_t h0 = 0; h0 < n1_; h0 += hb_) {
                const uint32_t hc = MinU32(hb_, n1_ - h0);
                const uint32_t pieces = (hc + FP32_PER_REP - 1u) / FP32_PER_REP;
                for (uint32_t p = 0; p < pieces; ++p) {
                    const uint32_t left = hc - p * FP32_PER_REP;
                    const uint32_t mask = left > FP32_PER_REP ? FP32_PER_REP : left;
                    Sub(sc_[j * n1Pad_ + h0 + p * FP32_PER_REP],
                        sc_[j * n1Pad_ + h0 + p * FP32_PER_REP], mv_[h0 + p * FP32_PER_REP], mask);
                }
                Exp(sc_[j * n1Pad_ + h0], sc_[j * n1Pad_ + h0], hc);
            }
            SumToPoint(tgt_[j], sc_[j * n1Pad_], n1_);
        }
        // 归一化 target：p = tgt / Σ_j tgt_j；softmax(sh)：pred = exp(sh-m)/Σ
        // 归一化系数是标量，整行只做一次 V->S 同步（最大值用向量广播，不下标量）
        SumToPoint(scalarA_, tgt_, visPad_);
        MaxToPoint(scalarB_, sh_, visPad_);
        if (visPad_ == 8u) {
            // Brcb 一次迭代写 8 个 DataBlock：block0 即 src[0] 的 8 份拷贝，正好铺满 8 个通道，
            // 因此 8 通道场景可以在不读标量的情况下完成 softmax 平移。
            Brcb(brc_, scalarB_, 1, {1, 8});
            Sub(sh_, sh_, brc_, visPad_);
            Exp(pred_, sh_, visPad_);
            // 先把 lane1..7 置 1：count=1 的向量搬移在部分实现下会按 DataBlock 生效，
            // 这样即使整块被写入，lz_ 得到的仍是 [Z,1,1,...]，Ln 后除 lane0 外都是 0。
            Duplicate(scalarB_, 1.0f, 8u);
            SumToPoint(scalarB_, pred_, visPad_);
            SetFlag<HardEvent::V_S>(EV_V_S);
            WaitFlag<HardEvent::V_S>(EV_V_S);
            const float totalT = scalarA_.GetValue(0);
            const float totalZ = scalarB_.GetValue(0);
            FinishTarget(totalT, totalZ);
            return;
        }
        SetFlag<HardEvent::V_S>(EV_V_S);
        WaitFlag<HardEvent::V_S>(EV_V_S);
        Adds(sh_, sh_, -scalarB_.GetValue(0), visPad_);
        Exp(pred_, sh_, visPad_);
        Duplicate(scalarB_, 1.0f, 8u);
        SumToPoint(scalarB_, pred_, visPad_);
        SetFlag<HardEvent::V_S>(EV_V_S);
        WaitFlag<HardEvent::V_S>(EV_V_S);
        const float totalT = scalarA_.GetValue(0);
        const float totalZ = scalarB_.GetValue(0);
        FinishTarget(totalT, totalZ);
    }

    __aicore__ inline void FinishTarget(float totalT, float totalZ) {
        Muls(tgt_, tgt_, 1.0f / totalT, visPad_);
        Muls(pred_, pred_, 1.0f / totalZ, visPad_);
        Sub(del_, pred_, tgt_, visPad_);
        Brcb(db_, del_, static_cast<uint8_t>(visPad_ / 8u), {1, 8});
        // loss 项：Σ p*(ln p - shifted) + ln Z
        Maxs(tmpA_, tgt_, 1.0e-30f, visPad_);
        Ln(tmpA_, tmpA_, visPad_);
        Mul(tmpA_, tmpA_, tgt_, visPad_);
        Mul(logits_, tgt_, sh_, visPad_);
        Sub(tmpA_, tmpA_, logits_, visPad_);
        Add(lossAcc_, lossAcc_, tmpA_, visPad_);
        // Ln 会把 lane1..7 覆盖为 0，跨行复用会让下一行得到 ln(0)=-inf，必须逐行重建
        Duplicate(lz_, 1.0f, 8u);
        Adds(lz_, scalarB_, 0.0f, 1);
        Ln(lz_, lz_, 8);
        Add(lossAcc_, lossAcc_, lz_, 8);
    }

    // ---------------------------------------------------------------- 梯度
    __aicore__ inline void GradKey(uint32_t j, uint32_t i0, uint32_t ic, uint32_t icPad) {
        const uint32_t dimPieces = PiecesDim();
        const uint8_t dimStride = static_cast<uint8_t>(dimPad_ / FP32_PER_BLK);
        // ds_i = w_i * delta_j * (U>0)
        // 暂存必须按 icPad(<=ib) 个元素分配：part_ 区长度 >= max(hb,ib) 且梯度阶段已不再使用
        const LocalTensor<float> dsTmp = part_;
        Muls(dsTmp, u_[j * nidxPad_ + i0], IND_SCALE, static_cast<int32_t>(icPad));
        Mins(dsTmp, dsTmp, 1.0f, static_cast<int32_t>(icPad));
        Mul(dsTmp, dsTmp, w_[i0], static_cast<int32_t>(icPad));
        const uint32_t dpieces = (icPad + FP32_PER_REP - 1u) / FP32_PER_REP;
        for (uint32_t p = 0; p < dpieces; ++p) {
            const uint32_t left = icPad - p * FP32_PER_REP;
            const uint32_t mask = left > FP32_PER_REP ? FP32_PER_REP : left;
            BinaryRepeatParams pr{1, 1, 0, 1, 1, 0};
            Mul(dsTmp[p * FP32_PER_REP], dsTmp[p * FP32_PER_REP], db_[j * 8], mask, 1, pr);
        }
        Brcb(dsb_, dsTmp, static_cast<uint8_t>(icPad / 8u), {1, 8});
        // dq += ki_j ⊗ ds
        for (uint32_t p = 0; p < dimPieces; ++p) {
            const uint32_t left = dim_ - p * FP32_PER_REP;
            const uint32_t mask = left > FP32_PER_REP ? FP32_PER_REP : left;
            BinaryRepeatParams dqParams{1, 1, 0, dimStride, 0, 1};
            MulAddDst(dq_[p * FP32_PER_REP], kiF_[j * dimPad_ + p * FP32_PER_REP], dsb_, mask,
                      static_cast<uint8_t>(icPad), dqParams);
        }
        // dk_j += Σ_i qi_i ⊗ ds_i
        for (uint32_t p = 0; p < dimPieces; ++p) {
            const uint32_t left = dim_ - p * FP32_PER_REP;
            const uint32_t mask = left > FP32_PER_REP ? FP32_PER_REP : left;
            BinaryRepeatParams dkParams{1, 1, 0, 0, dimStride, 1};
            MulAddDst(dk_[j * dimPad_ + p * FP32_PER_REP], qiF_[i0 * dimPad_ + p * FP32_PER_REP],
                      dsb_, mask, static_cast<uint8_t>(icPad), dkParams);
        }
        // dW += U_j * delta_j
        const uint32_t wpieces = (icPad + FP32_PER_REP - 1u) / FP32_PER_REP;
        for (uint32_t p = 0; p < wpieces; ++p) {
            const uint32_t left = icPad - p * FP32_PER_REP;
            const uint32_t mask = left > FP32_PER_REP ? FP32_PER_REP : left;
            BinaryRepeatParams pr{1, 1, 0, 1, 1, 0};
            MulAddDst(dw_[i0 + p * FP32_PER_REP], u_[j * nidxPad_ + i0 + p * FP32_PER_REP], db_[j * 8],
                      mask, 1, pr);
        }
    }

    // 写回：dq 分块 + dWeights 合并到同一次 V->MTE3 同步窗口
    __aicore__ inline void StoreOutputs(uint32_t b, uint32_t row, uint32_t i0, uint32_t ic,
                                        bool last) {
        const uint64_t dqBase = (static_cast<uint64_t>(b) * s1_ + row) * nidx_ * dim_ +
                                static_cast<uint64_t>(i0) * dim_;
        const uint64_t dwBase = (static_cast<uint64_t>(b) * s1_ + row) * nidx_;
        if constexpr (sizeof(CT) != 4u) {
            Cast(stageT_, dq_, RoundMode::CAST_NONE, ic * dimPad_);
        }
        if (last) {
            if constexpr (sizeof(TW) != 4u) {
                Cast(wRaw_, dw_, RoundMode::CAST_NONE, static_cast<int32_t>(nidx_));
            }
        }
        SyncVecToMte3();
        if constexpr (sizeof(CT) == 4u) {
            StoreRows(dQueryIndexGm_, dqBase, ic, dq_);
        } else {
            StoreRows(dQueryIndexGm_, dqBase, ic, stageT_);
        }
        if (last) {
            DataCopyExtParams params{1, static_cast<uint32_t>(nidx_ * sizeof(TW)), 0, 0, 0};
            if constexpr (sizeof(TW) == 4u) {
                DataCopyPad(dWeightsGm_[dwBase], dw_, params);
            } else {
                DataCopyPad(dWeightsGm_[dwBase], wRaw_, params);
            }
        }
        SyncMte3Done();
    }

    __aicore__ inline void StoreDk(uint32_t b) {
        const uint64_t base = static_cast<uint64_t>(b) * s2_ * dim_;
        if constexpr (sizeof(CT) == 4u) {
            SyncVecToMte3();
            StoreRows(dKeyIndexGm_, base, s2_, dk_);
        } else {
            Cast(stageT_, dk_, RoundMode::CAST_NONE, s2_ * dimPad_);
            SyncVecToMte3();
            StoreRows(dKeyIndexGm_, base, s2_, stageT_);
        }
        SyncMte3Done();
    }

    __aicore__ inline void ProcessRow(uint32_t b, uint32_t row) {
        // 一次性发起全部搬运，只等一次
        SyncVecDone();
        SyncMte3Done();
        const uint64_t qBase = (static_cast<uint64_t>(b) * s1_ + row) * n1_ * dim_;
        const uint64_t qiBase = (static_cast<uint64_t>(b) * s1_ + row) * nidx_ * dim_;
        const uint64_t kiBase = static_cast<uint64_t>(b) * s2_ * dim_;
        IssueWeights(b, row);
        LoadRows(queryGm_, qBase, n1_, qT_);
        LoadRows(queryIndexGm_, qiBase, nidx_, qiT_);
        LoadRows(keyIndexGm_, kiBase, s2_, kiT_);
        IssueKChunk(b, 0u, MinU32(kcRows_, s2_));
        SyncMte2ToVec();
        FinishWeights();
        if constexpr (sizeof(CT) != 4u) {
            Cast(kiF_, kiT_, RoundMode::CAST_NONE, visPad_ * dimPad_);
            Cast(qiF_, qiT_, RoundMode::CAST_NONE, nidxPad_ * dimPad_);
        }
        Duplicate(mv_, NEG_INF, n1Pad_);
        Duplicate(sh_, NEG_INF, visPad_);
        Duplicate(tgt_, 0.0f, visPad_);
        Duplicate(u_, 0.0f, visPad_ * nidxPad_);
        for (uint32_t c = 0; c < s2_; c += kcRows_) {
            const uint32_t rows = MinU32(kcRows_, s2_ - c);
            if (c != 0u) {
                SyncVecDone();
                IssueKChunk(b, c, rows);
                SyncMte2ToVec();
            }
            for (uint32_t jj = 0; jj < rows; ++jj) {
                const uint32_t j = c + jj;
                ScoresKey(jj, j);
                SimKey(j);
                LogitKey(j);
            }
        }
        ComputeTargetAndDelta();
        Duplicate(dw_, 0.0f, nidxPad_);
        for (uint32_t i0 = 0; i0 < nidx_; i0 += ib_) {
            const uint32_t ic = MinU32(ib_, nidx_ - i0);
            const uint32_t icPad = MinU32(ib_, Align8(ic));
            Duplicate(dq_, 0.0f, icPad * dimPad_);
            for (uint32_t j = 0; j < s2_; ++j) {
                GradKey(j, i0, ic, icPad);
            }
            StoreOutputs(b, row, i0, ic, i0 + ib_ >= nidx_);
        }
    }

private:
    uint32_t batch_ = 1, s1_ = 1, s2_ = 1, n1_ = 1, nidx_ = 1, dim_ = 1, dimPad_ = 1, visPad_ = 1;
    uint32_t n1Pad_ = 1, nidxPad_ = 1, hb_ = 1, ib_ = 1, kcRows_ = 1;
    uint32_t inputBytes_ = 2, weightBytes_ = 2, weightsFp32_ = 0;
    float scale_ = 1.0f;

    LocalTensor<CT> qT_, kT_, prodT_, qiT_, kiT_, stageT_;
    LocalTensor<float> prodF_, qiF_, kiF_, sc_, u_, sh_, tgt_, pred_, del_, db_, w_, dw_, dsb_;
    LocalTensor<float> dq_, dk_, mv_, tmp_, part_, logits_, tmpA_, scalarA_, scalarB_, brc_, lz_;
    LocalTensor<float> lossAcc_;
    LocalTensor<TW> wRaw_;

    GlobalTensor<T> queryGm_, keyGm_, queryIndexGm_, keyIndexGm_;
    GlobalTensor<T> dQueryIndexGm_, dKeyIndexGm_;
    GlobalTensor<TW> weightsGm_, dWeightsGm_;
    GlobalTensor<float> lossGm_;
};

template <typename DT_QUERY>
__global__ __aicore__ void dense_lightning_indexer_grad_kl_loss(
    GM_ADDR query, GM_ADDR key, GM_ADDR query_index, GM_ADDR key_index, GM_ADDR weights,
    GM_ADDR d_query_index, GM_ADDR d_key_index, GM_ADDR d_weights, GM_ADDR loss, GM_ADDR workspace,
    GM_ADDR tiling) {
    // 纯 Vector 算子：只启动 Vector 核，避免混合下发时 Cube 核的空转头开销；
    // 单核实现不含任何多核同步指令，因此可以安全使用 AIV_ONLY。
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
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
