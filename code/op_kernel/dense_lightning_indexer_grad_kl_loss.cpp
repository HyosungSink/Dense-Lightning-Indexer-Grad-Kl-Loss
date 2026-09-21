// Kernel侧核函数实现
#include "kernel_operator.h"

#include "dense_lightning_indexer_grad_kl_loss_tiling.h"
#include "tiling_key_dense_lightning_indexer_grad_kl_loss.h"

using namespace AscendC;

namespace {
constexpr int32_t BUFFER_NUM = 1;

template <typename T>
__aicore__ inline void CastUpToFloat(const LocalTensor<float> &dst, const LocalTensor<T> &src, uint32_t count) {
    if constexpr (std::is_same<T, float>::value) {
        DataCopy(dst, src, count);
    } else {
        Cast(dst, src, RoundMode::CAST_NONE, count);
    }
}

template <typename T>
__aicore__ inline void CastDownFromFloat(const LocalTensor<T> &dst, const LocalTensor<float> &src, uint32_t count) {
    if constexpr (std::is_same<T, float>::value) {
        DataCopy(dst, src, count);
    } else {
        Cast(dst, src, RoundMode::CAST_RINT, count);
    }
}
}  // namespace

template <class DT_QUERY, class DT_WEIGHT>
class KernelDenseLightningIndexerGradKlLoss {
public:
    __aicore__ inline KernelDenseLightningIndexerGradKlLoss() {}

    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR queryIndex, GM_ADDR keyIndex,
                                 GM_ADDR weights, GM_ADDR dQueryIndex, GM_ADDR dKeyIndex,
                                 GM_ADDR dWeights, GM_ADDR loss,
                                 const DenseLightningIndexerGradKlLossTilingData &tiling) {
        batch_ = tiling.batch;
        s1_ = tiling.s1;
        s2_ = tiling.s2;
        n1_ = tiling.n1;
        nidx1_ = tiling.nidx1;
        d_ = tiling.d;
        scale_ = tiling.scaleValue;

        uint32_t coreIdx = GetBlockIdx();
        uint32_t coreNum = tiling.coreNum;
        uint32_t bigCoreCount = tiling.bigCoreCount;
        uint32_t rowsBase = tiling.rowsPerCoreBase;
        if (coreIdx < bigCoreCount) {
            rowCount_ = rowsBase + 1;
            rowStart_ = coreIdx * (rowsBase + 1);
        } else {
            rowCount_ = rowsBase;
            rowStart_ = bigCoreCount * (rowsBase + 1) + (coreIdx - bigCoreCount) * rowsBase;
        }
        isActive_ = (coreIdx < coreNum) && (rowCount_ > 0);

        queryGm_.SetGlobalBuffer((__gm__ DT_QUERY *)query);
        keyGm_.SetGlobalBuffer((__gm__ DT_QUERY *)key);
        queryIndexGm_.SetGlobalBuffer((__gm__ DT_QUERY *)queryIndex);
        keyIndexGm_.SetGlobalBuffer((__gm__ DT_QUERY *)keyIndex);
        weightsGm_.SetGlobalBuffer((__gm__ DT_WEIGHT *)weights);
        dQueryIndexGm_.SetGlobalBuffer((__gm__ DT_QUERY *)dQueryIndex);
        dKeyIndexGm_.SetGlobalBuffer((__gm__ DT_QUERY *)dKeyIndex);
        dWeightsGm_.SetGlobalBuffer((__gm__ DT_WEIGHT *)dWeights);
        lossGm_.SetGlobalBuffer((__gm__ float *)loss);

        uint32_t n1d = n1_ * d_;
        uint32_t nidx1d = nidx1_ * d_;
        uint32_t s2n1 = s2_ * n1_;
        uint32_t s2nidx1 = s2_ * nidx1_;
        uint32_t wideMax = n1d > nidx1d ? n1d : nidx1d;

        pipe_.InitBuffer(qFullBuf_, n1d * sizeof(float));
        pipe_.InitBuffer(qiFullBuf_, nidx1d * sizeof(float));
        pipe_.InitBuffer(wFullBuf_, nidx1_ * sizeof(float));
        pipe_.InitBuffer(kRowBuf_, n1d * sizeof(float));
        pipe_.InitBuffer(kiBcBuf_, nidx1d * sizeof(float));
        pipe_.InitBuffer(mainScoresBuf_, s2n1 * sizeof(float));
        pipe_.InitBuffer(headVecBuf_, n1_ * sizeof(float));
        pipe_.InitBuffer(headVecBuf2_, n1_ * sizeof(float));
        pipe_.InitBuffer(headVecBuf3_, n1_ * sizeof(float));
        pipe_.InitBuffer(indexSimBuf_, s2nidx1 * sizeof(float));
        pipe_.InitBuffer(dSBuf_, s2nidx1 * sizeof(float));
        pipe_.InitBuffer(idxVecBuf_, nidx1_ * sizeof(float));
        pipe_.InitBuffer(idxVecBuf2_, nidx1_ * sizeof(float));
        pipe_.InitBuffer(s2VecBuf_, (s2_ + 8) * sizeof(float));
        pipe_.InitBuffer(s2VecBuf2_, (s2_ + 8) * sizeof(float));
        pipe_.InitBuffer(s2VecBuf3_, (s2_ + 8) * sizeof(float));
        pipe_.InitBuffer(s2VecBuf4_, (s2_ + 8) * sizeof(float));
        pipe_.InitBuffer(maskU8Buf_, wideMax * sizeof(uint8_t));
        pipe_.InitBuffer(sharedTmpBuf_, wideMax * sizeof(float));
        pipe_.InitBuffer(wideScratchBuf_, wideMax * sizeof(float));
        pipe_.InitBuffer(dWideBuf_, d_ * sizeof(float));
        pipe_.InitBuffer(dWideBuf2_, d_ * sizeof(float));
        pipe_.InitBuffer(castOutBuf_, wideMax * sizeof(DT_QUERY));
        pipe_.InitBuffer(castOutWBuf_, nidx1_ * sizeof(DT_WEIGHT));
        pipe_.InitBuffer(zeroWideBuf_, 8192 * sizeof(float));
        pipe_.InitBuffer(lossAccumBuf_, 8 * sizeof(float));
    }

    __aicore__ inline void Process() {
        ZeroOutputs();
        SyncAll();
        if (!isActive_) {
            return;
        }

        LocalTensor<float> lossAccum = lossAccumBuf_.Get<float>();
        Duplicate(lossAccum, 0.0f, 8);

        uint32_t curBatch = 0xFFFFFFFFu;
        for (uint32_t r = 0; r < rowCount_; ++r) {
            uint32_t rowIdx = rowStart_ + r;
            uint32_t b = rowIdx / s1_;
            uint32_t s = rowIdx % s1_;
            (void)curBatch;
            ProcessRow(b, s, lossAccum);
        }

        LocalTensor<float> lossSum = sharedTmpBuf_.Get<float>();
        ReduceSum<float>(lossSum, lossAccum, lossAccum, 8);
        SetAtomicAdd<float>();
        DataCopy(lossGm_, lossSum, 8);
        SetAtomicNone();
    }

private:
    __aicore__ inline void ZeroOutputs() {
        LocalTensor<float> zeroBuf = zeroWideBuf_.Get<float>();
        Duplicate(zeroBuf, 0.0f, 8192);
        LocalTensor<DT_QUERY> zeroCast = castOutBuf_.Get<DT_QUERY>();
        uint32_t wideMax = zeroCast.GetSize();
        uint32_t chunk = wideMax < 8192 ? wideMax : 8192;
        if (chunk == 0) {
            chunk = 1;
        }
        CastDownFromFloat<DT_QUERY>(zeroCast, zeroBuf, chunk);

        uint32_t coreIdx = GetBlockIdx();
        uint32_t coreNumTotal = GetBlockNum();
        uint64_t totalKeyElems = static_cast<uint64_t>(batch_) * s2_ * d_;
        uint64_t perCore = (totalKeyElems + coreNumTotal - 1) / coreNumTotal;
        uint64_t start = static_cast<uint64_t>(coreIdx) * perCore;
        uint64_t end = start + perCore;
        if (end > totalKeyElems) {
            end = totalKeyElems;
        }
        uint64_t pos = start;
        while (pos < end) {
            uint64_t remain = end - pos;
            uint32_t take = remain < chunk ? static_cast<uint32_t>(remain) : chunk;
            DataCopy(dKeyIndexGm_[pos], zeroCast, take);
            pos += take;
        }
        if (coreIdx == 0) {
            LocalTensor<float> zeroOne = sharedTmpBuf_.Get<float>();
            Duplicate(zeroOne, 0.0f, 8);
            DataCopy(lossGm_, zeroOne, 8);
        }
    }

    __aicore__ inline void LoadBatchKeyIndexRow(uint32_t b, uint32_t j, LocalTensor<float> &kiBc) {
        LocalTensor<DT_QUERY> raw = castOutBuf_.Get<DT_QUERY>();
        uint64_t offset = (static_cast<uint64_t>(b) * s2_ + j) * d_;
        DataCopy(raw, keyIndexGm_[offset], d_);
        CastUpToFloat<DT_QUERY>(kiBc, raw, d_);
        uint32_t filled = d_;
        uint32_t total = nidx1_ * d_;
        while (filled < total) {
            uint32_t copyLen = filled < (total - filled) ? filled : (total - filled);
            DataCopy(kiBc[filled], kiBc, copyLen);
            filled += copyLen;
        }
    }

    __aicore__ inline void ProcessRow(uint32_t b, uint32_t s, LocalTensor<float> &lossAccum) {
        LocalTensor<float> qFull = qFullBuf_.Get<float>();
        LocalTensor<float> qiFull = qiFullBuf_.Get<float>();
        LocalTensor<float> wFull = wFullBuf_.Get<float>();
        LocalTensor<float> kRow = kRowBuf_.Get<float>();
        LocalTensor<float> kiBc = kiBcBuf_.Get<float>();
        LocalTensor<float> mainScores = mainScoresBuf_.Get<float>();
        LocalTensor<float> indexSim = indexSimBuf_.Get<float>();
        LocalTensor<float> dS = dSBuf_.Get<float>();
        LocalTensor<float> sharedTmp = sharedTmpBuf_.Get<float>();
        LocalTensor<uint8_t> maskU8 = maskU8Buf_.Get<uint8_t>();
        LocalTensor<DT_QUERY> castOut = castOutBuf_.Get<DT_QUERY>();
        LocalTensor<DT_WEIGHT> castOutW = castOutWBuf_.Get<DT_WEIGHT>();

        uint32_t n1d = n1_ * d_;
        uint32_t nidx1d = nidx1_ * d_;

        {
            LocalTensor<DT_QUERY> raw = castOut;
            DataCopy(raw, queryGm_[(static_cast<uint64_t>(b) * s1_ + s) * n1d], n1d);
            CastUpToFloat<DT_QUERY>(qFull, raw, n1d);
        }
        {
            LocalTensor<DT_QUERY> raw = castOut;
            DataCopy(raw, queryIndexGm_[(static_cast<uint64_t>(b) * s1_ + s) * nidx1d], nidx1d);
            CastUpToFloat<DT_QUERY>(qiFull, raw, nidx1d);
        }
        {
            LocalTensor<DT_WEIGHT> raw = castOutW;
            DataCopy(raw, weightsGm_[(static_cast<uint64_t>(b) * s1_ + s) * nidx1_], nidx1_);
            CastUpToFloat<DT_WEIGHT>(wFull, raw, nidx1_);
        }

        // ---- main attention scores: mainScores[j, :] = scale * sum_d q*k ----
        for (uint32_t j = 0; j < s2_; ++j) {
            LocalTensor<DT_QUERY> raw = castOut;
            DataCopy(raw, keyGm_[(static_cast<uint64_t>(b) * s2_ + j) * n1d], n1d);
            CastUpToFloat<DT_QUERY>(kRow, raw, n1d);
            Mul(kRow, qFull, kRow, n1d);
            Sum<float>(mainScores[j * n1_], kRow, sharedTmp, {n1_, RoundUp32(d_), d_});
            Muls(mainScores[j * n1_], mainScores[j * n1_], scale_, n1_);
        }

        // ---- per-head softmax over s2, then sum-over-heads (in place: mainScores becomes exp) ----
        LocalTensor<float> runningMax = headVecBuf_.Get<float>();
        LocalTensor<float> runningSum = headVecBuf2_.Get<float>();
        LocalTensor<float> recipMain = headVecBuf3_.Get<float>();
        DataCopy(runningMax, mainScores, n1_);
        for (uint32_t j = 1; j < s2_; ++j) {
            Max(runningMax, runningMax, mainScores[j * n1_], n1_);
        }
        Duplicate(runningSum, 0.0f, n1_);
        for (uint32_t j = 0; j < s2_; ++j) {
            Sub(mainScores[j * n1_], mainScores[j * n1_], runningMax, n1_);
            Exp(mainScores[j * n1_], mainScores[j * n1_], n1_);
            Add(runningSum, runningSum, mainScores[j * n1_], n1_);
        }
        Reciprocal(recipMain, runningSum, n1_);

        LocalTensor<float> pRaw = s2VecBuf_.Get<float>();
        for (uint32_t j = 0; j < s2_; ++j) {
            Mul(mainScores[j * n1_], mainScores[j * n1_], recipMain, n1_);
            ReduceSum<float>(sharedTmp, mainScores[j * n1_], sharedTmp, n1_);
            float rowSum = sharedTmp.GetValue(0);
            pRaw.SetValue(j, rowSum);
        }
        ReduceSum<float>(sharedTmp, pRaw, sharedTmp, s2_);
        float pTotal = sharedTmp.GetValue(0);
        float pTotalRecip = 1.0f / pTotal;
        LocalTensor<float> p = s2VecBuf2_.Get<float>();
        Muls(p, pRaw, pTotalRecip, s2_);

        // ---- indexer forward: indexSim, relu(in place), index scores ----
        LocalTensor<float> indexScores = s2VecBuf3_.Get<float>();
        for (uint32_t j = 0; j < s2_; ++j) {
            LoadBatchKeyIndexRow(b, j, kiBc);
            Mul(kiBc, qiFull, kiBc, nidx1d);
            Sum<float>(indexSim[j * nidx1_], kiBc, sharedTmp, {nidx1_, RoundUp32(d_), d_});
            Relu(indexSim[j * nidx1_], indexSim[j * nidx1_], nidx1_);
            LocalTensor<float> prod = idxVecBuf_.Get<float>();
            Mul(prod, indexSim[j * nidx1_], wFull, nidx1_);
            ReduceSum<float>(sharedTmp, prod, sharedTmp, nidx1_);
            indexScores.SetValue(j, sharedTmp.GetValue(0));
        }

        // ---- softmax over indexScores(s2,) ----
        ReduceMax<float>(sharedTmp, indexScores, sharedTmp, s2_);
        float iMax = sharedTmp.GetValue(0);
        LocalTensor<float> predicted = s2VecBuf4_.Get<float>();
        Adds(predicted, indexScores, -iMax, s2_);
        Exp(predicted, predicted, s2_);
        ReduceSum<float>(sharedTmp, predicted, sharedTmp, s2_);
        float iSum = sharedTmp.GetValue(0);
        float iSumRecip = 1.0f / iSum;
        Muls(predicted, predicted, iSumRecip, s2_);

        float lnISum = LogScalar(iSum);
        // logPredicted = indexScores - iMax - lnISum ; reuse indexScores buffer.
        Adds(indexScores, indexScores, -(iMax + lnISum), s2_);

        // dI = predicted - p ; reuse predicted buffer.
        Sub(predicted, predicted, p, s2_);

        // loss row: sum over j where p[j] > 0 of p[j] * (ln(p[j]) - logPredicted[j])
        LocalTensor<float> lnP = sharedTmp;
        Ln(lnP, p, s2_);
        Sub(lnP, lnP, indexScores, s2_);
        Mul(lnP, lnP, p, s2_);
        CompareScalar(maskU8, p, 0.0f, CMPMODE::GT, s2_);
        LocalTensor<float> zeroFill = idxVecBuf2_.Get<float>();
        Duplicate(zeroFill, 0.0f, s2_ > nidx1_ ? s2_ : nidx1_);
        Select(lnP, maskU8, lnP, zeroFill, SELMODE::VSEL_TENSOR_TENSOR_MODE, s2_);
        LocalTensor<float> rowLossBuf = idxVecBuf_.Get<float>();
        ReduceSum<float>(rowLossBuf, lnP, rowLossBuf, s2_);
        Add(lossAccum, lossAccum, rowLossBuf, 1);

        // ---- backward: dW, dS ----
        LocalTensor<float> dW = idxVecBuf_.Get<float>();
        Duplicate(dW, 0.0f, nidx1_);
        for (uint32_t j = 0; j < s2_; ++j) {
            float dIj = predicted.GetValue(j);
            LocalTensor<float> tmpNidx = idxVecBuf2_.Get<float>();
            Muls(tmpNidx, indexSim[j * nidx1_], dIj, nidx1_);
            Add(dW, dW, tmpNidx, nidx1_);

            CompareScalar(maskU8, indexSim[j * nidx1_], 0.0f, CMPMODE::GT, nidx1_);
            Muls(tmpNidx, wFull, dIj, nidx1_);
            LocalTensor<float> zeroN = wideScratchBuf_.Get<float>();
            Duplicate(zeroN, 0.0f, nidx1_);
            Select(dS[j * nidx1_], maskU8, tmpNidx, zeroN, SELMODE::VSEL_TENSOR_TENSOR_MODE, nidx1_);
        }

        {
            LocalTensor<DT_WEIGHT> outW = castOutW;
            CastDownFromFloat<DT_WEIGHT>(outW, dW, nidx1_);
            DataCopy(dWeightsGm_[(static_cast<uint64_t>(b) * s1_ + s) * nidx1_], outW, nidx1_);
        }

        // dQueryIndex[hi,:] = sum_j dS[j,hi] * KI[j,:]
        LocalTensor<float> dQIndex = qFull;  // main-attention phase for q is done; reuse its buffer.
        Duplicate(dQIndex, 0.0f, nidx1d);
        for (uint32_t j = 0; j < s2_; ++j) {
            LocalTensor<DT_QUERY> raw = castOut;
            uint64_t offset = (static_cast<uint64_t>(b) * s2_ + j) * d_;
            DataCopy(raw, keyIndexGm_[offset], d_);
            LocalTensor<float> kiRow = dWideBuf_.Get<float>();
            CastUpToFloat<DT_QUERY>(kiRow, raw, d_);
            for (uint32_t hi = 0; hi < nidx1_; ++hi) {
                float dsVal = dS.GetValue(j * nidx1_ + hi);
                LocalTensor<float> tmpD = dWideBuf2_.Get<float>();
                Muls(tmpD, kiRow, dsVal, d_);
                Add(dQIndex[hi * d_], dQIndex[hi * d_], tmpD, d_);
            }
        }
        {
            LocalTensor<DT_QUERY> outCast = castOut;
            CastDownFromFloat<DT_QUERY>(outCast, dQIndex, nidx1d);
            DataCopy(dQueryIndexGm_[(static_cast<uint64_t>(b) * s1_ + s) * nidx1d], outCast, nidx1d);
        }

        // dKeyIndex[j,:] += sum_hi dS[j,hi] * QI[hi,:]  (atomic add across rows/cores)
        for (uint32_t j = 0; j < s2_; ++j) {
            LocalTensor<float> rowAcc = dWideBuf_.Get<float>();
            Duplicate(rowAcc, 0.0f, d_);
            for (uint32_t hi = 0; hi < nidx1_; ++hi) {
                float dsVal = dS.GetValue(j * nidx1_ + hi);
                LocalTensor<float> tmpD = dWideBuf2_.Get<float>();
                Muls(tmpD, qiFull[hi * d_], dsVal, d_);
                Add(rowAcc, rowAcc, tmpD, d_);
            }
            LocalTensor<DT_QUERY> outCast = castOut;
            CastDownFromFloat<DT_QUERY>(outCast, rowAcc, d_);
            SetAtomicAdd<DT_QUERY>();
            DataCopy(dKeyIndexGm_[(static_cast<uint64_t>(b) * s2_ + j) * d_], outCast, d_);
            SetAtomicNone();
        }
    }

    __aicore__ inline uint32_t RoundUp32(uint32_t n) {
        uint32_t bytes = n * sizeof(float);
        uint32_t rounded = (bytes + 31u) / 32u * 32u;
        return rounded / sizeof(float);
    }

    __aicore__ inline float LogScalar(float value) {
        LocalTensor<float> tmp = wideScratchBuf_.Get<float>();
        Duplicate(tmp, value, 8);
        Ln(tmp, tmp, 8);
        return tmp.GetValue(0);
    }

    GlobalTensor<DT_QUERY> queryGm_, keyGm_, queryIndexGm_, keyIndexGm_, dQueryIndexGm_, dKeyIndexGm_;
    GlobalTensor<DT_WEIGHT> weightsGm_, dWeightsGm_;
    GlobalTensor<float> lossGm_;

    TPipe pipe_;
    TBuf<TPosition::VECCALC> qFullBuf_, qiFullBuf_, wFullBuf_, kRowBuf_, kiBcBuf_, mainScoresBuf_;
    TBuf<TPosition::VECCALC> headVecBuf_, headVecBuf2_, headVecBuf3_;
    TBuf<TPosition::VECCALC> indexSimBuf_, dSBuf_, idxVecBuf_, idxVecBuf2_;
    TBuf<TPosition::VECCALC> s2VecBuf_, s2VecBuf2_, s2VecBuf3_, s2VecBuf4_;
    TBuf<TPosition::VECCALC> maskU8Buf_, sharedTmpBuf_, wideScratchBuf_, dWideBuf_, dWideBuf2_;
    TBuf<TPosition::VECCALC> castOutBuf_, castOutWBuf_, zeroWideBuf_, lossAccumBuf_;

    uint32_t batch_ = 0, s1_ = 0, s2_ = 0, n1_ = 0, nidx1_ = 0, d_ = 0;
    float scale_ = 0.0f;
    uint32_t rowStart_ = 0, rowCount_ = 0;
    bool isActive_ = false;
};

template <typename DT_QUERY, typename DT_WEIGHT>
__global__ __aicore__ void dense_lightning_indexer_grad_kl_loss(
    GM_ADDR query, GM_ADDR key, GM_ADDR query_index, GM_ADDR key_index, GM_ADDR weights,
    GM_ADDR d_query_index, GM_ADDR d_key_index, GM_ADDR d_weights, GM_ADDR loss,
    GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(DenseLightningIndexerGradKlLossTilingData);
    GET_TILING_DATA_WITH_STRUCT(DenseLightningIndexerGradKlLossTilingData, tiling_data, tiling);
    KernelDenseLightningIndexerGradKlLoss<DT_QUERY, DT_WEIGHT> op;
    op.Init(query, key, query_index, key_index, weights, d_query_index, d_key_index, d_weights, loss,
            tiling_data);
    op.Process();
}
