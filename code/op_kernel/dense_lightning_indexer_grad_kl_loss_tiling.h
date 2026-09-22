// Tiling结构体定义：DenseLightningIndexerGradKlLoss
//
// 设计要点：单核（单 block）、纯 Vector（AIV_ONLY）实现，Host 预先算好 UB 布局，
// Kernel 只做行循环与按 key 的向量运算，避免核内做切分搜索与多核同步。
#pragma once

#include <cstdint>

struct DenseLightningIndexerGradKlLossTilingData {
    // ---- 形状 ----
    uint32_t batch;    // B
    uint32_t s1;       // S1
    uint32_t s2;       // S2
    uint32_t n1;       // 主注意力头数
    uint32_t nidx;     // indexer 头数
    uint32_t dim;      // D
    uint32_t dimPad;   // D 按 32B 对齐后的行宽（元素）
    uint32_t visPad;   // S2 按 8 对齐
    uint32_t n1Pad;    // N1 按 8 对齐
    uint32_t nidxPad;  // Nidx1 按 8 对齐
    uint32_t tmpPart;  // 辅助区内分片归约区长度（元素）
    uint32_t tmpLogit; // 辅助区内 logits 暂存区长度（元素）
    uint32_t tmpElems; // 辅助区浮点数个数

    // ---- 分块参数 ----
    uint32_t hb;      // 主注意力头块大小（<= n1）
    uint32_t ib;      // indexer 头块大小（<= nidx）
    uint32_t kcRows;  // 一次 DMA 载入的 key 行数（>= 1）

    // ---- UB 偏移（字节，均为 32B 对齐）----
    uint32_t offQ;      // (n1Pad, dimPad) CT
    uint32_t offK;      // (kcRows * n1Pad, dimPad) CT
    uint32_t offProdT;  // (hb, dimPad) CT（fp32 输入时与 prodF 同址）
    uint32_t offProdF;  // (hb, dimPad) fp32
    uint32_t offQi;     // (nidxPad, dimPad) CT 载入暂存
    uint32_t offQiF;    // (nidxPad, dimPad) fp32
    uint32_t offKi;     // (visPad, dimPad) CT
    uint32_t offKiF;    // (visPad, dimPad) fp32（fp32 输入时与 offKi 同址）
    uint32_t offSc;     // (visPad, n1Pad) fp32
    uint32_t offU;      // (visPad, nidxPad) fp32
    uint32_t offSh;     // visPad fp32（当前 logits）
    uint32_t offTgt;    // visPad fp32（未归一化 target）
    uint32_t offPred;   // visPad fp32
    uint32_t offDel;    // visPad fp32
    uint32_t offDb;     // 8 * visPad fp32
    uint32_t offW;      // nidxPad fp32
    uint32_t offWRaw;   // nidxPad * weightBytes：weights 原始载入暂存
    uint32_t offDw;     // nidxPad fp32
    uint32_t offDsb;    // 8 * ib fp32
    uint32_t offDq;     // (ib, dimPad) fp32
    uint32_t offDk;     // (visPad, dimPad) fp32
    uint32_t offMv;     // n1Pad fp32（每头最大值）
    uint32_t offStage;  // max(ib, visPad) * dimPad CT（写回转换暂存）
    uint32_t offTmp;    // tmpElems fp32（分片归约 + 标量暂存 + lz）
    uint32_t ubBytes;   // 总占用

    // ---- 其它 ----
    uint32_t weightsFp32;  // 1: weights 为 float32
    float scale;           // 注意力缩放系数
};

// UB 布局计算：Host 用它对候选 (hb, ib, kcRows) 做可行性判断，Kernel 直接取偏移。
// inputBytes: query 元素字节数（2/4）；weightBytes: weights 元素字节数（2/4）。
inline DenseLightningIndexerGradKlLossTilingData DliglComputeLayout(
    uint32_t batch, uint32_t s1, uint32_t s2, uint32_t n1, uint32_t nidx, uint32_t dim,
    uint32_t hb, uint32_t ib, uint32_t kcRows, uint32_t inputBytes, uint32_t weightBytes,
    float scale, uint32_t weightsFp32) {
    const uint32_t ctBytes = (inputBytes == 4u) ? 4u : 2u;
    const auto align32 = [](uint64_t v) { return static_cast<uint32_t>((v + 31ull) & ~31ull); };

    DenseLightningIndexerGradKlLossTilingData t{};
    t.batch = batch;
    t.s1 = s1;
    t.s2 = s2;
    t.n1 = n1;
    t.nidx = nidx;
    t.dim = dim;
    t.dimPad = (dim + (32u / inputBytes) - 1u) / (32u / inputBytes) * (32u / inputBytes);
    t.visPad = ((s2 + 7u) / 8u) * 8u;
    t.n1Pad = ((n1 + 7u) / 8u) * 8u;
    t.nidxPad = ((nidx + 7u) / 8u) * 8u;
    t.hb = hb;
    t.ib = ib;
    t.kcRows = kcRows;
    t.weightsFp32 = weightsFp32;
    t.scale = scale;

    const uint64_t dimPad = t.dimPad;
    // 辅助区：分片归约区 + logits 暂存 + 逐元素临时 + 标量暂存
    const uint32_t rowsMax = (hb > ib) ? hb : ib;
    const uint32_t p1 = (dim + 63u) / 64u;
    const uint32_t p2 = (t.nidxPad + 63u) / 64u;
    const uint32_t p3 = (t.visPad + 63u) / 64u;
    uint32_t maxPieces = p1 > p2 ? p1 : p2;
    if (p3 > maxPieces) {
        maxPieces = p3;
    }
    if (maxPieces < 5u) {
        maxPieces = 5u;
    }
    t.tmpPart = maxPieces * (rowsMax > 8u ? rowsMax : 8u) + 64u;
    t.tmpLogit = t.nidxPad > t.visPad ? t.nidxPad : t.visPad;
    // 布局：分片归约 | logits | tmpA(visPad) | brc(64，Brcb 固定写 8 个 block)
    //       | lz | scalarA | scalarB | lossAcc(visPad)
    t.tmpElems = t.tmpPart + t.tmpLogit + t.visPad + 64u + 24u + t.visPad + 16u;

    uint32_t off = 0;
    t.offQ = off;
    off += align32(static_cast<uint64_t>(t.n1Pad) * dimPad * ctBytes);
    t.offK = off;
    off += align32(static_cast<uint64_t>(kcRows) * n1 * dimPad * ctBytes);
    t.offProdF = off;
    off += align32(static_cast<uint64_t>(hb) * dimPad * 4u);
    t.offProdT = (ctBytes == 4u) ? t.offProdF : off;
    if (ctBytes != 4u) {
        off += align32(static_cast<uint64_t>(hb) * dimPad * ctBytes);
    }
    t.offQi = off;
    off += align32(static_cast<uint64_t>(t.nidxPad) * dimPad * ctBytes);
    t.offQiF = off;
    if (ctBytes != 4u) {
        off += align32(static_cast<uint64_t>(t.nidxPad) * dimPad * 4u);
    } else {
        t.offQiF = t.offQi;  // fp32 输入无需精度转换，直接原地使用
    }
    t.offKi = off;
    off += align32(static_cast<uint64_t>(t.visPad) * dimPad * ctBytes);
    t.offKiF = off;
    if (ctBytes != 4u) {
        off += align32(static_cast<uint64_t>(t.visPad) * dimPad * 4u);
    } else {
        t.offKiF = t.offKi;
    }
    t.offSc = off;
    off += align32(static_cast<uint64_t>(t.visPad) * t.n1Pad * 4u);
    t.offU = off;
    off += align32(static_cast<uint64_t>(t.visPad) * t.nidxPad * 4u);
    t.offSh = off;
    off += align32(static_cast<uint64_t>(t.visPad) * 4u);
    t.offTgt = off;
    off += align32(static_cast<uint64_t>(t.visPad) * 4u);
    t.offPred = off;
    off += align32(static_cast<uint64_t>(t.visPad) * 4u);
    t.offDel = off;
    off += align32(static_cast<uint64_t>(t.visPad) * 4u);
    t.offDb = off;
    off += align32(static_cast<uint64_t>(t.visPad) * 8u * 4u);
    t.offW = off;
    if (weightBytes == 4u) {
        t.offWRaw = off;  // fp32 weights：原始数据即最终数据，共用一块
    }
    off += align32(static_cast<uint64_t>(t.nidxPad) * 4u);
    if (weightBytes != 4u) {
        t.offWRaw = off;
        off += align32(static_cast<uint64_t>(t.nidxPad) * weightBytes);
    }
    t.offDw = off;
    off += align32(static_cast<uint64_t>(t.nidxPad) * 4u);
    t.offDsb = off;
    off += align32(static_cast<uint64_t>(ib) * 8u * 4u);
    t.offDq = off;
    off += align32(static_cast<uint64_t>(ib) * dimPad * 4u);
    t.offDk = off;
    off += align32(static_cast<uint64_t>(t.visPad) * dimPad * 4u);
    t.offMv = off;
    off += align32(static_cast<uint64_t>(t.n1Pad) * 4u);
    t.offStage = off;
    off += align32(static_cast<uint64_t>(ib > t.visPad ? ib : t.visPad) * dimPad * ctBytes);
    t.offTmp = off;
    off += align32(static_cast<uint64_t>(t.tmpElems) * 4u);
    t.ubBytes = off;
    return t;
}
