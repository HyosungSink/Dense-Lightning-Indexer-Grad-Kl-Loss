// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

// 行分派与 UB 切分参数全部由 Host 计算，Kernel 只做通用循环。
struct DenseLightningIndexerGradKlLossTilingData {
    uint32_t batch;           // B
    uint32_t s1;              // S1
    uint32_t s2;              // S2
    uint32_t n1;              // N1 主注意力头数
    uint32_t nidx;            // Nidx1 indexer 头数
    uint32_t dim;             // D
    uint32_t dimPad;          // D 按 32B 对齐的 fp32 行宽
    uint32_t visPad;          // 可见 key 数按 32B 对齐
    uint32_t n1Pad;           // N1 按 32B 对齐
    uint32_t nidxPad;         // Nidx1 按 32B 对齐
    uint32_t causal;          // 1: rightDownCausal
    uint32_t rowsPerTask;     // 每任务 query 行数
    uint32_t blocksPerBatch;  // 每 batch 的任务数
    uint32_t taskCount;       // 任务总数
    uint32_t headBlock;       // 主注意力头块
    uint32_t dkRows;          // UB 中 dk 累加器 key 行数
    uint32_t splitDk;         // 1: 需要跨任务归约 dKeyIndex
    uint32_t dkPartialElems;  // 每任务 dk 部分和元素数（0 表示不需要）
    uint32_t dkOffset;        // workspace 中 dk 部分和起始浮点偏移
    uint32_t lossOffset;      // workspace 中 loss 累加槽偏移（单个 32B 对齐槽）
    uint32_t wsSysBytes;      // workspace 起始处系统保留字节数（用户数据偏移）
    uint32_t kiCacheRows;     // keyIndex 行缓存行数（0 表示不做缓存，逐 key 载入）
    uint32_t weightsFp32;     // 1: weights 为 float32
    uint32_t stageElems;      // bf16 输入时的 T 暂存元素数（0 表示不需要）
    uint32_t blockDimUsed;    // 启动核数
    float scale;              // 注意力缩放系数
};

// UB 布局：Host 用它挑选可行切分，Kernel 用它取每个缓冲区的字节偏移。
struct DliglUbLayout {
    uint32_t outTOff;    // dimPad 个 T（写回时的转换暂存）
    uint32_t stageOff;   // bf16 输入时的原始暂存（T）
    uint32_t qTOff;      // (n1, dimPad) CT 整行 query
    uint32_t kTOff;      // (n1, dimPad) T  该 key 的整行 key
    uint32_t prodTOff;   // (headBlock, dimPad) T  乘积（低精度）
    uint32_t prodOff;    // (headBlock, dimPad) fp32 乘积（相似度按 headBlock 行分块复用）
    uint32_t qiStageOff; // (QI_STAGE, dimPad) T 分块暂存
    uint32_t qiFOff;     // (nidx, dimPad) fp32
    uint32_t kiTOff;     // dimPad T
    uint32_t kiFOff;     // dimPad fp32
    uint32_t kiCacheOff;  // (kiCacheRows, dimPad) fp32：keyIndex 行缓存
    uint32_t kiCacheTOff; // (kiCacheRows, dimPad) T：keyIndex 行缓存载入暂存
    uint32_t scOff;      // (visPad, n1Pad) fp32 主注意力分数（key-major）
    uint32_t uOff;       // (visPad, nidxPad) fp32 indexer 相似度（key-major, relu 后）
    uint32_t tgtOff;
    uint32_t shOff;
    uint32_t lossAccOff;  // visPad：按 key 累积的 loss 项
    uint32_t lzOff;       // 8：lnZ 暂存
    uint32_t predOff;
    uint32_t delOff;
    uint32_t dkOff;      // (dkRows, dimPad) fp32
    uint32_t dkRowOff;   // dimPad fp32
    uint32_t dqOff;      // (nidx, dimPad) fp32
    uint32_t maxOff;     // n1Pad fp32
    uint32_t dblkOff;    // visPad*8 fp32
    uint32_t mblkOff;    // nidx*8 fp32
    uint32_t partOff;    // 2048 fp32（逐行归约分片）
    uint32_t wRawOff;
    uint32_t wOff;
    uint32_t wbOff;
    uint32_t dwOff;
    uint32_t dwRowOff;
    uint32_t tmpOff;     // 8*64 fp32
    uint32_t totalBytes;
};

#ifndef __aicore__
#define DLIGL_UB_FN inline
#else
#define DLIGL_UB_FN __aicore__ inline
#endif

constexpr uint32_t QI_STAGE_ROWS = 8u;

DLIGL_UB_FN DliglUbLayout DliglComputeUbLayout(uint32_t headBlock, uint32_t n1, uint32_t nidx,
                                               uint32_t dimPad, uint32_t visPad, uint32_t n1Pad,
                                               uint32_t nidxPad, uint32_t dkRows, uint32_t inputBytes,
                                               uint32_t weightBytes, uint32_t stageElems,
                                               uint32_t kiCacheRows = 0u) {
    DliglUbLayout l;
    uint32_t off = 0;
    uint32_t partElems = ((dimPad + 63u) / 64u) * ((n1 > nidx ? n1 : nidx) + 7u & ~7u);
    if (partElems < 64u) {
        partElems = 64u;
    }
    if (partElems > 2048u) {
        partElems = 2048u;
    }
    l.outTOff = off;
    if (inputBytes != 4u) {
        off += ((dimPad * 2u + 31u) & ~31u);
    }
    l.stageOff = off;
    if (stageElems != 0u) {
        off += ((stageElems * inputBytes + 31u) & ~31u);
    }
    l.qTOff = off;
    off += ((n1 * dimPad * (inputBytes == 4u ? 4u : 2u) + 31u) & ~31u);
    l.kTOff = off;
    off += ((n1 * dimPad * (inputBytes == 4u ? 4u : 2u) + 31u) & ~31u);
    l.prodTOff = off;
    if (inputBytes != 4u) {
        off += ((headBlock * dimPad * 2u + 31u) & ~31u);
    }
    l.prodOff = off;
    off += ((headBlock * dimPad * 4u + 31u) & ~31u);
    l.qiStageOff = off;
    if (inputBytes != 4u) {
        off += ((QI_STAGE_ROWS * dimPad * 2u + 31u) & ~31u);
    }
    l.qiFOff = off;
    off += ((nidx * dimPad * 4u + 31u) & ~31u);
    l.kiTOff = off;
    off += ((dimPad * (inputBytes == 4u ? 4u : 2u) + 31u) & ~31u);
    l.kiFOff = off;
    off += ((dimPad * 4u + 31u) & ~31u);
    // keyIndex 行缓存：一次载入全部 key 的 ki，相似度与梯度两遍都走 UB，省掉 2*vis 次 DMA/同步
    l.kiCacheOff = off;
    off += ((kiCacheRows * dimPad * 4u + 31u) & ~31u);
    l.kiCacheTOff = off;
    if (inputBytes != 4u) {
        off += ((kiCacheRows * dimPad * 2u + 31u) & ~31u);
    }
    l.scOff = off;
    off += ((visPad * n1Pad * 4u + 31u) & ~31u);
    l.uOff = off;
    off += ((visPad * nidxPad * 4u + 31u) & ~31u);
    l.tgtOff = off;
    off += ((visPad * 4u + 31u) & ~31u);
    l.shOff = off;
    off += ((visPad * 4u + 31u) & ~31u);
    l.lossAccOff = off;
    off += ((visPad * 4u + 31u) & ~31u);
    l.lzOff = off;
    off += ((64u * 4u + 31u) & ~31u);
    l.predOff = off;
    off += ((visPad * 4u + 31u) & ~31u);
    l.delOff = off;
    off += ((visPad * 4u + 31u) & ~31u);
    l.dkOff = off;
    off += ((dkRows * dimPad * 4u + 31u) & ~31u);
    l.dkRowOff = off;
    off += ((dimPad * 4u + 31u) & ~31u);
    l.dqOff = off;
    off += ((nidx * dimPad * 4u + 31u) & ~31u);
    l.maxOff = off;
    off += ((n1Pad * 4u + 63u) & ~63u);
    l.dblkOff = off;
    off += ((visPad * 8u * 4u + 63u) & ~63u);
    l.mblkOff = off;
    off += ((nidx * 8u * 4u + 63u) & ~63u);
    l.partOff = off;
    off += ((partElems * 4u + 31u) & ~31u);
    l.wRawOff = off;
    off += ((nidx * weightBytes + 63u) & ~63u);
    l.wOff = off;
    off += ((nidx * 4u + 63u) & ~63u);
    l.wbOff = off;
    off += ((nidx * 8u * 4u + 63u) & ~63u);
    l.dwOff = off;
    off += ((nidx * 4u + 63u) & ~63u);
    l.dwRowOff = off;
    off += ((nidx * 4u + 63u) & ~63u);
    l.tmpOff = off;
    off += ((8u * 64u * 4u + 63u) & ~63u);
    l.totalBytes = off;
    return l;
}
