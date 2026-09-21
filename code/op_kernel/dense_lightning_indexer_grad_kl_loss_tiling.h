// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

// 行分派与 UB 切分参数全部由 Host 计算，Kernel 只做通用循环。
struct DenseLightningIndexerGradKlLossTilingData {
    uint32_t batch;           // B
    uint32_t s1;              // S1 (query 序列长度)
    uint32_t s2;              // S2 (key 序列长度)
    uint32_t n1;              // N1 主注意力头数
    uint32_t nidx;            // Nidx1 indexer 头数
    uint32_t dim;             // D
    uint32_t dimPad;          // D 按 32B 对齐后的 fp32 行宽
    uint32_t visPad;          // 可见 key 数按 32B 对齐后的 fp32 行宽
    uint32_t causal;          // 1: rightDownCausal 掩码, 0: 全可见
    uint32_t rowsPerTask;     // 每个任务处理的 query 行数
    uint32_t blocksPerBatch;  // 每个 batch 的任务数
    uint32_t taskCount;       // 任务总数
    uint32_t headBlock;       // 主注意力分数缓存的头块大小
    uint32_t keyChunk;        // 一次处理的 key 数
    uint32_t simPitch;        // indexer 相似度分块的行宽（对齐后）
    uint32_t dkRows;          // UB 中 dk 累加器可容纳的 key 行数
    uint32_t storeSim;        // 1: indexer 相似度常驻 UB
    uint32_t splitDk;         // 1: 需要跨任务归约 dKeyIndex
    uint32_t dkPartialElems;  // 每个任务的 dk 部分和元素数（0 表示不需要 workspace）
    uint32_t lossOffset;      // workspace 中 loss 部分和的浮点偏移
    uint32_t weightsFp32;     // 1: weights 为 float32
    uint32_t blockDimUsed;    // 启动核数
    float scale;              // 注意力缩放系数
};

// Kernel 侧包含 kernel_operator.h 后 __aicore__ 已定义；Host 侧退化为普通 inline。
#ifndef __aicore__
#define DLIGL_UB_FN inline
#else
#define DLIGL_UB_FN __aicore__ inline
#endif

// UB 布局：Host 用它挑选可行切分，Kernel 用它取每个缓冲区的字节偏移。
struct DliglUbLayout {
    uint32_t qOff;
    uint32_t kTOff;
    uint32_t kFOff;
    uint32_t prodOff;
    uint32_t scoreOff;
    uint32_t simOff;
    uint32_t tgtOff;
    uint32_t shOff;
    uint32_t predOff;
    uint32_t delOff;
    uint32_t dkOff;
    uint32_t dkcOff;
    uint32_t qiOff;
    uint32_t qiTOff;
    uint32_t dqOff;
    uint32_t partOff;
    uint32_t bcastOff;
    uint32_t maxOff;
    uint32_t wRawOff;
    uint32_t wOff;
    uint32_t wbOff;
    uint32_t dwOff;
    uint32_t dwRowOff;
    uint32_t tmpOff;
    uint32_t totalBytes;
};

DLIGL_UB_FN DliglUbLayout DliglComputeUbLayout(uint32_t headBlock, uint32_t keyChunk, uint32_t nidx,
                                               uint32_t dimPad, uint32_t visPad, uint32_t simCols,
                                               uint32_t dkRows, uint32_t inputBytes,
                                               uint32_t weightBytes) {
    DliglUbLayout l;
    uint32_t off = 0;
    l.qOff = off;
    off += ((headBlock * dimPad * 4u + 31u) & ~31u);
    l.kTOff = off;
    if (inputBytes != 4u) {
        off += ((keyChunk * dimPad * inputBytes + 31u) & ~31u);
    }
    l.kFOff = off;
    off += ((keyChunk * dimPad * 4u + 31u) & ~31u);
    l.prodOff = off;
    off += ((keyChunk * dimPad * 4u + 31u) & ~31u);
    l.scoreOff = off;
    off += ((headBlock * visPad * 4u + 31u) & ~31u);
    l.simOff = off;
    off += ((nidx * simCols * 4u + 31u) & ~31u);
    l.tgtOff = off;
    off += ((visPad * 4u + 31u) & ~31u);
    l.shOff = off;
    off += ((visPad * 4u + 31u) & ~31u);
    l.predOff = off;
    off += ((visPad * 4u + 31u) & ~31u);
    l.delOff = off;
    off += ((visPad * 4u + 31u) & ~31u);
    l.dkOff = off;
    off += ((dkRows * dimPad * 4u + 31u) & ~31u);
    l.dkcOff = off;
    off += ((keyChunk * dimPad * 4u + 31u) & ~31u);
    l.qiOff = off;
    off += ((dimPad * 4u + 31u) & ~31u);
    l.qiTOff = off;
    if (inputBytes != 4u) {
        off += ((dimPad * inputBytes + 31u) & ~31u);
    }
    l.dqOff = off;
    off += ((nidx * dimPad * 4u + 31u) & ~31u);
    l.partOff = off;
    off += ((64u * 8u * 4u + 31u) & ~31u);
    l.bcastOff = off;
    off += (((headBlock > keyChunk ? headBlock : keyChunk) * 8u * 4u + 63u) & ~63u);
    l.maxOff = off;
    off += ((64u * 8u * 4u + 63u) & ~63u);
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
