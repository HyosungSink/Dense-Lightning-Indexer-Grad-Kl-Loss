// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct DenseLightningIndexerGradKlLossTilingData {
    uint32_t batch;
    uint32_t s1;
    uint32_t s2;
    uint32_t n1;
    uint32_t nidx1;
    uint32_t d;
    uint32_t coreNum;
    uint32_t bigCoreCount;
    uint32_t rowsPerCoreBase;
    float scaleValue;
};
