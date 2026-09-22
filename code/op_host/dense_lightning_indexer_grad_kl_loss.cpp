// Host侧Tiling实现：DenseLightningIndexerGradKlLoss
//
// 单核（blockDim=1）+ 纯 Vector（AIV_ONLY）路线：
//   * 目标 shape 规模很小，多核的头开销与硬同步开销远大于并行收益；
//   * Host 侧把 UB 布局与分块参数（hb / ib / kcRows）算好，Kernel 只做循环；
//   * 仅当 UB 放不下时按“头块 / key 块”降级，天然支持任意合法 shape（无白名单）。
#include <algorithm>
#include <cstdint>
#include <limits>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/dense_lightning_indexer_grad_kl_loss_tiling.h"
#include "../op_kernel/tiling_key_dense_lightning_indexer_grad_kl_loss.h"

namespace optiling {
namespace {
constexpr uint32_t UB_TOTAL_BYTES = 192u * 1024u;
constexpr uint32_t UB_RESERVE_BYTES = 12u * 1024u;
// 32B 步长参数为 uint8，最大 255 个 DataBlock。
constexpr uint32_t MAX_PAD_ELEMS = 255u * 8u;

inline uint32_t AlignUp(uint32_t value, uint32_t multiple) {
    return (value + multiple - 1u) / multiple * multiple;
}

inline uint32_t DimOf(const gert::Shape &shape, size_t index, uint32_t fallback) {
    if (index >= static_cast<size_t>(shape.GetDimNum())) {
        return fallback;
    }
    const int64_t value = shape.GetDim(index);
    if (value <= 0) {
        return fallback;
    }
    return static_cast<uint32_t>(value);
}

struct Choice {
    uint32_t hb = 0;
    uint32_t ib = 0;
    uint32_t kcRows = 1;
    uint32_t cost = 0;
    DenseLightningIndexerGradKlLossTilingData tiling{};
    bool valid = false;
};
}  // namespace

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubBytes = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    uint32_t ubBudget = static_cast<uint32_t>(ubBytes > UB_RESERVE_BYTES ? ubBytes - UB_RESERVE_BYTES : 0u);
    if (ubBudget > UB_TOTAL_BYTES - UB_RESERVE_BYTES) {
        ubBudget = UB_TOTAL_BYTES - UB_RESERVE_BYTES;
    }

    const gert::StorageShape *queryShape = context->GetInputShape(0);
    const gert::StorageShape *keyShape = context->GetInputShape(1);
    const gert::StorageShape *queryIndexShape = context->GetInputShape(2);
    const gert::StorageShape *keyIndexShape = context->GetInputShape(3);
    const gert::StorageShape *weightsShape = context->GetInputShape(4);
    if (queryShape == nullptr || keyShape == nullptr || queryIndexShape == nullptr ||
        keyIndexShape == nullptr || weightsShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape &qShape = queryShape->GetStorageShape();
    const gert::Shape &kShape = keyShape->GetStorageShape();
    const gert::Shape &qiShape = queryIndexShape->GetStorageShape();
    const gert::Shape &kiShape = keyIndexShape->GetStorageShape();
    const gert::Shape &wShape = weightsShape->GetStorageShape();

    const uint32_t batch = DimOf(qShape, 0, 1u);
    const uint32_t s1 = DimOf(qShape, 1, 1u);
    uint32_t n1 = DimOf(qShape, 2, 1u);
    const uint32_t dim = DimOf(qShape, 3, 1u);
    const uint32_t s2 = DimOf(kShape, 1, 1u);
    uint32_t nidx = DimOf(qiShape, 2, 1u);
    // 主注意力 key 头数与 query 头数不一致时按 key 的实际头数计算，保留通用回退路径。
    const uint32_t n1FromKey = DimOf(kShape, 2, n1);
    if (n1FromKey != n1) {
        n1 = n1FromKey;
    }
    if (nidx == 0u) {
        nidx = 1u;
    }
    const uint32_t nidxFromWeights = DimOf(wShape, 2, nidx);
    if (nidxFromWeights < nidx) {
        nidx = nidxFromWeights == 0u ? 1u : nidxFromWeights;
    }
    if (batch == 0u || s1 == 0u || s2 == 0u || n1 == 0u || nidx == 0u || dim == 0u) {
        return ge::GRAPH_FAILED;
    }

    auto queryDesc = context->GetInputDesc(0);
    auto weightsDesc = context->GetInputDesc(4);
    const ge::DataType queryDtype = queryDesc->GetDataType();
    const ge::DataType weightsDtype = weightsDesc != nullptr ? weightsDesc->GetDataType() : queryDtype;
    uint32_t inputBytes = static_cast<uint32_t>(ge::GetSizeByDataType(queryDtype));
    if (inputBytes != 2u && inputBytes != 4u) {
        return ge::GRAPH_FAILED;
    }
    uint32_t weightBytes = static_cast<uint32_t>(ge::GetSizeByDataType(weightsDtype));
    if (weightBytes != 2u && weightBytes != 4u) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t weightsFp32 = (weightsDtype == ge::DT_FLOAT) ? 1u : 0u;

    const uint32_t DT_QUERY = static_cast<uint32_t>(queryDtype);
    ASCENDC_TPL_SEL_PARAM(context, DT_QUERY);

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const float *scalePtr = attrs != nullptr ? attrs->GetFloat(0) : nullptr;
    const float scale = scalePtr != nullptr ? *scalePtr : 0.08838834764831845f;

    const uint32_t dimPad = AlignUp(dim, 32u / inputBytes);
    const uint32_t visPad = AlignUp(s2, 8u);
    if (dimPad > MAX_PAD_ELEMS || visPad > MAX_PAD_ELEMS || n1 > 255u || nidx > 255u) {
        return ge::GRAPH_FAILED;
    }

    // ---- 单核（AIV_ONLY）路径：搜索 UB 可行的 (hb, ib, kcRows) ----
    const uint32_t n1Pad = AlignUp(n1, 8u);
    const uint32_t nidxPad = AlignUp(nidx, 8u);
    const uint32_t keyRowBytes = AlignUp(static_cast<uint64_t>(n1) * dimPad * inputBytes, 32u);
    const uint32_t hbCands[5] = {n1, 128u, 64u, 32u, 16u};
    const uint32_t ibCands[5] = {128u, 64u, 32u, 16u, 8u};
    Choice best;
    for (uint32_t hi = 0; hi < 5u; ++hi) {
        const uint32_t hb = hbCands[hi] > n1 ? n1 : hbCands[hi];
        if (hb == 0u || (hi > 0u && hb == (hbCands[hi - 1u] > n1 ? n1 : hbCands[hi - 1u]))) {
            continue;  // 去重
        }
        for (uint32_t ii = 0; ii < 5u; ++ii) {
            const uint32_t ib = ibCands[ii] > nidxPad ? nidxPad : ibCands[ii];
            if (ib == 0u || (ii > 0u && ib == (ibCands[ii - 1u] > nidxPad ? nidxPad : ibCands[ii - 1u]))) {
                continue;
            }
            const DenseLightningIndexerGradKlLossTilingData probe =
                DliglComputeLayout(batch, s1, s2, n1, nidx, dim, hb, ib, 1u, inputBytes,
                                   weightBytes, scale, weightsFp32);
            if (probe.ubBytes > ubBudget) {
                continue;
            }
            uint32_t kcRows = 1u;
            if (keyRowBytes > 0u) {
                const uint32_t extra = (ubBudget - probe.ubBytes) / keyRowBytes;
                kcRows = std::min(s2, 1u + extra);
            }
            const DenseLightningIndexerGradKlLossTilingData full =
                DliglComputeLayout(batch, s1, s2, n1, nidx, dim, hb, ib, kcRows, inputBytes,
                                   weightBytes, scale, weightsFp32);
            if (full.ubBytes > ubBudget) {
                continue;
            }
            // 头块数越少向量指令越少；kcRows 越大 DMA 往返越少。
            const uint32_t cost = ((n1 + hb - 1u) / hb) * 5u + ((nidx + ib - 1u) / ib) * 10u +
                                  ((s2 + kcRows - 1u) / kcRows) * 2u;
            if (!best.valid || cost < best.cost || (cost == best.cost && kcRows > best.kcRows)) {
                best.hb = hb;
                best.ib = ib;
                best.kcRows = kcRows;
                best.cost = cost;
                best.tiling = full;
                best.valid = true;
            }
        }
    }
    if (!best.valid) {
        return ge::GRAPH_FAILED;
    }
    (void)n1Pad;

    auto *tiling = context->GetTilingData<DenseLightningIndexerGradKlLossTilingData>();
    *tiling = best.tiling;

    const uint32_t sysWorkspace = ascendcPlatform.GetLibApiWorkSpaceSize();
    if (sysWorkspace == std::numeric_limits<uint32_t>::max()) {
        return ge::GRAPH_FAILED;
    }
    size_t *workspace = context->GetWorkspaceSizes(1);
    // 单核实现不需要用户 workspace；保留系统保留区即可。
    workspace[0] = static_cast<size_t>(sysWorkspace);

    context->SetBlockDim(1);
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class DenseLightningIndexerGradKlLoss : public OpDef {
public:
    explicit DenseLightningIndexerGradKlLoss(const char *name) : OpDef(name) {
        // dtype 组合：fp16/bf16/fp32 输入，weights 允许 fp16/bf16/fp32，loss 恒为 float32。
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("query_index")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("key_index")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("weights")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("d_query_index")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("d_key_index")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("d_weights")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("loss")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("scale_value").AttrType(OPTIONAL).Float(0.08838834764831845f);
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(DenseLightningIndexerGradKlLoss);
}  // namespace ops
