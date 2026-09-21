// Host侧Tiling实现：把 (B,S1,S2,N1,Nidx1,D) 映射成行分派 + UB 切分 + workspace 布局。
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
constexpr uint64_t WS_LIMIT_BYTES = 384ull * 1024ull * 1024ull;
// 单核工作量阈值：低于该规模时同步开销大于并行收益，直接用单核。
constexpr uint64_t SINGLE_CORE_MACS = 120000ull;
// 32B 步长参数为 uint8，最大 255 个 DataBlock。
constexpr uint32_t MAX_PAD_ELEMS = 255u * 8u;

inline uint32_t AlignUp(uint32_t value, uint32_t multiple) {
    return (value + multiple - 1u) / multiple * multiple;
}

inline uint32_t CeilDiv(uint32_t a, uint32_t b) { return b == 0u ? 0u : (a + b - 1u) / b; }

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
    uint32_t headBlock = 1;
    uint32_t dkRows = 0;
    DliglUbLayout layout{};
    bool valid = false;
};
}  // namespace

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    if (aivNum < 1) {
        aivNum = 1;
    }
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

    uint32_t batch = DimOf(qShape, 0, 1u);
    uint32_t s1 = DimOf(qShape, 1, 1u);
    uint32_t n1 = DimOf(qShape, 2, 1u);
    uint32_t dim = DimOf(qShape, 3, 1u);
    uint32_t s2 = DimOf(kShape, 1, 1u);
    uint32_t nidx = DimOf(qiShape, 2, DimOf(kiShape, 2, 1u));
    if (DimOf(kShape, 2, n1) != n1) {
        // 主注意力 key 头数与 query 头数不一致时按 key 的实际头数计算，保留通用 fallback。
        n1 = DimOf(kShape, 2, n1);
    }
    if (nidx == 0u) {
        nidx = 1u;
    }
    const uint32_t nidxFromWeights = DimOf(wShape, 2, nidx);
    if (nidxFromWeights < nidx) {
        nidx = nidxFromWeights == 0u ? 1u : nidxFromWeights;
    }

    auto queryDesc = context->GetInputDesc(0);
    auto weightsDesc = context->GetInputDesc(4);
    const ge::DataType queryDtype = queryDesc->GetDataType();
    const ge::DataType weightsDtype = weightsDesc != nullptr ? weightsDesc->GetDataType() : queryDtype;
    uint32_t inputBytes = static_cast<uint32_t>(ge::GetSizeByDataType(queryDtype));
    if (inputBytes == 0u) {
        inputBytes = 2u;
    }
    uint32_t weightBytes = static_cast<uint32_t>(ge::GetSizeByDataType(weightsDtype));
    if (weightBytes == 0u) {
        weightBytes = 2u;
    }
    const uint32_t weightsFp32 = (weightsDtype == ge::DT_FLOAT) ? 1u : 0u;

    const uint32_t DT_QUERY = static_cast<uint32_t>(queryDtype);
    ASCENDC_TPL_SEL_PARAM(context, DT_QUERY);

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const float *scalePtr = attrs != nullptr ? attrs->GetFloat(0) : nullptr;
    const float scale = scalePtr != nullptr ? *scalePtr : 0.08838834764831845f;

    const uint32_t padUnit = std::max(8u, 32u / inputBytes);
    const uint32_t dimPad = AlignUp(dim, padUnit);
    // bf16 输入：向量乘加不支持 bf16，需要一块 T 暂存做精度转换
    // 该平台向量指令不支持 bf16 乘加、也不支持 bf16->half 转换，bf16 组合不下发。
    const uint32_t stageElems = 0u;
    const uint32_t visPad = std::max(AlignUp(s2, 8u), 8u);
    if (dimPad > MAX_PAD_ELEMS || visPad > MAX_PAD_ELEMS || n1 > 255u || nidx > 255u) {
        return ge::GRAPH_FAILED;
    }

    // 工作量估计：主注意力 + indexer 相似度 + dq + dk（dq/dk 在梯度阶段复用相似度）。
    const uint64_t macs = static_cast<uint64_t>(batch) * s1 *
                          (static_cast<uint64_t>(n1) + 3ull * nidx) * s2 * dim;
    const uint64_t rowsTotal = static_cast<uint64_t>(batch) * s1;

    uint32_t rowsPerTask = 1u;
    // blocksPerBatch 必须与 rowsPerTask 自洽：同一 batch 有多个任务时必须走 partial + 归约，
    // 否则各任务会各自写整段 dk 而互相覆盖。
    uint32_t blocksPerBatch = CeilDiv(s1, rowsPerTask);
    uint32_t taskCount = batch * blocksPerBatch;
    if (macs > SINGLE_CORE_MACS && rowsTotal > 1u) {
        uint32_t cores = static_cast<uint32_t>(std::min<uint64_t>(static_cast<uint64_t>(aivNum), rowsTotal));
        uint32_t perBatchBlocks = cores / batch;
        if (perBatchBlocks == 0u) {
            perBatchBlocks = 1u;
        }
        rowsPerTask = CeilDiv(s1, perBatchBlocks);
        if (rowsPerTask == 0u) {
            rowsPerTask = 1u;
        }
        blocksPerBatch = CeilDiv(s1, rowsPerTask);
        taskCount = batch * blocksPerBatch;
    }

    // ---- UB 切分搜索 ----
    const uint32_t n1Pad = AlignUp(n1, 8u);
    const uint32_t nidxPad = AlignUp(nidx, 8u);
    Choice choice;
    const uint32_t headCandidates[] = {32u, 16u, 8u, 4u, 2u, 1u};
    for (uint32_t hbRaw : headCandidates) {
        const uint32_t hb = std::min(hbRaw, n1);
        const DliglUbLayout base =
            DliglComputeUbLayout(hb, n1, nidx, dimPad, visPad, n1Pad, nidxPad, 0u, inputBytes,
                                 weightBytes, stageElems);
        if (base.totalBytes > ubBudget) {
            continue;
        }
        const uint32_t remain = ubBudget - base.totalBytes;
        const uint32_t dkRows = std::min(s2, remain / (dimPad * 4u));
        const DliglUbLayout probe =
            DliglComputeUbLayout(hb, n1, nidx, dimPad, visPad, n1Pad, nidxPad, dkRows, inputBytes,
                                 weightBytes, stageElems);
        if (probe.totalBytes <= ubBudget) {
            choice.headBlock = hb;
            choice.dkRows = dkRows;
            choice.valid = true;
            break;
        }
    }
    if (!choice.valid) {
        return ge::GRAPH_FAILED;
    }

    uint32_t blockDim = std::min(taskCount, static_cast<uint32_t>(aivNum));
    if (blockDim == 0u) {
        blockDim = 1u;
    }
    const uint32_t splitDk = (blocksPerBatch > 1u) ? 1u : 0u;
    const uint32_t needPartial = (splitDk != 0u || choice.dkRows < s2) ? 1u : 0u;
    const uint64_t dkPartialElems =
        needPartial != 0u ? static_cast<uint64_t>(s2) * dimPad : 0ull;

    // 与系统 workspace（SyncAll 标志/屏障）保持安全距离：所有用户数据都从 guard 之后开始
    constexpr uint64_t WS_GUARD_FLOATS = 16384u;
    uint64_t userFloats = WS_GUARD_FLOATS;
    const uint64_t dkOffset = userFloats;
    userFloats += dkPartialElems * taskCount;
    const uint64_t lossOffset = userFloats;
    const uint64_t lossSlots = (taskCount < 8u) ? 8ull : (static_cast<uint64_t>(taskCount) + 7ull) / 8ull * 8ull;
    if (blockDim > 1u) {
        userFloats += lossSlots;
    }
    const uint64_t userBytes = userFloats * 4ull;
    if (userBytes > WS_LIMIT_BYTES) {
        return ge::GRAPH_FAILED;
    }
    size_t *workspace = context->GetWorkspaceSizes(1);
    const uint32_t sysWorkspace = ascendcPlatform.GetLibApiWorkSpaceSize();
    if (sysWorkspace == std::numeric_limits<uint32_t>::max()) {
        return ge::GRAPH_FAILED;
    }
    workspace[0] = static_cast<size_t>(userBytes) + static_cast<size_t>(sysWorkspace);

    auto *tiling = context->GetTilingData<DenseLightningIndexerGradKlLossTilingData>();
    tiling->batch = batch;
    tiling->s1 = s1;
    tiling->s2 = s2;
    tiling->n1 = n1;
    tiling->nidx = nidx;
    tiling->dim = dim;
    tiling->dimPad = dimPad;
    tiling->visPad = visPad;
    tiling->causal = 1u;
    tiling->rowsPerTask = rowsPerTask;
    tiling->blocksPerBatch = blocksPerBatch;
    tiling->taskCount = taskCount;
    tiling->n1Pad = n1Pad;
    tiling->nidxPad = nidxPad;
    tiling->headBlock = choice.headBlock;
    tiling->dkRows = choice.dkRows;
    tiling->splitDk = splitDk;
    tiling->dkPartialElems = static_cast<uint32_t>(dkPartialElems);
    tiling->dkOffset = static_cast<uint32_t>(dkOffset);
    tiling->lossOffset = static_cast<uint32_t>(lossOffset);
    tiling->weightsFp32 = weightsFp32;
    tiling->stageElems = stageElems;
    tiling->blockDimUsed = blockDim;
    tiling->scale = scale;

    context->SetBlockDim(blockDim);
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
