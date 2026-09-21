// Host侧Tiling实现
#include <vector>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/dense_lightning_indexer_grad_kl_loss_tiling.h"
#include "../op_kernel/tiling_key_dense_lightning_indexer_grad_kl_loss.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t numCoresAiv = static_cast<uint32_t>(platform.GetCoreNumAiv());

        const gert::Shape *queryShape = context->GetInputShape(0);
        const gert::Shape *keyShape = context->GetInputShape(1);
        const gert::Shape *queryIndexShape = context->GetInputShape(2);
        const gert::Shape *weightsShape = context->GetInputShape(4);

        uint32_t batch = static_cast<uint32_t>(queryShape->GetDim(0));
        uint32_t s1 = static_cast<uint32_t>(queryShape->GetDim(1));
        uint32_t n1 = static_cast<uint32_t>(queryShape->GetDim(2));
        uint32_t d = static_cast<uint32_t>(queryShape->GetDim(3));
        uint32_t s2 = static_cast<uint32_t>(keyShape->GetDim(1));
        uint32_t nidx1 = static_cast<uint32_t>(queryIndexShape->GetDim(2));
        (void)weightsShape;

        uint64_t totalRows = static_cast<uint64_t>(batch) * static_cast<uint64_t>(s1);
        uint32_t coreNum = static_cast<uint32_t>(
            totalRows < numCoresAiv ? totalRows : numCoresAiv);
        if (coreNum == 0) {
            coreNum = 1;
        }
        uint32_t rowsPerCoreBase = static_cast<uint32_t>(totalRows / coreNum);
        uint32_t bigCoreCount = static_cast<uint32_t>(totalRows % coreNum);

        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        const float *attrScaleValue = attrs->GetFloat(0);

        DenseLightningIndexerGradKlLossTilingData *tiling =
            context->GetTilingData<DenseLightningIndexerGradKlLossTilingData>();
        tiling->batch = batch;
        tiling->s1 = s1;
        tiling->s2 = s2;
        tiling->n1 = n1;
        tiling->nidx1 = nidx1;
        tiling->d = d;
        tiling->coreNum = coreNum;
        tiling->bigCoreCount = bigCoreCount;
        tiling->rowsPerCoreBase = rowsPerCoreBase;
        tiling->scaleValue = attrScaleValue != nullptr ? *attrScaleValue : 0.0884f;

        uint32_t dtypeQuery = static_cast<uint32_t>(context->GetInputDesc(0)->GetDataType());
        uint32_t dtypeWeight = static_cast<uint32_t>(context->GetInputDesc(4)->GetDataType());
        ASCENDC_TPL_SEL_PARAM(context, dtypeQuery, dtypeWeight);

        context->SetBlockDim(coreNum);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *queryIndexShape = context->GetInputShape(2);
        const gert::Shape *keyIndexShape = context->GetInputShape(3);
        const gert::Shape *weightsShape = context->GetInputShape(4);
        gert::Shape *dQueryIndexShape = context->GetOutputShape(0);
        gert::Shape *dKeyIndexShape = context->GetOutputShape(1);
        gert::Shape *dWeightsShape = context->GetOutputShape(2);
        gert::Shape *lossShape = context->GetOutputShape(3);

        *dQueryIndexShape = *queryIndexShape;
        *dKeyIndexShape = *keyIndexShape;
        *dWeightsShape = *weightsShape;
        lossShape->SetDimNum(1);
        lossShape->SetDim(0, 1);
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, context->GetInputDataType(2));
        context->SetOutputDataType(1, context->GetInputDataType(3));
        context->SetOutputDataType(2, context->GetInputDataType(4));
        context->SetOutputDataType(3, ge::DT_FLOAT);
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class DenseLightningIndexerGradKlLoss : public OpDef {
    public:
        explicit DenseLightningIndexerGradKlLoss(const char *name) : OpDef(name) {
            // Five zipped dtype instances (matched by position across every
            // Input/Output): (query,weight) in
            // {(f16,f16), (f16,f32), (bf16,bf16), (bf16,f32), (f32,f32)}.
            const std::vector<ge::DataType> mainDtypes = {
                ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT};
            const std::vector<ge::DataType> weightDtypes = {
                ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT};
            const std::vector<ge::DataType> lossDtypes = {
                ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
            const std::vector<ge::Format> allFormats(mainDtypes.size(), ge::FORMAT_ND);

            this->Input("query")
                .ParamType(REQUIRED)
                .DataType(mainDtypes)
                .Format(allFormats);
            this->Input("key")
                .ParamType(REQUIRED)
                .DataType(mainDtypes)
                .Format(allFormats);
            this->Input("query_index")
                .ParamType(REQUIRED)
                .DataType(mainDtypes)
                .Format(allFormats);
            this->Input("key_index")
                .ParamType(REQUIRED)
                .DataType(mainDtypes)
                .Format(allFormats);
            this->Input("weights")
                .ParamType(REQUIRED)
                .DataType(weightDtypes)
                .Format(allFormats);
            this->Output("d_query_index")
                .ParamType(REQUIRED)
                .DataType(mainDtypes)
                .Format(allFormats);
            this->Output("d_key_index")
                .ParamType(REQUIRED)
                .DataType(mainDtypes)
                .Format(allFormats);
            this->Output("d_weights")
                .ParamType(REQUIRED)
                .DataType(weightDtypes)
                .Format(allFormats);
            this->Output("loss")
                .ParamType(REQUIRED)
                .DataType(lossDtypes)
                .Format(allFormats);
            this->Attr("scale_value").AttrType(OPTIONAL).Float(0.0884);
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(DenseLightningIndexerGradKlLoss);
}  // namespace ops
