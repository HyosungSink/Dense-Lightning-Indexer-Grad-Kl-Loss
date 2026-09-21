// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/dense_lightning_indexer_grad_kl_loss_tiling.h"
#include "../op_kernel/tiling_key_dense_lightning_indexer_grad_kl_loss.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 示例: 获取平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
        // 示例: 获取算子输入数组信息
        const gert::Tensor *tensor_query = context->GetRequiredInputTensor(0);
        const gert::Tensor *tensor_key = context->GetRequiredInputTensor(1);
        const gert::Tensor *tensor_query_index = context->GetRequiredInputTensor(2);
        const gert::Tensor *tensor_key_index = context->GetRequiredInputTensor(3);
        const gert::Tensor *tensor_weights = context->GetRequiredInputTensor(4);
        ge::DataType dtype_query = tensor_query->GetDataType(); // 获取数据类型
        int dtype_size_query = ge::GetSizeByDataType(dtype_query); // 获取数据类型的字长
        uint32_t length_query = tensor_query->GetShapeSize(); // 获取元素个数
        uint32_t size_query = tensor_query->GetSize(); // 获取内存大小
        // 示例: 获取算子输入属性
        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        const float *attr_scale_value = attrs->GetFloat(0);
        // 示例: 配置tiling key, 从而实现kernel侧不同数据类型/算法的区分
        uint32_t DT_QUERY = static_cast<uint32_t>(dtype_query);
        ASCENDC_TPL_SEL_PARAM(context, DT_QUERY);
        // 示例: 计算tiling方案并填充tiling结构体
        DenseLightningIndexerGradKlLossTilingData *tiling = context->GetTilingData<DenseLightningIndexerGradKlLossTilingData>();
        tiling->length = length_query;
        // 配置启动核数
        context->SetBlockDim(num_cores_aiv);
        // 配置workspace大小
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
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
            this->Input("query")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("key")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("query_index")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("key_index")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("weights")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("d_query_index")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("d_key_index")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("d_weights")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("loss")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("scale_value").AttrType(OPTIONAL).Float(0.0884);
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(DenseLightningIndexerGradKlLoss);
}  // namespace ops
