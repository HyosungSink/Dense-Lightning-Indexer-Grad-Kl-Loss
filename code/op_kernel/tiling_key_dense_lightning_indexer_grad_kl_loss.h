// TilingKey模板定义的头文件
#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

// 计算路径只依赖 query 的数据类型；weights 的差异通过 tiling 字段在 Kernel 内分支。
ASCENDC_TPL_ARGS_DECL(DenseLightningIndexerGradKlLoss,
    ASCENDC_TPL_DATATYPE_DECL(DT_QUERY, C_DT_FLOAT, C_DT_FLOAT16),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_QUERY, C_DT_FLOAT),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_QUERY, C_DT_FLOAT16),
    ),
);
