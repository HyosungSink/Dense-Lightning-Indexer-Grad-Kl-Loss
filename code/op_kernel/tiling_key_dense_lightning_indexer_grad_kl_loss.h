// TilingKey模板定义的头文件
#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(DenseLightningIndexerGradKlLoss,
    ASCENDC_TPL_DATATYPE_DECL(DT_QUERY, C_DT_FLOAT, C_DT_FLOAT16),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_QUERY, C_DT_FLOAT, C_DT_FLOAT16),
    ),
);