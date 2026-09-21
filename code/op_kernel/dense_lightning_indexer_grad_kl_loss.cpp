// Kernel侧核函数实现
#include "kernel_operator.h"

#include "dense_lightning_indexer_grad_kl_loss_tiling.h"
#include "tiling_key_dense_lightning_indexer_grad_kl_loss.h"

template <class DT_QUERY>
class KernelDenseLightningIndexerGradKlLoss {
public:
    __aicore__ inline KernelDenseLightningIndexerGradKlLoss() {}
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR query_index, GM_ADDR key_index, GM_ADDR weights, GM_ADDR d_query_index, GM_ADDR d_key_index, GM_ADDR d_weights, GM_ADDR loss, uint32_t length) {

    }
    __aicore__ inline void Process() {

    }
private:

};

template <typename DT_QUERY>
 __global__ __aicore__ void dense_lightning_indexer_grad_kl_loss(GM_ADDR query, GM_ADDR key, GM_ADDR query_index, GM_ADDR key_index, GM_ADDR weights, GM_ADDR d_query_index, GM_ADDR d_key_index, GM_ADDR d_weights, GM_ADDR loss, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(DenseLightningIndexerGradKlLossTilingData);
    GET_TILING_DATA_WITH_STRUCT(DenseLightningIndexerGradKlLossTilingData, tiling_data, tiling);
    KernelDenseLightningIndexerGradKlLoss<DT_QUERY> op;
    op.Init(query, key, query_index, key_index, weights, d_query_index, d_key_index, d_weights, loss, tiling_data.length);
    op.Process();
}
