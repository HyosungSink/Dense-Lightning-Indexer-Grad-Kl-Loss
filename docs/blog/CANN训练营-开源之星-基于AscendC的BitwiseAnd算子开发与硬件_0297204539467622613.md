# 【CANN训练营】+开源之星+基于AscendC的BitwiseAnd算子开发与硬件优化实践

- topicId: 0297204539467622613
- source: https://www.hiascend.com/developer/blog/details/0297204539467622613
- section: Ascend C
- createTime: 20260123083108

一、引言：位运算算子在深度学习与通用计算中的价值

 

位运算（Bitwise Operation）是计算机底层最基础、最高效的操作之一，在深度学习推理优化、特征掩码生成、量化模型压缩、稀疏计算、加密与哈希等场景中扮演着关键角色。其中，BitwiseAnd（按位与） 作为最常用的位运算算子，通过对两个输入张量的每一位执行逻辑与操作，实现掩码筛选、特征提取、数据压缩等功能，是构建高效计算图的基础组件。

 

昇腾AI芯片凭借高算力、高带宽的硬件特性，不仅适用于深度学习浮点计算，也能通过AscendC编程框架高效支撑位运算等通用计算任务。本文基于提供的BitwiseAnd算子代码，从设计思路、核心实现、精度适配、硬件优化等维度，系统阐述AscendC平台下BitwiseAnd算子的开发实践，为昇腾平台上位运算类算子的开发提供标准化范式。

 

二、BitwiseAnd算子的数学原理与设计约束

 

1. 核心数学定义

 

BitwiseAnd算子的数学定义极为简洁：对两个输入张量 X1 、 X2 的每一位执行逻辑与操作，输出张量 Y 的每一位满足：

 Y[i] = X1[i] & X2[i] （按位与）

 

其中， & 为C/C++中的按位与运算符，适用于所有整数类型（int8/uint8、int16/uint16、int32/uint32等），每一位的运算规则为：

 

- 1 & 1 = 1

- 1 & 0 = 0

- 0 & 1 = 0

- 0 & 0 = 0

 

2. 设计核心约束

 

- 多整数类型兼容：需支持int8/uint8、int16/uint16、int32/uint32等主流整数类型，覆盖不同位宽的计算需求；

- 硬件指令适配：充分利用昇腾AI Core的位运算专用指令，避免软件模拟带来的性能损耗；

- 多核并行与流水线优化：通过数据分块与三级流水线，隐藏数据搬运延迟，提升计算吞吐量；

- 特殊类型处理：针对int32等特殊位宽类型，通过类型重解释（ReinterpretCast）适配硬件指令的位宽限制；

- 鲁棒性：支持任意尺寸张量，处理尾数据、边界数据时无越界、无计算错误。

 

三、BitwiseAnd算子的整体架构设计

 

基于AscendC框架的模板化+流水线+多核分块设计思想，BitwiseAnd算子以 KernelBitwiseAnd 模板类为核心，通过模板参数 TYPE_X1 实现多整数类型兼容，通过 TPipe 与 TQue 实现三级流水线，通过核间数据分块实现多核并行，整体架构简洁、高效、可扩展。

 

1. 核心类结构

 

cpp   

template <typename TYPE_X1>

class KernelBitwiseAnd {

public:

    __aicore__ inline void Init(...);  // 初始化：数据分块、内存绑定、缓存初始化

    __aicore__ inline void Process();  // 主流程：分块循环+流水线调度

private:

    __aicore__ inline void CopyIn(int32_t progress);   // 数据读入：全局→本地

    __aicore__ inline void CopyOut(int32_t progress);  // 结果回写：本地→全局

    __aicore__ inline void Compute(int32_t progress);  // 核心计算：按位与+特殊类型适配

    // 硬件资源与数据缓存

    AscendC::TPipe pipe;  // 流水线控制器

    AscendC::TQue<...> inQueueX1, inQueueX2, outQueueY;  // 输入/输出队列

    AscendC::GlobalTensor<TYPE_X1> x1Gm, x2Gm, yGm;  // 全局内存张量

    // 分块与临时参数

    uint64_t coreDataNum, tileNum, tileDataNum, tailDataNum;  // 主分块参数

    uint64_t tmpTileDataNum, tmpTailDataNum, tmpProcessDataNum;  // 特殊类型临时分块参数

    uint64_t processDataNum;  // 当前处理数据量

};

 

 

2. 执行流程

 

算子执行分为初始化→主流程循环两个阶段：

 

1. 初始化阶段（Init）：完成核间数据划分、全局/本地内存绑定、流水线缓存初始化，区分普通整数类型与int32特殊类型的分块参数；

2. 主流程阶段（Process）：按分块循环执行「CopyIn→Compute→CopyOut」三级流水线，先处理完整分块，再处理尾数据分块，针对int32类型动态调整临时分块参数，确保计算正确性。

 

四、核心模块实现细节

 

1. 初始化模块（Init）：数据分块与内存适配

 

初始化模块是算子的“资源配置中心”，核心解决核间数据均衡分配与多类型缓存初始化两大问题，同时为int32特殊类型预留临时分块参数。

 

（1）核间数据分块策略

 

昇腾AI芯片采用多核并行架构，需将全局数据均匀分配至每个AI Core，避免负载不均。代码中通过 GetBlockIdx() 获取核ID（coreId），采用大核块+小核块的分块策略：

 

- 尾核判断：通过 tailBlockNum 区分尾核与非尾核，尾核处理 bigCoreDataNum 数据，非尾核处理 smallCoreDataNum 数据，解决数据总量无法被核数整除的问题；

- 全局地址计算：根据核ID计算当前核的全局数据起始地址（ globalBufferIndex ），非尾核需调整地址偏移，避免数据越界；

- 分块参数设置：为每个核分配主分块参数（ coreDataNum 、 tileNum 、 tileDataNum 、 tailDataNum ）与int32专用临时分块参数（ tmpTileDataNum 、 tmpTailDataNum ），为后续循环执行提供参数支撑；

- 合法性校验：通过 ASSERT 校验核数合法性，避免运行时核数为0的错误。

 

（2）多类型内存绑定与缓存初始化

 

- 全局内存绑定：通过 GlobalTensor::SetGlobalBuffer 将输入 x1 、 x2 与输出 y 绑定至全局内存，指定数据起始地址与长度，建立全局内存与本地内存的映射关系；

- 流水线缓存初始化：输入队列（ inQueueX1 、 inQueueX2 ）、输出队列（ outQueueY ）采用 BUFFER_NUM=2 的双缓冲设计，实现数据拷贝与计算的并行重叠，缓存大小按 tileDataNum * sizeof(TYPE_X1) 分配，适配不同类型的位宽差异；

- 无额外临时缓存：BitwiseAnd计算逻辑简单，无需额外临时计算缓存，仅需输入/输出队列，最大化节省本地内存（L1/L2）资源。

 

2. 数据拷贝模块（CopyIn/CopyOut）：流水线数据搬运

 

数据拷贝模块是流水线的“数据通道”，负责在全局内存（GM）与本地内存（LM）之间高效搬运数据，通过 TQue 实现队列管理，隐藏数据搬运延迟。

 

（1）CopyIn：全局→本地数据读入

 

- 张量分配：通过 inQueueX1.AllocTensor<TYPE_X1>() 、 inQueueX2.AllocTensor<TYPE_X1>() 从输入队列分配本地张量，避免内存重复申请；

- 数据拷贝：调用 AscendC::DataCopy 将全局内存中的 x1 、 x2 数据拷贝至本地张量，按分块进度（ progress ）计算全局地址偏移，支持尾数据的非对齐拷贝；

- 队列入队：将拷贝完成的本地张量加入输入队列，供计算模块使用，实现数据的异步传递。

 

（2）CopyOut：本地→全局结果回写

 

- 张量出队：通过 outQueueY.DeQue<TYPE_X1>() 从输出队列取出计算完成的本地张量（int32类型需特殊处理，出队类型为int16）；

- 结果回写：调用 AscendC::DataCopy 将本地计算结果拷贝至全局内存的 y 地址，按分块进度寻址，int32类型通过内存布局兼容实现自动类型映射；

- 张量释放：通过 FreeTensor 释放本地张量，回收队列资源，避免内存泄漏，保证流水线循环执行的稳定性。

 

3. 计算模块（Compute）：按位与核心实现与特殊类型适配

 

计算模块是算子的“核心大脑”，实现BitwiseAnd的按位与逻辑，同时通过类型分支与重解释转换适配昇腾AI Core的位运算指令限制，是算子开发的关键环节。代码通过 constexpr if 对普通整数类型与int32类型进行分支处理。

 

（1）普通整数类型分支（非int32）：直接按位与

 

int8/uint8、int16/uint16等普通整数类型的位宽与昇腾AI Core的位运算指令位宽完全匹配，计算逻辑简单直接：

 

1. 张量出队：通过 inQueueX1.DeQue<TYPE_X1>() 、 inQueueX2.DeQue<TYPE_X1>() 从输入队列取出 x1 、 x2 的本地张量；

2. 输出张量分配：通过 outQueueY.AllocTensor<TYPE_X1>() 从输出队列分配结果本地张量；

3. 按位与计算：调用 AscendC::And 指令，对 x1Local 、 x2Local 的每一位执行按位与操作，结果存入 yLocal ，处理数据量为 processDataNum ；

4. 队列入队与释放：将结果张量加入输出队列，释放输入队列的本地张量。

 

核心代码片段（普通类型）：

 

cpp   

AscendC::LocalTensor<TYPE_X1> yLocal = outQueueY.AllocTensor<TYPE_X1>();

AscendC::And(yLocal, x1Local, x2Local, this->processDataNum);

outQueueY.EnQue<TYPE_X1>(yLocal);

 

 

（2）int32特殊类型分支：重解释转换适配

 

int32类型的位宽（32位）与昇腾AI Core部分位运算指令的默认位宽（16位）不匹配，直接执行 And 指令会导致位宽错误，因此采用类型重解释（ReinterpretCast） 方案：

 

1. 张量出队：从输入队列取出 x1 、 x2 的int32类型本地张量；

2. 类型重解释：通过 template ReinterpretCast<int16_t>() 将int32张量按内存位模式重解释为int16张量，不改变底层二进制数据，仅调整类型视图，将32位数据拆分为两个16位单元；

3. 输出张量分配：从输出队列分配int16类型的结果张量（ yLocal ）；

4. 按位与计算：调用 AscendC::And 指令，对重解释后的int16张量执行按位与操作，处理数据量为 tmpProcessDataNum （适配16位位宽的分块参数）；

5. 队列入队与释放：将int16类型的结果张量加入输出队列，释放输入队列的本地张量；

6. 内存布局兼容：int16类型的结果张量在内存中按连续32位存储，与int32类型的内存布局完全兼容，CopyOut时无需额外转换，直接回写至全局内存即可。

 

核心代码片段（int32类型）：

 

cpp   

AscendC::LocalTensor<int16_t> yLocal = outQueueY.AllocTensor<int16_t>();

AscendC::And(yLocal, x1Local.template ReinterpretCast<int16_t>(), x2Local.template ReinterpretCast<int16_t>(), this->tmpProcessDataNum);

outQueueY.EnQue<int16_t>(yLocal);

 

 

核心设计思路：通过类型重解释规避硬件指令的位宽限制，不改变底层二进制数据，保证按位与计算的正确性，同时充分利用硬件指令的高效性，避免软件模拟位运算的性能损耗。

 

（3）指令优化技巧

 

- 硬件指令优先：全部使用AscendC内置的 And 位运算专用指令，替代软件循环模拟，充分利用AI Core的位运算单元，提升指令并行度；

- 无临时缓存：计算过程无需额外临时缓存，输入/输出张量直接参与计算，最大化节省本地内存资源；

- 无分支计算：除类型分支外，计算流程无条件分支，避免AI Core的分支预测开销，保证指令流水线的流畅执行；

- 类型重解释零开销： ReinterpretCast 为编译期类型转换，无运行时开销，仅调整张量的类型视图，不改变底层数据。

 

4. 主流程模块（Process）：流水线循环调度与特殊类型适配

 

主流程模块是算子的“调度中心”，通过分块循环与流水线调度实现数据的完整处理，同时针对int32类型动态调整临时分块参数，确保计算正确性。

 

（1）分块循环逻辑

 

- 完整分块处理：循环次数为 loopCount = tileNum ，先处理 loopCount - 1 个完整分块，普通类型的处理数据量为 tileDataNum ，int32类型的临时处理数据量为 tmpTileDataNum ；

- 尾数据处理：最后处理1个尾数据分块，普通类型的处理数据量为 tailDataNum ，int32类型的临时处理数据量为 tmpTailDataNum ，解决数据总量无法被单块数据量整除的问题，避免数据越界；

- 参数动态调整：通过 constexpr if 判断类型，int32类型在循环前与尾数据处理前，分别将 tmpProcessDataNum 设置为 tmpTileDataNum 与 tmpTailDataNum ，适配16位位宽的分块参数。

 

（2）流水线调度

 

循环内依次调用 CopyIn → Compute → CopyOut ，结合双缓冲队列设计，实现数据拷贝与计算的并行重叠：

 

- 当第i块数据在 Compute 时，第i+1块数据可同时在 CopyIn ，第i-1块数据可同时在 CopyOut ；

- 流水线的并行度由 BUFFER_NUM=2 决定，平衡内存占用与并行效率，充分隐藏数据搬运的延迟；

- 整个流程无锁、无同步，最大化利用AI Core的算力资源。

 

五、关键技术优化与性能提升

 

1. 多类型兼容优化

 

- 模板化类型适配：通过 template <typename TYPE_X1> 实现int8/uint8、int16/uint16、int32/uint32等多整数类型兼容，无需为每种类型单独编写代码，提升代码复用性；

- 特殊类型重解释：针对int32类型的位宽限制，采用 ReinterpretCast 实现类型视图转换，不改变底层数据，保证按位与计算的正确性，同时兼容硬件指令；

- 分块参数分离：为int32类型单独设置临时分块参数（ tmpTileDataNum 、 tmpTailDataNum ），适配16位位宽的计算需求，避免分块越界。

 

2. 内存高效利用优化

 

- 双缓冲流水线：输入/输出队列采用 BUFFER_NUM=2 的双缓冲设计，实现数据拷贝与计算的并行，提升硬件利用率；

- 无临时缓存：计算过程无需额外临时计算缓存，仅需输入/输出队列，本地内存占用极低，适配昇腾AI Core的L1/L2本地内存容量限制；

- 全局内存连续访问：核间数据分块保证每个核的全局内存访问为连续地址，提升DDR带宽利用率，降低数据搬运延迟；

- 内存布局兼容：int32类型通过int16重解释计算，结果内存布局与int32完全兼容，CopyOut时无需额外转换，节省内存带宽。

 

3. 硬件架构适配优化

 

- 多核负载均衡：通过大核块+小核块的分块策略，保证每个核处理的数据量基本一致，避免多核负载不均导致的性能瓶颈；

- AI Core位运算指令适配：全部使用AscendC内置的 And 位运算专用指令，匹配昇腾芯片的位运算单元，充分发挥硬件算力，性能远超软件模拟；

- 无分支计算设计：计算流程除类型分支外无条件分支，避免AI Core的分支预测失败开销，保证指令流水线的满负荷运行；

​

- 核内SIMD并行：通过向量 And 指令实现单指令多数据（SIMD）并行，单AI Core可同时处理多个数据元素，提升计算吞吐量。

 

4. 鲁棒性与可扩展性优化

 

- 边界数据处理：通过尾数据分块处理，支持任意尺寸张量，避免数据越界与计算错误；

​

- 合法性校验：初始化阶段通过 ASSERT 校验核数合法性，避免运行时错误；

​

- 模块化设计：将Init、CopyIn、Compute、CopyOut拆分为独立模块，便于后续维护与优化，如新增整数类型、调整计算逻辑时仅需修改对应模块；

​

- 预留扩展接口：代码中通过模板与 constexpr if 预留了类型扩展空间，未来可轻松支持uint32、int64（需进一步重解释）等类型，提升算子的可扩展性。
