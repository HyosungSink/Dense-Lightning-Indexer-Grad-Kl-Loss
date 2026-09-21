# Ascend C算子优化实用技巧04——Tiling优化

- techColumnCode: 20240920-1
- source: https://www.hiascend.com/developer/techArticles/20240920-1

## 01 简介

Ascend C是CANN针对算子开发场景推出的编程语言，原生支持C和C++标准规范，兼具开发效率和运行性能。使用Ascend C，开发者可以基于昇腾AI硬件，高效的实现自定义的创新算法。

目前已经有越来越多的开发者使用Ascend C，我们将通过几期“Ascend C算子性能优化”专题分享，围绕开发者最为关心的算子性能优化环节，介绍Ascend C算子常用的优化技巧，帮助开发者自主构建出更优性能的算子。专题内容将围绕流水优化、搬运优化、内存优化、API使用优化以及Tiling优化等优化技巧，从方案讲解、优化案例、性能对比等多角度展开介绍。前期内容回顾：

·[《Ascend
C算子性能优化实用技巧01——流水优化》](https://www.hiascend.com/zh/developer/techArticles/20240819-1)

·[《Ascend
C算子性能优化实用技巧02——内存优化》](https://www.hiascend.com/zh/developer/techArticles/20240823-1)

·[《Ascend
C算子性能优化实用技巧03——搬运优化》](https://www.hiascend.com/zh/developer/techArticles/20240906-1)

下面进入第四期内容：Ascend
C Tiling优化，您将了解到以下优化技巧：

· 多核切分

· L2Cache切分

· 核间负载均衡

## 02 什么是Tiling

大多数情况下，AI Core内部的Unified Buffer，无法完整的容纳算子的输入与输出，需要每次搬运一部分输入进行计算然后搬出，再搬运下一部分输入进行计算，直到得到完整的最终结果，这个数据切分、分块计算的过程称之为**Tiling**，切分数据的算法称为**Tiling****算法或者Tiling****策略**。

## 03 多核切分

AI处理器上一般包括多个AI Core处理核心，为了实现多核并行，提升计算效率，需要将矩阵数据进行切分，分配到不同的核上进行处理。多核切分是最常见、最基本的Tiling策略。

通过SetBlockDim接口设置整个算子计算所用核数blockDim。

```
context->SetBlockDim(BLOCK_DIM);
```

blockDim规定了核函数将会在几个核上执行。例如，需要计算8M的数据，每个核上计算1M的数据，blockDim设置为8，但是为了充分利用硬件资源，一般将blockDim设置为硬件平台的核数，根据核数进行数据切分。

blockDim是逻辑核的概念，取值范围为[1,65535]。为了充分利用硬件资源，一般设置为物理核的核数或其倍数。对于耦合架构和分离架构，blockDim在运行时的意义和设置规则有一些区别：

· **耦合架构**：由于其Vector、Cube单元是集成在一起的，blockDim用于设置启动多个AICore核实例执行，不区分Vector、Cube。AI Core的核数可以通过GetCoreNumAiv或者GetCoreNumAic获取。

· **分离架构**

   · 针对仅包含Vector计算的算子，blockDim用于设置启动多少个Vector（AIV）实例执行，比如某款AI处理器上有40个Vector核，建议设置为40。

   ·
针对仅包含Cube计算的算子，blockDim用于设置启动多少个Cube（AIC）实例执行，比如某款AI处理器上有20个Cube核，建议设置为20。

   ·
针对Vector/Cube融合计算的算子，启动时，按照AIV和AIC组合启动，blockDim用于设置启动多少个组合执行，比如某款AI处理器上有40个Vector核和20个Cube核，一个组合是2个Vector核和1个Cube核，建议设置为20，此时会启动20个组合，即40个Vector核和20个Cube核。**注意：该场景下，设置的****blockDim****逻辑核的核数不能超过物理核（2****个Vector****核和1****个Cube****核组合为1****个物理核）的核数。**

   · AIC/AIV的核数分别通过GetCoreNumAic和GetCoreNumAiv接口获取。

## 04 L2Cache切分

假设AI处理器的L2Cache大小为192MB，L2Cache读写混合带宽约为7TB/s，而AI Core外部存储Global Memory的带宽约为1.6TB/s，两者之间存在较大差距。搬入或搬出相同数据量的情况下，访问L2Cache读写数据比HBM更快。若数据无法命中L2Cache，即需要访问的数据不在L2Cache内，导致需要去HBM上读写，带宽利用效率较低，最终算子搬入或搬出数据变为算子整个运行过程的性能瓶颈。切分策略建议：当输入和输出数据的数据量超过L2Cache大小时，Tiling中使能L2Cache切分策略。下面举个例子来说明L2Cache切分帮助大家理解。

假设输入数据大小为InputTotalSize
= 384MB，L2Cache大小为192MB，总核数为20个核，数据未切分的情况下，整体一次完成计算。假设20个核一次可以处理共192MB的数据，则每个核至少两次读取输入数据。

未使能L2Cache切分如下所示。

```
constexpr int32_t TOTAL_LENGTH = 384 * 1024 * 1024 / sizeof(half); 
constexpr int32_t USE_CORE_NUM = 20; 
constexpr int32_t TILE_NUM = 2; 
constexpr int32_t BLOCK_LENGTH = TOTAL_LENGTH / USE_CORE_NUM; 
constexpr int32_t TILE_LENGTH = BLOCK_LENGTH / TILE_NUM; 
 
class KernelSample { 
public: 
    __aicore__ inline KernelSample() {} 
    __aicore__ inline void Init(GM_ADDR x) 
    { 
        xGm.SetGlobalBuffer((__gm__ half*)x + BLOCK_LENGTH * GetBlockIdx(), BLOCK_LENGTH); 
        pipe.InitBuffer(inQueueX, 1, BLOCK_LENGTH * sizeof(half)); 
        pipe.InitBuffer(inQueueY, 1, BLOCK_LENGTH * sizeof(half)); 
    } 
    __aicore__ inline void Process() 
    { 
        // 示例演示对输入数据加2的运算 
        constexpr int32_t loopCount = 2; 
        for (int32_t i = 0; i  xLocal = inQueueX.AllocTensor(); 
        // 对于每个核，除了首次读取外，读取第0块数据时，L2Cache内缓存的是第1块数据； 
        // 对于每个核，读取第1块数据时，L2Cache内缓存的是第0块数据； 
        // 每个核需要4次读取GM上的数据 
        DataCopy(xLocal, xGm[process * TILE_LENGTH], TILE_LENGTH ); 
        inQueueX.EnQue(xLocal); 
    } 
    __aicore__ inline void Compute() 
    { 
        LocalTensor yLocal = inQueueY.AllocTensor(); 
        LocalTensor xLocal = inQueueX.DeQue(); 
        Adds(yLocal, xLocal, 1, TILE_LENGTH);    
        inQueueY.EnQue(yLocal); 
        inQueueX.FreeTensor(xLocal); 
    } 
    __aicore__ inline void CopyOut(int32_t process) 
    { 
        LocalTensor yLocal = inQueueY.DeQue(); 
        DataCopy(yGm[process * TILE_LENGTH], yLocal, TILE_LENGTH); 
        inQueueY.FreeTensor(yLocal); 
    } 
} 
...
```

使能L2Cache切分后，输入数据均等切分成2份数据，则整体分两次进行计算，每次的计算量为192MB，第一次20个核先计算前192MB的数据，第二次20个核计算后192MB的数据。每次计算前读取的数据能够命中L2Cache，提升算子性能。

使能L2Cache切分如下所示。

```
constexpr int32_t TOTAL_LENGTH = 384 * 1024 * 1024 / sizeof(half); 
constexpr int32_t TILE_NUM = 2; 
constexpr int32_t USE_CORE_NUM = 20; 
constexpr int32_t TILE_LENGTH = TOTAL_LENGTH / TILE_NUM; 
constexpr int32_t BLOCK_LENGTH = TILE_LENGTH / USE_CORE_NUM; 
 
class KernelSample { 
public: 
    __aicore__ inline KernelSample() {} 
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, int32_t index) 
    { 
        xGm.SetGlobalBuffer((__gm__ half*)x + BLOCK_LENGTH * GetBlockIdx() + index * TILE_LENGTH, BLOCK_LENGTH); 
        yGm.SetGlobalBuffer((__gm__ half*)y + BLOCK_LENGTH  * GetBlockIdx() + index * TILE_LENGTH, BLOCK_LENGTH); 
        pipe.InitBuffer(inQueueX, 1, BLOCK_LENGTH * sizeof(half)); 
        pipe.InitBuffer(inQueueY, 1, BLOCK_LENGTH * sizeof(half)); 
    } 
    __aicore__ inline void Process() 
    { 
        // 示例演示对输入数据加2的运算 
        constexpr int32_t loopCount = 2; 
        for (int32_t i = 0; i  xLocal = inQueueX.AllocTensor(); 
        // 对于每个核，除了首次读取外，第二次读取可以命中L2Cache； 
        // 每个核2次读取GM上的数据，2次访问L2Cache读数据 
        DataCopy(xLocal, xGm, BLOCK_LENGTH ); 
        inQueueX.EnQue(xLocal); 
    } 
    __aicore__ inline void Compute() 
    { 
        LocalTensor yLocal = inQueueY.AllocTensor(); 
        LocalTensor xLocal = inQueueX.DeQue(); 
        Adds(yLocal, xLocal, 1, BLOCK_LENGTH);    
        inQueueY.EnQue(yLocal); 
        inQueueX.FreeTensor(xLocal); 
    } 
    __aicore__ inline void CopyOut() 
    { 
        LocalTensor yLocal = inQueueY.DeQue(); 
        DataCopy(yGm, yLocal, BLOCK_LENGTH); 
        inQueueY.FreeTensor(yLocal); 
    } 
} 
... 
 
extern "C" __global__ __aicore__ void simple_kernel(__gm__ uint8_t* srcGm, __gm__ uint8_t* dstGm) 
{ 
    AscendC::KernelAdd op; 
    // 输入数据均等切分成2份数据进行计算 
    for (int32_t i = 0; i AI处理器的物理核数是固定的，当L2Cache切分之后，可能发生部分核有计算拖尾的情况，即每次所有核计算量除以每个核处理的数据量不能被核数整除，导致最后需要部分尾核来计算尾块数据。而在尾核计算时，部分核始终处于空闲状态，从而使得算子的整体性能变差。

如下图所示，假设总的数据量为TotalSize，L2Cache切分之后分为两份TotalSize / 2，每个核每次的计算量为TotalSize / 2 / 25，即需要25个核进行处理，由于AI处理器的核数为20，因此每次计算时，1到5核的每个核需要多算一份数据，导致发生拖尾的情况。

计算拖尾示意图如下所示。

针对上述切分策略，调整拖尾核的位置后可以达到全局负载最优，如下图所示，完成所有计算时，1到10核多一次数据块的计算，可以实现全局负载最优。

核间负载均衡示意图如下所示。

## 06 更多学习资源

了解更多Ascend C算子性能优化手段和实践案例，请访问：[昇腾社区Ascend C信息专区](https://www.hiascend.com/ascend-c)。
