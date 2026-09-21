# Ascend C算子性能优化实用技巧05——API使用优化

- techColumnCode: 20241107-1
- source: https://www.hiascend.com/developer/techArticles/20241107-1

## 1 简介

Ascend C是CANN针对算子开发场景推出的编程语言，原生支持C和C++标准规范，兼具开发效率和运行性能。使用Ascend C，开发者可以基于昇腾AI硬件，高效的实现自定义的创新算法。

目前已经有越来越多的开发者使用Ascend C，我们将通过几期“Ascend C算子性能优化”专题分享，围绕开发者最为关心的算子性能优化环节，介绍Ascend C算子常用的优化技巧，帮助开发者自主构建出更优性能的算子。专题内容将围绕流水优化、搬运优化、内存优化、API使用优化以及Tiling优化等优化技巧，从方案讲解、优化案例、性能对比等多角度展开介绍。前期内容回顾：

·  [《Ascend
C算子性能优化实用技巧01——流水优化》](https://www.hiascend.com/zh/developer/techArticles/20240819-1)

·  [《Ascend
C算子性能优化实用技巧02——内存优化》](https://www.hiascend.com/zh/developer/techArticles/20240823-1)

·  [《Ascend
C算子性能优化实用技巧03——搬运优化》](https://www.hiascend.com/zh/developer/techArticles/20240906-1)

·  [《Ascend
C算子性能优化实用技巧04——Tiling优化》](https://www.hiascend.com/zh/developer/techArticles/20240920-1)

下面进入第五期内容：Ascend
C API使用优化，介绍API的相关使用技巧，提升算子执行性能：

·  在算子kernel类外创建和初始化TPipe对象，触发类内Scalar编译优化

·  通过TQueBind接口，实现VECIN和VECOUT内存复用

·  通过SetMaskCount接口开启Counter模式，简化mask处理逻辑

·  Matmul使能AtomicAdd选项，减少Vector计算

·  合理使用归约指令，兼顾累加效率和执行性能

## 2 在算子kernel类外创建和初始化TPipe对象，触发类内Scalar编译优化

程序在创建类对象时，会分配内存空间，用于存储类中的相关成员变量或函数。当类中变量需要参与计算时，变量值从内存被加载到寄存器，计算完成后，变量从寄存器存储回内存。Scalar常量折叠和常量传播是编译器编译时的优化方式，优化前编译器会判断变量是否只初始化过一次或只赋值过一次，若满足此编译优化的前提条件，变量值将会尽量驻留在寄存器中，从而在后续使用变量时，将减少读取内存的操作，提升运行性能。

TPipe是用来管理全局内存和同步的框架，用户可以调用TPipe的接口，为TQue/TBuf进行内存分配。在编写Ascend C算子过程中，经常用一个类存放计算所需的相关变量，这里称类名为KernelExample。当TPipe对象在KernelExample类的实现中定义并初始化时，TPipe对象的内存空间在整个KernelExample对象的内存空间之中；需要注意的是，创建TPipe对象时，对象初始化会设置全局变量的TPipe指针，这导致KernelExample对象的内存有被外部污染的风险，此时编译器的编译优化将采取保守策略，不会对KernelExample对象中的Scalar变量进行常量折叠和常量传播。因此，在任何场景下，我们都建议将TPipe对象创建于KernelExample类外部，使得TPipe对象的内存空间独立于KernelExample类对象的内存空间，触发编译器对KernelExample类内Scalar的编译优化，减少算子Scalar指令耗时。

【反例】

代码中TPipe对象由KernelExample类内部创建并初始化，影响编译器scalar折叠优化，在npu侧导致scalar无谓增加。

```
template  class KernelExample { 
 public: 
     __aicore__ inline KernelExample() {} 
 
     __aicore__ inline void Init(...) 
     { 
         ... 
         pipe.InitBuffer(xxxBuf, BUFFER_NUM, xxxSize); 
         ... 
     } 
 
 private: 
     ... 
     TPipe pipe; 
     ... 
 }; 
 
 extern "C" __global__ __aicore__ void example_kernel(...) 
 { 
     ... 
     KernelExample op; 
     op.Init(...); 
     ... 
 }
```

【正例】

改为由kernel入口函数创建TPipe对象，在KernelExample类中保存TPipe指针使用。

```
template  class KernelExample { 
 public: 
     __aicore__ inline KernelExample() {} 
 
     __aicore__ inline void Init(..., TPipe* pipeIn) 
     { 
         ... 
         pipe = pipeIn; 
         pipe->InitBuffer(xxxBuf, BUFFER_NUM, xxxSize); 
         ... 
     } 
 
 private: 
     ... 
     TPipe* pipe; 
     ... 
 }; 
 
 extern "C" __global__ __aicore__ void example_kernel(...) 
 { 
     ... 
     TPipe pipe; 
     KernelExample op; 
     op.Init(..., &pipe); 
     ... 
 }
```

【性能对比】

aiv_scalar_time优化前后对比如下所示。

aiv_scalar_ratio优化前后对比如下所示。

通过性能数据对比可以看出，scalar
time优化明显，平均时间从281us减少到236us，下降17%；平均scalar_time时延占比从21%下降到17%。因此在scalar bound的场景下可以使用此优化措施。

## 3 通过TQueBind接口，实现VECIN和VECOUT内存复用

纯搬运类算子在执行并不涉及实际vector计算，若存在冗余的vector计算，会导致算子整体执行时间变长。这种场景可以使用Ascend C针对纯搬运类算子提供的TQueBind接口，该接口可以将VECIN和VECOUT之间绑定，省略将数据从VECIN拷贝到VECOUT的步骤，从而避免vector的无谓消耗。

【反例】

此代码片段中存在LocalTensor
-> LocalTensor的DataCopy指令，目的是为了保证搬入和搬出之间的流水同步。

```
template  class KernelExample { 
 public: 
     ... 
     __aicore__ inline void Process(...) 
     { 
         for (int i = 0; i (); 
             DataCopy(iLocal, inGm[i * 32], size); 
             QueI.EnQue(iLocal); 
             auto iLocal = QueI.DeQue(); 
             for (int j = 0; j (); 
                 DataCopy(oLocal, iLocal, size); // LocalTensor -> LocalTensor的DataCopy指令,以实现数据从VECIN到VECOUT的搬移 
                 QueO.EnQue(oLocal); 
 
                 auto oLocal = QueO.DeQue(); 
                 DataCopyPad(outGm[j], oLocal, ...); 
                 QueO.FreeTensor(oLocal); 
             } 
             QueI.FreeTensor(iLocal); 
         } 
     } 
 
 private: 
     ...  
     TQue QueI; 
     TQue QueO; 
     ... 
 }; 
 
 extern "C" __global__ __aicore__ void example_kernel(...) 
 { 
     ... 
     op.Process(...); 
 }
```

【正例】

将LocalTensor
-> LocalTensor的DataCopy指令替换为TQueBind接口，避免将VECIN拷贝到VECOUT的步骤，从而避免了冗余vector计算。

```
template  class KernelExample { 
 public: 
     ... 
     __aicore__ inline void Process(...) 
     { 
         for (int i = 0; i (); 
             DataCopy(bindLocal, inGm[i * 32], size); 
             queBind.EnQue(bindLocal); 
             auto bindLocal = queBind.DeQue(); 
             for (int j = 0; j  queBind; // 使用TQueBind替换原来QueI，QueO 
     ... 
 }; 
 
 extern "C" __global__ __aicore__ void example_kernel(...) 
 { 
     ... 
     op.Process(...); 
 }
```

【性能对比】

aiv_vec_time优化前后对比如下所示。

如上图所示，将反例中DataCopy指令替换为TQueBind之后有明显优化。由于省略了数据从VECIN拷贝到VECOUT的步骤，aiv_vec_time几乎缩减为0。

## 4 通过SetMaskCount接口开启Counter模式，简化mask处理逻辑

开发者可以通过**mask**参数进行掩码操作来控制实际参与计算的个数，目前支持Normal和Counter两种模式：

·  Normal模式：开发者需要自行指定每次迭代内参与计算的元素以及迭代次数，系统默认为Normal模式。

·  Counter模式：开发者直接传入计算数据量，实际迭代次数由Vector计算单元自动推断。

通过对比我们可以看到，Normal模式虽然具备单次迭代内的mask能力，但使用不如Counter模式方便，Counter不需要开发者感知迭代次数、处理非对齐尾块的操作。

具体来看，Normal模式下，当用户想要指定API计算的总元素个数时，首先需要自行判断是否存在不同的主尾块，主块需要将mask设置为全部元素参与计算，并且计算主块所需迭代次数，然后根据尾块中剩余元素个数重置mask，再进行尾块的运算，这中间涉及大量Scalar计算。Counter模式下，用户不需要计算迭代次数以及判断是否存在尾块，将mask模式设置为Counter模式后，只需要设置mask为{0, 总元素个数}，然后调用相应的API，处理逻辑更简便，同时减少代码量和Scalar计算量。

以下反例和正例中的代码均以[AddCustom](https://gitee.com/ascend/samples/tree/master/operator/AddCustomSample/FrameworkLaunch/AddCustom)算子为例，修改其中如下Add接口的调用代码，以说明Counter模式的优势。

```
AscendC::Add(zLocal, xLocal, yLocal, this->tileLength);
```

【反例】

输入数据类型为half的xLocal, yLocal，数据量均为15000。Normal模式下，每个迭代内参与计算的元素个数最多为256B/sizeof(half)=128个，所以15000次Add计算会被分为：主块计算15000/128=117次迭代，每次迭代128个元素参与计算；尾块计算1次迭代，该迭代15000-117*128=24个元素参与计算。从代码角度，需要计算主块的repeatTimes、尾块元素个数；主块计算时，设置mask值为128，尾块计算时，需要设置mask值为尾块元素个数24；这些过程均涉及Scalar计算。

```
uint32_t ELE_SIZE = 15000; 
AscendC::BinaryRepeatParams binaryParams; 
 
uint32_t numPerRepeat = 256 / sizeof(DTYPE_X);  // DTYPE_X为half数据类型 
uint32_t mainRepeatTimes = ELE_SIZE / numPerRepeat; 
uint32_t tailEleNum = ELE_SIZE % numPerRepeat; 
 
AscendC::SetMaskNorm(); 
AscendC::SetVectorMask(numPerRepeat); // 设置normal模式mask，使每个迭代计算128个数 
AscendC::Add(zLocal, xLocal, yLocal, AscendC::MASK_PLACEHOLDER, mainRepeatTimes, binaryParams);   // MASK_PLACEHOLDER值为0，此处为mask占位，实际mask值以SetVectorMask设置的为准 
if (tailEleNum > 0) { 
     AscendC::SetVectorMask(tailEleNum); // 设置normal模式mask，使每个迭代计算24个数 
     // 偏移tensor的起始地址，在xLocal和yLocal的14976个元素处，进行尾块计算 
     AscendC::Add(zLocal[mainRepeatTimes * numPerRepeat], xLocal[mainRepeatTimes * numPerRepeat],  
           yLocal[mainRepeatTimes * numPerRepeat], AscendC::MASK_PLACEHOLDER, 1, binaryParams);   
} 
AscendC::ResetMask();  // 还原mask值
```

【正例】

输入数据类型为half的xLocal, yLocal，数据量均为15000。Counter模式下，只需要设置mask为所有参与计算的元素个数15000，然后直接调用Add指令，即可完成所有计算，不需要繁琐的主尾块计算，代码较为简练。

当要处理多种15000个元素的矢量计算时，Counter模式的优势更明显，不需要反复修改主块和尾块不同的mask值。

```
uint32_t ELE_SIZE = 15000; 
AscendC::BinaryRepeatParams binaryParams; 
AscendC::SetMaskCount(); 
AscendC::SetVectorMask(ELE_SIZE);  // 设置counter模式mask，总共计算15000个数 
AscendC::Add(zLocal, xLocal, yLocal, AscendC::MASK_PLACEHOLDER, 1, binaryParams);                // MASK_PLACEHOLDER值为0，此处为mask占位，实际mask值以SetVectorMask设置的为准 
AscendC::ResetMask();  // 还原mask值
```

【性能数据】

normal模式 vs counter模式scalar
time如下所示。

normal模式 vs counter模式aiv
cycle如下所示。

normal模式 vs counter模式
total time如下所示。

从上述三幅性能对比图和示例代码可以看到，使用Counter模式能够大幅度简化代码，易于维护，同时能够降低Scalar和Vector计算耗时，获得性能提升。

## 5 Matmul使能AtomicAdd选项，减少Vector计算

对于Matmul得到的结果矩阵C(m, n)，若后续需要和GM上的矩阵D(m, n)进行Add操作，则可以在GetTensorC接口或者IterateAll接口的GM通路上，将enAtomic参数设为1，开启AtomicAdd累加操作，在搬出矩阵C到GM时，矩阵C的结果将直接累加到矩阵D的GM地址上，从而实现与矩阵D的Add操作。

【反例】

将Matmul的结果矩阵C和GM上的矩阵D分别搬到UB上，做完Add操作后，结果再搬出到GM。这样至少要多分配一块UB内存给矩阵D，假设在分离架构的处理器上执行，将多做三次搬运操作（矩阵C从GM搬到UB、矩阵D从GM搬到UB、Add结果从UB搬出到GM）。

```
 template  
 __aicore__ inline void MatMulKernel(...) 
 { 
    ... 
    Matmul mm; 
    TPipe pipe; 
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm); 
 
    mm.SetTensorA(gm_a); 
    mm.SetTensorB(gm_b); 
    mm.SetBias(gm_bias); 
 
    mm.IterateAll(local_c); 
 
    // while (mm.Iterate()) { 
        // mm.GetTensorC(local_c); 
    // } 
     
    DataCopy(local_d, gm_d, d_size); 
    event_t eventIdMTE2ToV = static_cast(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V)); 
    SetFlag(eventIdMTE2ToV); 
    WaitFlag(eventIdMTE2ToV); 
    Add(local_d, local_d, local_c, d_size); 
    DataCopy(gm_d, local_d, d_size); 
    ... 
 } 
 
 extern "C" __global__ __aicore__ void example_kernel(...) 
 { 
     ... 
     typedef MatmulType aType;  
     typedef MatmulType bType;  
     typedef MatmulType cType;  
     typedef MatmulType biasType; 
     MatMulKernel【正例】

计算Matmul结果时，调用IterateAll接口或者GetTensorC接口搬运到矩阵D的GM地址上，同时将接口中enAtomic参数设为1，搬出到GM时，Matmul结果矩阵C会累加到矩阵D上，从而得到两个矩阵Add后的结果。

```
 template  
 __aicore__ inline void MatMulKernel(...) 
 { 
    ... 
    Matmul mm; 
    TPipe pipe; 
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm); 
 
    mm.SetTensorA(gm_a); 
    mm.SetTensorB(gm_b); 
    mm.SetBias(gm_bias); 
 
    mm.IterateAll(gm_d, 1); // IterateAll接口中的enAtomic设为1 
     
    // while (mm. Iterate ()) { 
        // mm.GetTensorC(gm_d, 1);     // GetTensorC接口中的enAtomic设为1 
    // } 
    ... 
 } 
 
 extern "C" __global__ __aicore__ void example_kernel(...) 
 { 
     ... 
     typedef MatmulType aType;  
     typedef MatmulType bType;  
     typedef MatmulType cType;  
     typedef MatmulType biasType; 
     MatMulKernel【性能对比】

Matmul使能AtomicAdd选项前后性能对比如下所示。

以矩阵维度M=64，N=256，K=256，矩阵D为(64, 256)为例，Matmul使能AtomicAdd选项前后的性能对比如上图所示，平均cycle数从开启AtomicAdd选项前的154181变为开启后的135054，性能优化12.4%。因此在这种场景下，使能AtomicAdd选项能获取更优的性能。

## 6 合理使用归约指令，兼顾累加效率和执行性能

对于需要将一段连续buffer上的元素全部累加到一个元素上的场景。使用WholeReduceSum的累加效率较高，但是单条指令的执行速度不如BlockReduceSum，因此，为了兼顾累加的效率以及尽量使用执行较快的指令，需要根据不同的shape，通过不同的指令组合达到最佳的性能。

例如在float类型输入，shape大小为256时，使用两次WholeReduceSum或者三次BlockReduceSum都可以得到256个float的累加结果。考虑规约指令之间的组合，一次BlockReduceSum加一次WholeReduceSum也可以完成上述累加效果。由于单条指令，BlockReduceSum执行速度更快，所以性能更优于两次WholeReduceSum，并且由于只使用了两次规约指令，所以性能同样优于三次BlockReduceSum的方案。

【反例】

```
反例1： 
... 
static constexpr uint32_t REP_LEN = 256; 
TBuf calcBuf; 
pipe.InitBuffer(calcBuf, totalLength * sizeof(float)); 
AscendC::LocalTensor tempTensor1 = calcBuf.Get(); 
constexpr uint32_t repCount = REP_LEN / sizeof(float); 
const uint32_t repNum0 = (totalLength + repCount - 1) / repCount; 
AscendC::SetMaskCount(); 
AscendC::SetVectorMask(0, totalLength); 
AscendC::WholeReduceSum(tempTensor1, xLocal, AscendC::MASK_PLACEHOLDER, 1, 
    DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE); 
AscendC::PipeBarrier(); 
AscendC::SetVectorMask(0, repNum0); 
AscendC::WholeReduceSum(zLocal, tempTensor1, AscendC::MASK_PLACEHOLDER, 1, 
    DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE); 
AscendC::PipeBarrier(); 
AscendC::SetMaskNorm(); 
... 
反例2 
... 
constexpr uint32_t c0Count = BLK_LEN / sizeof(DTYPE_X); 
const uint32_t blockNum0 = (totalLength + c0Count - 1) / c0Count; 
const uint32_t blockNum1 = (blockNum0 + c0Count - 1) / c0Count; 
AscendC::SetMaskCount(); 
AscendC::SetVectorMask(0, totalLength); 
AscendC::BlockReduceSum(tempTensor1, xLocal, AscendC::MASK_PLACEHOLDER, 1, 
    DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE); 
AscendC::PipeBarrier(); 
AscendC::SetVectorMask(0, blockNum0); 
AscendC::BlockReduceSum(tempTensor1, tempTensor1, AscendC::MASK_PLACEHOLDER, 1, 
    DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE); 
AscendC::PipeBarrier(); 
AscendC::SetVectorMask(0, blockNum1); 
AscendC::BlockReduceSum(zLocal, tempTensor1, AscendC::MASK_PLACEHOLDER, 1, 
    DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE); 
AscendC::PipeBarrier(); 
AscendC::SetMaskNorm(); 
...
```

【正例】

当输入shape为256，输入类型为float时。采用一次BlockReduceSum加一次WholeReduceSum的组合方式实现将256个float元素求和。完整样例链接请参考[ReduceCustom](https://gitee.com/ascend/samples/tree/master/operator/ReduceSample/FrameworkLaunch/ReduceCustom)。

```
... 
static constexpr uint32_t BLK_LEN = 32; 
TBuf calcBuf; 
pipe.InitBuffer(calcBuf, totalLength * sizeof(float)); 
AscendC::LocalTensor tempTensor1 = calcBuf.Get(); 
constexpr uint32_t c0Count = BLK_LEN / sizeof(float); 
const uint32_t blockNum0 = (totalLength + c0Count - 1) / c0Count; 
AscendC::SetMaskCount(); 
AscendC::SetVectorMask(0, totalLength); 
AscendC::BlockReduceSum(tempTensor1, xLocal, AscendC::MASK_PLACEHOLDER, 1, 
    DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE); 
AscendC::PipeBarrier(); 
AscendC::SetVectorMask(0, blockNum0); 
AscendC::WholeReduceSum(zLocal, tempTensor1, AscendC::MASK_PLACEHOLDER, 1, 
    DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REP_STRIDE); 
AscendC::PipeBarrier(); 
AscendC::SetMaskNorm(); 
...
```

【性能数据】

输入shape为256，数据类型为float。上述示例的性能数据如下：

表6-1 两次WholeReduceSum（反例1）、三次BlockReduceSum（反例2）、一次WholeReduceSum加一次BlockReduceSum（正例），三种累加方式的性能数据（循环100次的时间总和）

   反例1

   
   反例2

   
   正例

   
  13us

  
  13.94us

  
  8.44us

  

## 7 更多学习资源

了解更多Ascend C算子性能优化手段和实践案例，请访问：[昇腾社区Ascend C信息专区](https://www.hiascend.com/ascend-c)。
