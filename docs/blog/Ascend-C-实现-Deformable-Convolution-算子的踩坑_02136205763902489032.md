# Ascend C 实现 Deformable Convolution 算子的踩坑全记录，从偏移采样到双线性插值的优化死磕历程；

- topicId: 02136205763902489032
- source: https://www.hiascend.com/developer/blog/details/02136205763902489032
- section: Ascend C
- createTime: 20260206123822

十月份在做 Deformable DETR 模型部署，这个模型的核心是 Deformable Convolution 和 Deformable Attention，能自适应调整采样位置提升检测精度。整个推理流程跑下来发现 Deformable Conv 层奇慢无比，单层就要 25ms，模型里有十几层，总耗时占了 40% 以上。PyTorch 的 DCNv2 实现只有 CUDA 版本，昇腾平台根本跑不了，只能 fallback 到 CPU，数据在 NPU 和 CPU 之间反复搬运，白白浪费带宽。查了一圈发现昇腾没有现成的 DCN 算子，MMDetection 的昇腾版本也没适配这个。没办法只能用 Ascend C 自己写，把可变形卷积从零实现一遍。听起来就是卷积加偏移采样，实际开发时才发现偏移量计算、插值采样、梯度反传每个环节都是硬骨头。从开始到优化完用了整整两个半月，把可变形卷积的数学原理、实现细节、性能优化全摸透了。

需求这块先说清楚，Deformable Convolution 是 DCNv1 提出的，DCNv2 做了改进加了 modulation。核心思想是卷积的采样位置不是固定的 grid，而是加上可学习的偏移量。标准卷积在位置 p0 采样时，采样点是 p0 + pn，pn 是固定的如 {(-1,-1), (-1,0), ..., (1,1)}。可变形卷积变成 p0 + pn + Δpn，Δpn 是网络预测的偏移量。偏移量是浮点数，采样位置不在整数格点上，要用双线性插值。DCNv2 还加了 modulation 权重 Δmn，最终卷积公式是 y(p0) = Σ w(pn) · Δmn · x(p0 + pn + Δpn)。这个算子的输入有三个：特征图 x、卷积权重 w、偏移量 offset（还有可选的 modulation mask）。输出是卷积后的特征图。关键难点是偏移后的采样位置是浮点坐标，要对特征图做插值采样，而且每个位置的偏移都不同，访存完全不规则。

开发环境用的 CANN 7.0.1，MindStudio 也是最新版。Ascend C 的文档里没有任何可变形卷积的参考，只能看论文和 CUDA 代码。DCNv2 的官方实现看了源码，CUDA 版本用了很多 trick，原子操作、共享内存、线程同步，移植到 Ascend C 要全部重写。PyTorch 的 C++ 接口看了，理解了前向和反向的数学公式，但实现细节还是要自己摸索。

第一版实现按照最朴素的思路写，五层循环遍历输出的每个位置。外层是 batch、output_h、output_w，内层是 channel 和卷积核的位置。对每个卷积核位置，读取对应的偏移量 offset_h 和 offset_w，计算采样位置 sample_h = h * stride + kh * dilation + offset_h，sample_w 同理。采样位置是浮点数，对特征图做双线性插值。插值完后乘以卷积权重累加。写了个 bilinear_interpolate 函数，输入是特征图、浮点坐标 (h, w)，输出是插值结果。逻辑是找到周围 4 个整数位置 (i, j), (i+1, j), (i, j+1), (i+1, j+1)，按距离加权平均。

第一版跑起来性能惨绝人寰，单层要 120ms，比 CPU 还慢好几倍。全是标量操作，一个采样点一个采样点算，完全没向量化。而且采样位置随机，访存全是 Cache Miss。Msprof 看了下 NPU 利用率只有 15%，大部分时间在等访存。

偏移量的读取优化了下，offset 张量的形状是 [batch, deformable_group * 2 * kernel_h * kernel_w, out_h, out_w]。每个输出位置对应一组偏移量，2 表示 h 和 w 两个方向。deformable_group 是分组数，通常是 4 或 8。读 offset 时要计算索引，batch_idx、group_idx、kernel_idx、out_h、out_w 多个维度，索引计算很复杂。预计算了一些偏移量，减少重复计算。还把 offset 预加载到 L1 Buffer，所有输出位置共享。

双线性插值的实现踩了很多坑，坐标越界要处理。采样位置加上偏移后可能超出特征图范围，要 clamp 或者 padding。选了 padding 策略，越界位置返回 0。但 padding 0 会影响卷积结果，特别是偏移量很大时。改成重复边界像素，但这样梯度计算会有问题。最后用了 padding 0，限制偏移量的范围不要太大。

插值的权重计算也优化了，权重是 (1-dh)(1-dw), dh*(1-dw), (1-dh)*dw, dh*dw 四个。dh = h - floor(h)，dw = w - floor(w)。这些权重每个采样点都不一样，要每次算。试了查表法，把 [0, 1] 区间离散化成 256 个值，预计算权重表。查表比计算快，但精度略有损失。测试了下精度影响在 0.5% 以内，能接受。

卷积权重的访问也优化了，weight 张量是 [out_channels, in_channels // deformable_group, kernel_h, kernel_w]。读取时要按照 group 来，每个 group 处理部分 channel。预加载到 L1 Buffer，减少 GM 访问。但 weight 很大，几百 KB，L1 放不下所有 channel。分批加载，一次加载 16 个 out_channel。

向量化改造研究了很久，可变形卷积的难点是采样位置不规则，很难向量化。试了在 channel 维向量化，一次处理 64 个 channel。但每个 channel 要做独立的插值采样，位置不同，没法批量处理。改成 SIMD within a register，每个向量元素独立处理，用 Mask 控制。但效果不好，向量化的收益很小。

并行策略调整了，按输出位置并行。每个 Block 处理一个 (batch, out_h, out_w) 位置，Block 内遍历 channel 和卷积核。多个 Block 并行处理多个输出位置，充分利用多核。测试了下性能提升到 60ms，但还是不够快。

Deformable Group 的处理也要注意，分组是为了减少计算量。输入 channel 分成 deformable_group 组，每组独立做可变形卷积。组之间不共享偏移量，更灵活。实现时要正确计算每个组的 channel 范围和 offset 索引。写了个辅助函数计算组内的索引映射。

Modulation 权重加上了，DCNv2 的改进是加了可学习的调制权重。modulation 张量形状是 [batch, deformable_group * kernel_h * kernel_w, out_h, out_w]。每个采样点有个权重，做 sigmoid 后在 [0, 1] 范围。插值结果乘以 modulation 再乘卷积权重。Sigmoid 用 Ascend C 的向量指令，但还是标量调用，一个个算。

采样点的预计算优化了性能，卷积核的位置 (kh, kw) 是固定的。对于 3x3 卷积，9 个位置的基础偏移可以提前算好。运行时只需要加上学习的 offset，不用每次都算。建了个查表，kernel_size * kernel_size 的表，存基础偏移。

特征图的分块加载试了下，整个特征图太大放不进 L1。按照输出的位置，只加载需要的区域。一个输出位置对应输入的一块区域，大小是 kernel_size + max_offset。max_offset 是偏移量的最大值，通常限制在 [-10, 10]。这样每个 Block 只加载一小块，L1 能放下。但不同 Block 的区域可能重叠，重复加载浪费带宽。

内存访问模式实在太随机了，插值采样时访问特征图是完全随机的。试了软件 prefetch，预取下一个采样点的数据。但下一个采样点在哪不知道，要先算 offset，prefetch 效果不好。改用数据复用，一个 Block 内多个采样点可能访问相同区域，缓存在 L0 Buffer。但缓存管理复杂，收益有限。

数值精度又是个问题，插值计算用 FP32 精度高，但输入是 FP16。Cast 转换有开销，而且来回转换累积误差。用了混合精度，关键计算用 FP32，其他用 FP16。偏移量的精度特别重要，offset 是 FP32，插值权重也用 FP32，最后结果转回 FP16。

边界情况处理了一堆，采样位置超出范围、偏移量是 NaN、modulation 是负数，各种异常都要能处理。加了输入检查和数值 clamp，保证中间结果在合理范围。测试了极端情况，offset 全是 100、特征图全是 0、卷积核全是 0，都要能正确输出。

梯度计算是最麻烦的部分，反向传播要算三个梯度：特征图的梯度、offset 的梯度、modulation 的梯度（还有 weight 的梯度）。特征图的梯度要把输出梯度反向插值回去，插值的权重和前向一样。Offset 的梯度要算采样位置对输出的影响，涉及插值权重对坐标的导数。公式推导了半天，d(interpolate(x, h, w)) / dh 要算出来。Modulation 的梯度相对简单，就是输出梯度乘以对应的卷积结果。

反向传播的实现写了两周，前向用了一个月，反向又是一个月。每个梯度都要仔细推导公式，然后写代码验证。用数值梯度检查，torch.autograd.gradcheck 验证梯度正确性。一开始误差很大，调了很多次才对上。

原子操作加上了，反向传播时多个输出位置的梯度要累加到同一个输入位置。多个 Block 并发写，要用原子操作保证正确性。Ascend C 的 AtomicAdd 指令，保证累加不会冲突。但原子操作慢，成了新的瓶颈。优化了下，每个 Block 先在本地累加，最后一次性原子加到全局。

性能测试终于得到一些提升，优化后的 Deformable Conv 从 120ms 降到 18ms，提升了 6 倍多。但对比 CPU 的 25ms，才快了 30%，还不够理想。继续优化了几轮，最后降到 12ms，比 CPU 快了一倍。

不同参数测试了几组，kernel_size 3x3 是 12ms，5x5 是 28ms，卷积核越大越慢。deformable_group 4 是 12ms，8 是 14ms，分组数影响不大。特征图大小的影响最明显，[1, 256, 50, 50] 是 12ms，[1, 256, 100, 100] 是 45ms，四倍面积四倍时间。

功耗和温度监控了下，Deformable Conv 的 NPU 利用率 35%，比标准卷积低很多。主要是访存不规则，计算单元等数据的时间长。功耗 8W，温度 55 度，散热没压力。

算子注册到 PyTorch 框架写了扩展，接口和 mmcv.ops.deform_conv2d 保持一致。输入是 input, offset, weight, stride, padding, dilation, groups, deformable_groups, bias, modulation。DCNv1 没有 modulation，传 None 就行。

单元测试写了很多，对比 CUDA 版本的 DCNv2 实现。前向结果误差在 1e-3 以内，主要是插值精度的差异。反向梯度误差稍大，1e-2 左右，但在可接受范围。还测试了不同配置，stride、padding、dilation 各种组合。

实际部署在 Deformable DETR 模型上，整个推理流程测试了端到端性能。推理时间从 350ms 降到 220ms，Deformable Conv 不再是最大瓶颈。检测精度没有变化，AP 和 PyTorch 一致。

CenterNet、DCN-Det 这些模型也测试了，都用到 Deformable Conv。性能都有 50% 以上的提升，效果不错。

多线程并发测试了 4 个线程，每个线程处理一张图。线程安全，Context 隔离。4 线程吞吐量是单线程的 3.2 倍，扩展性一般，主要是访存竞争。

错误处理加了输入检查，offset 的 shape 要匹配 kernel_size 和 deformable_group。stride、padding、dilation 要合法。modulation 如果提供，shape 也要对。内存分配失败要捕获，返回错误码。

版本兼容性测试了 CANN 不同版本，7.0 和 6.3 接口有些差异。用宏定义做适配。PyTorch 1.11、2.0 都能用。ONNX 导出比较麻烦，Deformable Conv 不是标准算子，要自定义。写了个转换脚本，把参数映射过去。

线上运行了两周，处理了几万张图片，稳定性还行。偶尔有几次输出异常，看日志是 offset 的值太大，采样位置超出很远。加了 offset 的 clamp，限制在 [-20, 20] 范围。

性能对比记录了数据，CPU Deformable Conv 25ms，自定义 NPU 12ms，提升 2.1 倍。端到端 Deformable DETR 从 350ms 降到 220ms，FPS 从 2.9 提升到 4.5。离实时还有距离，但已经能用了。

成本收益算了下，开发两个半月，代码 5500 行，包括前向、反向、测试。性能提升 2 倍，检测服务吞吐量提升 50%。虽然开发周期长，但 Deformable Conv 在检测领域用得越来越多，值得投入。

后续优化还想试试，采样位置的聚类，相近的采样点批量处理。还想研究 Deformable Attention 的实现，原理类似但计算量更大。这些列在计划里，有时间继续优化。

文档整理了详细的开发笔记，可变形卷积的数学原理、双线性插值的实现、梯度推导都记录了。踩过的坑也写成了 FAQ，浮点采样、不规则访存、原子操作这些细节。代码注释很详细，关键算法都解释了推导过程。
