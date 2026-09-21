# DenseLightningIndexerGradKlLoss 测试套件

本目录维护 Coverage、Correctness、Mock Cases 三类用例，以及运行它们所需的参考计算、
数据生成、设备执行和数值断言。验收工具、提交历史、性能校准和实验记录保存在仓库外。

## 用例分类

| 类别 | 文件 | 内容 |
|---|---|---|
| Coverage | `st/cases/coverage.json` | 50 项：18 个数值规格、3 个大 Shape、22 个拒绝、4 个平台、3 个 ABI 待定 |
| Correctness | `st/cases/correctness.json` | 35 项：11 个正式 ABI 数值案例、2 个确定性规格、22 个紧凑 ABI 回归 |
| Mock Cases | `st/cases/mock.json` | 7 个固定紧凑 ABI 负载，记录 Shape、dtype、seed 和计算约定 |

`common/case_matrix.py` 统一加载三类 JSON，保证 ID 唯一，支持按类别选择。
`kind=numeric` 使用正式完整 ABI；`kind=compact` 使用独立五输入参考。
大 Shape、拒绝、平台、确定性和 ABI 待定案例只导出规格，不把登记视为设备执行通过。
Mock 输入由固定 seed 生成，golden 由参考公式即时计算。

## 目录

~~~text
tests/
├── common/
│   ├── case_matrix.py         # 分类与用例查询
│   ├── coverage.py            # 题面覆盖自检
│   └── reference.py           # float32 完整 ABI 参考模型
├── ut/
│   ├── test_coverage.py
│   ├── test_reference.py
│   ├── test_compact_reference.py
│   ├── test_case_suites.py
│   └── test_st_tools.py
├── st/
│   ├── cases/{coverage,correctness,mock}.json
│   └── scripts/
│       ├── check_coverage.py
│       ├── gen_data.py
│       ├── compact_fixture.py
│       ├── device_runtime.py
│       ├── run_device.py        # 单例/Mock、多会话执行和局部重跑
│       └── verify_result.py     # 数值比较与原始记录统计
├── COVERAGE.md
└── run_tests.sh
~~~

## CPU 自检与数据生成

依赖 NumPy 和 pytest。在仓库根目录执行：

~~~bash
bash tests/run_tests.sh
python3 -m tests.st.scripts.gen_data --list --suite coverage
python3 -m tests.st.scripts.gen_data --list --suite correctness
python3 -m tests.st.scripts.gen_data --list --suite mock

python3 -m tests.st.scripts.gen_data --suite mock --all \
  --output-dir /tmp/dense_lightning_cases
python3 -m tests.st.scripts.gen_data --suite correctness \
  --case variable_lengths_and_padding_poison --include-debug \
  --output-dir /tmp/dense_lightning_cases
python3 -m tests.st.scripts.gen_data --suite all --all --include-spec-only \
  --output-dir /tmp/dense_lightning_cases
~~~

`--case` 可重复指定，但必须属于所选类别。默认单张量上限为 2000 万元素；
`--max-elements 0` 可关闭限制。二进制和测试输出默认放在仓库外。

设备执行及比较命令见 [ST 说明](st/README.md)。CPU 自检不代表 NPU 已通过。

## Mock 执行与采样

~~~bash
python3 -m tests.st.scripts.run_device --suite mock --profile \
  --build-dir /tmp/cannjudge/denselightningindexergradklloss/build \
  --actual-dir /tmp/dense_lightning_mock \
  --report /tmp/dense_lightning_mock/suite.json \
  --warmup 5 --repeat 5 --sessions 5
~~~

`--case mock_case_3` 可选择单例，参数可重复指定。每个会话使用独立进程和独立 msprof
目录；会话内按 kernel 原始采样取均值，会话间取中位数，同时记录各会话状态、均值、
最小值、最大值和相对波动。任一数值失败、运行错误或不完整采样都会使测试失败。

同一报告支持局部重跑：二进制、套件、测试代码和运行协议一致时，只替换所选 Case，
保留其余 Case 的结果及原始文件。输入或协议不同时使用独立报告路径。
报告中的 `passed` 表示已记录 Case 均通过，`complete` 表示七例均有结果。

~~~bash
python3 -m tests.st.scripts.verify_result \
  --run-report /tmp/dense_lightning_mock/suite.json \
  --aggregation first_failing --output /tmp/dense_lightning_mock/recomputed.json
~~~

统计重算读取原始 CSV 和逐轮比较记录，检查文件哈希、采样数量和数值有效性，
不执行 NPU，也不依赖报告中缓存的耗时或错误比例。

## ABI 与参考约定

正式题面和现有五输入 Host Schema 的差异继续保留在 Coverage 的 ABI 待定案例中。
完整 ABI 用例保留统计量、RoPE、长度及因果掩码；紧凑回归和 Mock 在各自描述中明确约定，
不据此扩大正式题面的输入范围。

完整参考采用 float32 中间计算；非零 RoPE 暂按加性点积解释，S1>S2 的空因果行按不活跃行
处理。紧凑案例可显式使用 `loss_epsilon` 复现题面示例的 KL 边界公式。约定及未闭合范围见
[COVERAGE.md](COVERAGE.md)。
