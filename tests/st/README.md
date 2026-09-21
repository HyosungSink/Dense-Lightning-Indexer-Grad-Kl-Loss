# ST 测试执行

用例来自 `cases/coverage.json`、`cases/correctness.json`、`cases/mock.json`。
生成器写出原始二进制及 `case.json`，记录输入顺序、Shape、dtype、比较容差与所属类别。
bfloat16 使用原始 uint16 位存储。

~~~bash
python3 -m tests.st.scripts.gen_data --suite mock --all \
  --output-dir /tmp/dense_lightning_cases
~~~

在 CANN 环境中，使用已构建的五输入 ACLNN 库执行一个紧凑用例：

~~~bash
python3 -m tests.st.scripts.run_device \
  --case-dir /tmp/dense_lightning_cases/mock_case_3 \
  --build-dir /tmp/cannjudge/denselightningindexergradklloss/build \
  --actual-dir /tmp/dense_lightning_outputs/mock_case_3 \
  --report /tmp/dense_lightning_outputs/mock_case_3/result.json \
  --repeat 3
~~~

运行前需加载 CANN 环境，并将该构建的 opapi、tiling 和算子包路径配置给运行时。
`--build-dir` 指向包含 `libcust_opapi.so` 的目录。
runner 对输出预填 NaN，检查输入/输出前后 64B guard、输入未被修改及每轮的四输出数值。
数值不符、越界或运行错误均返回非零退出码。

`iterations` 保存每轮四输出比较及对应的 `error_ratio`。顶层 `comparisons` 按输出
保留错误元素最多的一轮，顶层 `error_ratio` 据此汇总；它不代表某一次独立运行。
所有比例均从 `mismatch_count/element_count` 计算。没有完成数值比较时，
比例为 `null`，运行错误由 `status`、`stage` 和 `acl_status` 表达。

完整 ABI 的设备接口尚待确认，现有 runner 仅允许紧凑 fixture 或明确声明可桥接的案例；
非零 RoPE、显式长度及未声明桥接的正式案例不能丢弃参数后运行。规格型案例仅导出元数据。

也可独立比较其他 runner 产生的四个输出文件：

~~~bash
python3 -m tests.st.scripts.verify_result \
  --case-dir /tmp/dense_lightning_cases/mock_case_3 \
  --actual-dir /tmp/dense_lightning_outputs/mock_case_3/0
~~~

四输出必须存在、字节数匹配且全部有限，并逐元素满足
`abs(actual - golden) <= atol + rtol * abs(golden)`。
比较器只做测试数值断言，不解释提交状态、历史错误率或计时偏差。

比较器同时输出本地诊断：`total` 为所有输出的错误元素比例，`max_tensor` 为最大单输出
比例，`first_failing` 为声明顺序 `dQueryIndex → dKeyIndex → dWeights → loss` 中第一个
失败输出的比例，`first_failing_output` 给出其名称。所有输出均匹配时该比例为 0、名称为
`null`；没有可计数的比较时比例为 `null`。这些字段不改变逐元素容差或退出状态。

## 独立会话与 Mock 套件

`run_device` 接受 `--suite mock` 或一个 `--case-dir`。
`--sessions` 指定独立进程数；`--profile` 使用真实 msprof kernel 计时。
Profiler 模式默认每会话 5 次预热、5 条采样，普通模式默认不预热、执行 1 次。
会话内次数由 `--repeat` 指定，不与会话数相乘后放入同一进程。

~~~bash
python3 -m tests.st.scripts.run_device --suite mock --profile \
  --case mock_case_1 --case mock_case_3 \
  --build-dir /tmp/cannjudge/denselightningindexergradklloss/build \
  --actual-dir /tmp/dense_lightning_mock \
  --report /tmp/dense_lightning_mock/suite.json \
  --warmup 5 --repeat 5 --sessions 5 --timeout 120
~~~

每会话记录 runner JSON、数值输出、进程日志和原始 `OpBasicInfo*.csv`。
采样使用 `Task Duration(us)`，会话内取均值，会话间取中位数。
`session_timings` 保存会话均值的 mean/median/min/max 和
`spread=(max-min)/mean`；`session_statuses` 保存每会话状态。
不完整会话可以保留有效样本，但整个 Case 的耗时为 `null`，且测试不通过。
数值错误、guard 错误、进程非正常退出和超时均保留失败状态，不适用豁免。

`--timeout` 限制每个进程的墙钟时间，超时终止整个子进程组；墙钟时间不作为 kernel
耗时。`--device` 和 `--op-wait-seconds` 传入每个子进程。
默认使用 `/tmp/cannjudge/device-<device>.lock` 串行访问设备，
`--device-lock` 可指定共享锁。

使用同一 `--report` 路径重跑部分 Case 时，未选 Case 的结果保留；所选 Case 的失败
结果同样替换旧结果。构建产物、Mock 配置、测试 Python 文件或协议变化时拒绝合并。
每次执行使用新的结果目录，旧会话原始文件保持可读，写报告使用原子替换。

## 从原始记录重算

~~~bash
python3 -m tests.st.scripts.verify_result \
  --run-report /tmp/dense_lightning_mock/suite.json \
  --aggregation first_failing --output /tmp/dense_lightning_mock/recomputed.json
~~~

`--run-report` 支持单 Case 多会话报告和 Mock 套件报告。
重算检查构建与 fixture 身份、runner JSON 和 CSV 哈希；每会话必须有准确的采样数，
时长必须为有限正数，不能重复使用同一条记录。
会话边界由 `sessions` 保留，汇总耗时从 CSV 重新折叠；数值比例从每轮比较计数计算。
`--aggregation` 可选 `first_failing`、`total`、`max_tensor`，只改变显示的诊断口径。
