# 验证记录

日期：2026-11-08；环境：Node.js v24.22.1，Linux；无第三方依赖安装。

已执行：
1. `node --test metrics.test.js`：18 项通过，0 失败、0 跳过；完整输出在 TEST_OUTPUT.txt。
2. 所有 6 个 `.js` 文件执行 `node --check`：通过。
3. `node demo.js`：生成 EXPECTED_OUTPUT.json；确定性假时钟事件，无网络。
4. `node benchmark.js`：一次不报告 warmup + 5 次各10000请求的本地 recorder 开销测试，结果在 BENCHMARK.json。测的是记录/汇总操作，不是实际服务吞吐量，也不是用户延迟。

覆盖：空样本、零时间戳、nearest-rank分位数、异常参数、成功路径与数学参考一致、排队超时、运行超时后仍计为running、排队/运行取消、后端错误、未完成样本、重复事件、错误状态顺序、未知ID、时钟倒退/非有限时间、snapshot隔离、不可相加的分位数。

未验证：真实模型、真实网络或第三方集成、生产调度器插桩、GPU、跨机器时钟、TypeScript类型检查、长期负载/内存稳定性。未进行安装、部署、训练或付费调用。

父级复核：18项测试再次全过；demo与EXPECTED_OUTPUT.json逐字相同；6个JS语法复查通过。ZIP已在独立临时目录解压复测（见邮件交付凭据中的归档校验）；不包含依赖目录、密钥、个人数据、模型权重或环境文件。
