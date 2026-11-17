# 第10课代码：不要把调用方超时当成执行完成

## 环境与离线运行

- 实测 Node.js v24.22.1，ES modules JavaScript，使用内置 `node:perf_hooks`、`node:test`、`node:assert/strict`。
- 建议已有 Node 24.x；无第三方依赖，无安装步骤，无网络、模型、支付或 GPU 调用。无需 `.env`。
- 这是 JavaScript 实现，未声称执行 TypeScript 类型检查。

```sh
node --test metrics.test.js
node demo.js
node benchmark.js
```

也可用 `npm test`、`npm run demo`、`npm run bench`，但不需要 `npm install`。

`node demo.js` 应与 `EXPECTED_OUTPUT.json` 完全相同。测试应为 18 passed、0 failed。基准时间受机器影响，不应与本目录数值逐字比较。

## 文件

- `metrics.js`：主实现。独立的事件记录器，不负责调度、超时 timer 或真正取消后端。
- `reference.js`：成功路径的三个串行阶段数学参考。
- `scenario.js`：确定性模拟事件序列，所有时间都是手动设置的毫秒值。
- `demo.js` / `EXPECTED_OUTPUT.json`：样例入口与完整预期输出。
- `metrics.test.js`：18 项测试。
- `benchmark.js` / `BENCHMARK.json`：本机记录及汇总开销测量脚本/本轮结果。
- `package.json`：运行命令与 Node 版本范围，无外部依赖。
- `TEST_OUTPUT.txt` / `VALIDATION.md`：实测证据和验证边界。

## 接口契约与事件插入位置

`accept(id)` 在有界执行器**确认接纳**之后调用；队列满而拒绝的请求不属于本课 accepted 样本。`start(id)` 紧贴调用后端之前；`finishBackend(id, outcome)` 在后端 Promise 真正 settle 后调用；`finishCaller(id, outcome)` 在本服务决定调用方结果时调用。后者不是网络客户端收到响应的时刻，不含客户端网络耗时。

实际接入上一课执行器时，这些是插桩点，不是重写调度算法。不要在同一 `finally` 同时伪造 caller/backend 完成。timeout 或 cancelled 是**调用方结果标签**，不证明后端真的停止。后端取消需要独立的真实完成事件。本课只回放事件，未声称已经集成进旧执行器。

调用方成功/错误只允许在匹配的后端结果出现之后记录。调用方超时/取消可以先发生；排队时离开后禁止再 start；运行中离开后仍允许后端最终完成。同一时间可以有多次事件。非法事件会抛错；不要直接让教学中的诊断异常破坏生产请求，应在生产 adapter 中隔离、计数并报警。

## 关键预期值

示例请求 C：0ms 开始，30ms 成功；A：10ms 开始，40ms 调用方超时，但100ms 后端才成功；B：排队至20ms超时，从未执行。

40ms 截面：`running=1`，`callerPending=0`，execution 样本只有 C 一条。最终 execution 样本为 30、90ms，而 callerWait 样本为20、30、40ms。因此两者 p95 分别是90ms和40ms。A 的后端成功不能把已记录的 caller timeout 改成成功。

## 教学边界

这是有限批次、单进程、单一单调时钟的内存记录器，既不提供 HTTP 服务也不含实际模型后端。所有 trace 留存：空间 O(N)。report 采用数组筛选和排序，O(N log N)，不是生产流式监控。请求 ID 仅用于本地关联，绝不能作为 Prometheus label。示例不记录 prompt、用户信息或模型输出。生产宜使用低基数 outcome/route/model-family 标签、固定桶或合适的 histogram，明确窗口、采样、在途工作和数据丢失策略。

零样本分位数为 null，不是0。只有已完成执行进入 execution 分布；尚未结束任务不补零，因此必须同时看 running 等 gauge 和样本数，避免误读幸存者偏差。已启动排队时长与未启动就放弃的排队时长分开报告；不能默默将后者删除并宣称队列很快。
