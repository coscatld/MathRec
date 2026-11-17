# 验证记录
日期：2026-11-02（Asia/Shanghai）。
环境：Linux x64，Node.js v24.22.1；未安装依赖。

## 已运行
- `node --test model.test.ts`：13 tests / 13 pass / 0 fail / 0 skip。
- `node demo.ts`：与README中的完整JSON一致。
- `node benchmark.ts`：1000次预热、9批×10000次顺序调用，正确性哨兵未触发。
- 一次观测：批平均耗时中位数0.3551427微秒/调用，批平均范围0.2999257–0.9341047微秒/调用。仅本机local mock+契约检查+Promise调度，非LLM/网络延迟、非请求级p50/p95、非性能提升证据。

## 未运行/未证明
- 无tsc静态类型检查，无跨操作系统测试。
- 无真实模型、网络、GPU、tokenizer、服务负载或第三方集成。
- 30个Promise.all请求只验证结果隔离，不证明线程并行、吞吐能力或线程安全。
- 未测试恶意Proxy/getter或无限不resolve的provider；这些不在当前契约范围。
- UTF-16截断行为是教学简化，生产需处理Unicode字符边界与模型token预算。

## 验收重点
无效输入不调用后端；错误requestId/超长/错误字段返回INVALID_OUTPUT；同步throw和异步reject归一化；调用者修改原对象不改变已捕获快照；更换后端无需改answer；未知字段不进入公开响应。
