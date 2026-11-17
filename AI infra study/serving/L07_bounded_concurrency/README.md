# 第7课：有界并发与背压

配套日期：2026-11-05。完整中文讲义：`lesson.md`。

## 运行

本轮实测 Linux x64 / Node.js v24.22.1。使用 Node 原生可擦除 TypeScript，**不安装任何依赖**。`package.json` 无 dependencies。没有 API key、外部网络调用、真实模型、GPU、付款或部署。

```bash
node --version
node --test executor.test.ts
node audit_stress.ts
node demo.ts
node benchmark.ts
```

也可使用 `npm test` / `npm run demo` / `npm run bench`，这些只是本地脚本，**无需 npm install**。没有执行 `tsc`，Node 的类型擦除不等于类型检查。Mac 需要已有兼容 Node；本轮没有跨平台验证。

## 文件

- `executor.ts`：有界 FIFO 队列、并发槽位、端到端单调时钟 deadline、取消和清理。
- `reference.ts`：成功路径的顺序映射参考；不实现超时、队列或生产调度。
- `fakes.ts`：手动时钟、可控制完成/失败/是否响应取消的后端。
- `executor.test.ts`：22 项离线测试。
- `audit_stress.ts`：独立随机状态序列检查，200 个 seed × 100 操作。
- `demo.ts` / `DEMO_OUTPUT.txt`：超时后仍占槽的可重复示例及本轮输出。
- `benchmark.ts` / `BENCHMARK.json`：本地 timer mock 的整批墙钟时间及本轮样本。
- `VALIDATION.md`：实际验证范围与未验证项。
- `package.json`：模块/运行命令/版本范围声明，不是兼容性证明。

## 接口

`new BoundedExecutor<T>({ concurrency: C, maxQueue: Q }, clock?)`

- C：1..1024 的整数；Q：0..100000 的整数。教学保护上限不是生产推荐值。
- `submit(work, { timeoutMs, signal? })` 返回 `Promise<Outcome<T>>`。
- timeoutMs：1..60000 整数；包括排队时间。0 返回 INVALID_OPTIONS。
- work 的 Promise 必须代表资源占用的完整生命周期。不能传入“外层早超时、内层仍工作”的 Promise。
- C 限制已启动且完成尚未被观察到的工作数，Q 限制等待数。满队列立即返回 OVERLOADED。
- 正在运行的工作即使对外 TIMEOUT/CANCELLED，也只在底层 Promise 完成后释放槽位。
- 队列中超时/取消立即移除，不调用 work。启动和完成时再次核对 deadline。
- 若后端永不结束，它会永远占一个槽。同步 CPU 工作会阻塞事件循环，不能被抢占。

本实现是单进程内异步工作教学器。没有 HTTP 服务、跨进程限流、自动重试、认证、字节预算、任务持久化、优雅关闭或严格公平调度。FIFO 仅指幸存等待任务的开始顺序，不保证完成顺序。
