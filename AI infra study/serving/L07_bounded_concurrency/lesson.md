# Coding 与架构每日学习｜2026-11-05
# 第7课：有界并发与背压，让超时后的工作仍然占住槽位

**预计学习时间：45–60 分钟。** 今天回到 Agent / 模型后端这一条线。第4课建立可替换后端接口，第5课区分“停止等待”和“工作真正结束”；昨天第6课讲稳定行 Softmax。今天只解决一个新问题：**请求同时涌入时，如何限制真正尚未结束的工作，并限制等待队列？**

这是一份离线 TypeScript 教学实现。无需模型、API key、GPU、网络、npm 包或付费服务。它适合在已有兼容 Node 的电脑上阅读与运行；本轮只验证了 Linux x64 / Node.js v24.22.1，未验证 Mac。这里没有完成真实 Agent 服务，也没有重新实现 Triton 算子。

## 1. 先看一个会把服务压垮的写法

假设底层模型后端最多安全地同时执行 2 个请求。你给每个调用设置 100ms 超时，然后在返回给用户的 Promise 的 finally 中释放槽位：

```ts
// 反例：请勿复制为正确实现。
active++;
return answerWithDeadline(request).finally(() => { active--; });
```

问题不在 finally 这个语法，而在**它等待的是哪个生命周期**。

- A、B 开始工作，active=2。
- 100ms 到，用户收到 TIMEOUT；A、B 所用后端不响应取消，仍在运行。
- finally 把 active 降到 0，C、D 又开始。
- 仪表盘说“并发 2”，真实后端却有 A、B、C、D 四份工作。

不断超时、不断放行会让真实并发持续上涨。超时保护了调用者的等待体验，却没有保护后端资源。本课将两个终点拆开：

1. **对外结果终点**：成功、失败、超时、取消，只能出现一次。
2. **资源生命周期终点**：底层工作真的完成或失败，才能释放 active 槽位。

这与第5课直接相接，不代表第5课的单次超时函数已经实现了并发管理。

## 2. 今天的需求和边界

设 C 是允许的运行数，Q 是允许的等待数。

| 项目 | 本课约定 |
|---|---|
| 运行上限 | 0 ≤ active ≤ C；C 是 1..1024 的整数 |
| 等待上限 | 0 ≤ queued ≤ Q；Q 是 0..100000 的整数 |
| 满载行为 | C 个运行且 Q 个等待时，新请求立即返回 OVERLOADED |
| 开始顺序 | 未超时、未取消的排队请求按 FIFO 开始 |
| 超时预算 | 从接纳时开始，包括排队和运行；timeoutMs 为 1..60000 整数 |
| 取消排队任务 | 移出队列、清理 timer/listener，不调用后端 |
| 取消运行任务 | 先确定对外结果，再发 AbortSignal；底层结束前继续占槽 |
| 输入信任 | 调度器、work 回调、原生 AbortSignal 是可信的进程内依赖 |
| 输出 | 成功值或有限错误码，不直接暴露底层异常详情 |

Q=0 是“没有等待室”，不是“不能工作”。有空槽就启动，没有空槽就拒绝。以上数值上限是避免误配置的教学限制，不是推荐线上把队列开到十万。

本课把 `work(signal)` 当作可信的工作接口。第4课已有的 JSON 输入解析、请求快照和模型输出验证仍应放在服务边界；这里不把这些旧知识再扩写一遍。陌生 JSON、恶意 Proxy/getter、会抛错的取消监听器和违反契约的时钟都不在当前安全边界内。

## 3. 背压到底反馈了什么

本课采用**拒绝式接纳控制**：队列满时返回 OVERLOADED。上游因此知道此刻没有接纳能力，可以自行选择降低产生速度、稍后再试或把忙碌状态告诉用户。这是一个明确的背压信号，不会自动让所有上游减速。

三件容易混淆的事：

- **并发限制**：同一时刻多少任务尚未完成。
- **队列上限**：还允许多少任务等着开始。
- **速率限制**：一秒钟最多接纳多少请求。

C=4 并不能保证每秒不超过4次请求。若每个任务只需1ms，它可能在1秒内完成很多轮。Q=100 也不会提高后端处理能力，只是允许更多请求等待。

在足够简化且稳定的系统里，平均吞吐上限可粗略理解为 C / 平均服务时间。Little 定律 L=λW 描述稳定系统的平均在途数量、吞吐和停留时间关系。它不是一个不用测量就能选 C、Q 的公式。真实模型会受 GPU batching、token 长度、显存、限额、慢请求和负载变化影响。并发加大可能更快，也可能引起 OOM 或尾延迟上升。

## 4. 状态与不变量：写代码前先写这张图

```text
                    有空槽
接纳 ────────────────→ running ──底层完成──→ done
  │                      │
  └→ queued ──获得槽位───┘
       │                  │
       │超时/取消          │超时/取消
       ↓                  ↓
      done          对外已结束，但仍为 running
```

`publicSettled` 独立于 `phase`：

- `phase='running', publicSettled=false`：还在运行，调用者还在等待。
- `phase='running', publicSettled=true`：调用者已收到超时或取消，工作仍可能运行。
- `phase='done'`：不再占用等待或运行资源。

必须一直成立的四条不变量：

1. active 只在开始工作时加一，只在**底层 Promise 的完成处理器**中减一。
2. 每个 public Promise 最多确定一个结果。晚到成功/失败不得覆盖 TIMEOUT。
3. `queue` 只包含尚未开始的活跃等待者；排队取消要从数组中真实删除。
4. 满载拒绝不创建 timer、不调用后端、不进入队列。

顺便解释一个语法：`#active` 是 JavaScript 私有字段，调用者不能直接改它；`readonly` 是 TypeScript 静态约束，Node 擦除类型后不能单靠它提供运行时保护。`snapshot()` 返回的是新建并冻结的数值快照，不是内部可修改数组。

## 5. 最小成功路径参考

先理解没有调度复杂度的顺序版本。它对一个输入处理完，才开始下一个：

```ts
export async function sequentialMap<T, R>(
  items: readonly T[], work: (item: T) => Promise<R>,
): Promise<R[]> {
  const results: R[] = [];
  for (const item of items) results.push(await work(item));
  return results;
}
```

这个参考用于检查“给定工作是否得到预期结果”。它没有队列、拒绝、取消或超时语义，因此不能当作所有故障路径的 oracle。`Promise.all(items.map(work))` 则通常会立即创建所有工作，**Promise.all 负责收集结果，不负责把并发限制为某个数**。

本课最终 API 如下：

```ts
const pool = new BoundedExecutor<string>({ concurrency: 2, maxQueue: 4 });
const result = await pool.submit(
  signal => backend.generate(request, signal),
  { timeoutMs: 5000, signal: callerSignal },
);
```

这是接口示意，request、backend、callerSignal 由接入层提供。可直接运行的完整例子是 ZIP 中的 demo.ts。泛型 `<string>` 表示这个池的成功结果是字符串，不会自动运行时验证模型真的返回字符串；真实适配器仍要做第4课的输出校验。

## 6. 完整核心实现

下面内容与附件 `executor.ts` 保持一致。先通读控制流，再看第7节的逐段解释。

```ts
/** Bounded, in-process, FIFO executor. Node 24 native erasable TypeScript. */
export type Outcome<T> =
  | { ok: true; value: T }
  | { ok: false; code: 'INVALID_OPTIONS' | 'OVERLOADED' | 'TIMEOUT' | 'CANCELLED' | 'TASK_FAILED' };
export type Work<T> = (signal: AbortSignal) => T | PromiseLike<T>;
/** Trusted scheduler: never throws; callback runs later, not inside schedule(). */
export type Schedule = (callback: () => void, delayMs: number) => () => void;
export interface Clock { now: () => number; schedule: Schedule }
export const realClock: Clock = {
  now: () => performance.now(),
  schedule: (callback, delayMs) => {
    const timer = setTimeout(callback, delayMs);
    return () => clearTimeout(timer);
  },
};
interface Entry<T> {
  work: Work<T>;
  owned: AbortController;
  phase: 'queued' | 'running' | 'done';
  publicSettled: boolean;
  deadline: number;
  finish: (result: Outcome<T>) => boolean;
}
export class BoundedExecutor<T> {
  readonly #concurrency: number;
  readonly #maxQueue: number;
  readonly #clock: Clock;
  #queue: Entry<T>[] = [];
  #active = 0;
  #started = 0;
  #completed = 0;
  #overloaded = 0;
  #peakActive = 0;
  #peakQueued = 0;
  constructor(options: { concurrency: number; maxQueue: number }, clock: Clock = realClock) {
    const { concurrency, maxQueue } = options;
    if (!Number.isSafeInteger(concurrency) || concurrency < 1 || concurrency > 1024)
      throw new RangeError('concurrency must be an integer in [1, 1024]');
    if (!Number.isSafeInteger(maxQueue) || maxQueue < 0 || maxQueue > 100000)
      throw new RangeError('maxQueue must be an integer in [0, 100000]');
    this.#concurrency = concurrency;
    this.#maxQueue = maxQueue;
    this.#clock = clock;
  }
  snapshot() {
    return Object.freeze({ active: this.#active, queued: this.#queue.length,
      started: this.#started, completed: this.#completed, overloaded: this.#overloaded,
      peakActive: this.#peakActive, peakQueued: this.#peakQueued });
  }
  submit(work: Work<T>, options: { timeoutMs: number; signal?: AbortSignal }): Promise<Outcome<T>> {
    const { timeoutMs, signal: caller } = options;
    if (!Number.isInteger(timeoutMs) || timeoutMs < 1 || timeoutMs > 60000)
      return Promise.resolve({ ok: false, code: 'INVALID_OPTIONS' });
    if (caller?.aborted) return Promise.resolve({ ok: false, code: 'CANCELLED' });
    if (this.#active >= this.#concurrency && this.#queue.length >= this.#maxQueue) {
      this.#overloaded++;
      return Promise.resolve({ ok: false, code: 'OVERLOADED' });
    }
    return new Promise<Outcome<T>>((resolve) => {
      const entry: Entry<T> = { work, owned: new AbortController(), phase: 'queued',
        publicSettled: false, deadline: this.#clock.now() + timeoutMs, finish: () => false };
      let cancelTimer = () => {};
      const onAbort = () => stop('CANCELLED');
      entry.finish = (result) => {
        if (entry.publicSettled) return false;
        entry.publicSettled = true;
        cancelTimer();
        caller?.removeEventListener('abort', onAbort);
        resolve(result);
        return true;
      };
      const stop = (code: 'TIMEOUT' | 'CANCELLED') => {
        if (entry.publicSettled) return;
        if (entry.phase === 'queued') {
          const index = this.#queue.indexOf(entry);
          if (index !== -1) this.#queue.splice(index, 1);
          entry.phase = 'done'; // Never invoked: no underlying work to release.
          entry.finish({ ok: false, code });
        } else if (entry.phase === 'running') {
          // Public result first. Abort may cause the backend to reject immediately.
          entry.finish({ ok: false, code });
          entry.owned.abort();
          // DO NOT decrement active here. Only #complete owns that operation.
        }
      };
      caller?.addEventListener('abort', onAbort, { once: true });
      const checkDeadline = () => {
        const remaining = entry.deadline - this.#clock.now();
        if (remaining <= 0) stop('TIMEOUT');
        else cancelTimer = this.#clock.schedule(checkDeadline, Math.ceil(remaining));
      };
      cancelTimer = this.#clock.schedule(checkDeadline, timeoutMs);
      if (this.#active < this.#concurrency) this.#start(entry);
      else {
        this.#queue.push(entry);
        this.#peakQueued = Math.max(this.#peakQueued, this.#queue.length);
      }
    });
  }
  #start(entry: Entry<T>) {
    // A delayed timer must not let expired queued work reach the backend.
    if (this.#clock.now() >= entry.deadline) {
      entry.phase = 'done';
      entry.finish({ ok: false, code: 'TIMEOUT' });
      return;
    }
    entry.phase = 'running';
    this.#active++;
    this.#started++;
    this.#peakActive = Math.max(this.#peakActive, this.#active);
    let work: Promise<T>;
    try { work = Promise.resolve(entry.work(entry.owned.signal)); }
    catch (error) { work = Promise.reject(error); }
    // Both handlers remain attached after timeout; late rejection is observed.
    work.then(
      (value) => this.#complete(entry, { ok: true, value }),
      () => this.#complete(entry, { ok: false, code: 'TASK_FAILED' }),
    );
  }
  #complete(entry: Entry<T>, result: Outcome<T>) {
    entry.phase = 'done';
    this.#active--;
    this.#completed++;
    // Completion observed at/after the deadline is TIMEOUT, even if its timer lagged.
    entry.finish(this.#clock.now() >= entry.deadline
      ? { ok: false, code: 'TIMEOUT' } : result);
    // A previous timeout/cancellation result cannot be overwritten.
    while (this.#active < this.#concurrency && this.#queue.length > 0)
      this.#start(this.#queue.shift()!);
  }
}
```

## 7. 沿着一个请求走一遍

### 7.1 submit 先做便宜检查

先校验 timeoutMs，再检查预先取消，最后判断满载。该顺序也是接口的一部分：超时参数非法时返回 INVALID_OPTIONS；已取消的合法请求返回 CANCELLED，不占用队列。只有接纳后才分配 entry、自己的 AbortController、timer 和外部 signal 的监听器。

Node 的 setTimeout 会转换某些非法延时，而不会替你拒绝全部无意义输入，因此需要明确校验。我们拒绝 timeoutMs=0，也不允许小数、NaN、Infinity 或大于60000。

### 7.2 为什么要“自己的” AbortController

调用者的 signal 只表示外部要求取消。执行器还要因自身 deadline 触发取消，因此创建 owned controller，把它的 signal 交给 work。这样不用也不能替调用者修改它的 controller。

finish 做三件事：设置 publicSettled、清理 timer/listener、resolve 对外结果。stop 在运行态调用 finish **之后**才 abort。这样，即使合作式后端立即因 abort 拒绝，公共结果已经确定为 TIMEOUT 或 CANCELLED，不会被 TASK_FAILED 抢走。

### 7.3 queued 取消与 running 取消为何不同

queued 的 work 从未调用，直接从队列删除即可。running 的 work 已调用，我们无法根据“发出了信号”推断它已停止。于是 running stop 中故意没有 active--。

真正的释放点在 #complete。成功、异步失败、同步抛错最终都被转成 Promise 完成，再进入这个统一出口。即使用户早已收到 TIMEOUT，原始 Promise 上的成功和拒绝处理器也仍在，避免晚到的 rejection 无人观察。

`Promise.resolve(work(...))` 配合 try/catch，是为了同时覆盖“返回 Promise”和“调用时直接抛错”。代码在调用 work 之前先加 active，因此 work 内部重入 submit 也能看到正确占用。

### 7.4 deadline 为什么要用单调时钟再检查

第5课讲了“谁先进入完成处理器，谁先确定结果”的软超时。今天增加一个更明确的**观察时刻截止规则**：

- 接纳时记录 `deadline = now() + timeoutMs`。
- timer 到时检查是否真的到期，过早则安排剩余时间。
- 从队列取出、即将启动时再检查，到期就不启动。
- 底层完成被执行器观察到时，再检查；观察时刻 `now() >= deadline` 就返回 TIMEOUT。

为什么只设一个 timer 还不够？事件循环忙碌时，timer 回调可能没及时运行。若前一个工作此时完成，不再检查 deadline 就可能把已过期的排队请求放进后端。

`performance.now()` 是此处使用的单调计时来源，避免系统时钟校准造成墙钟跳变。注入的 Clock 必须提供有限、非倒退的毫秒时间，并且 schedule 不抛错、不同步调用回调。手动时钟用于控制测试顺序，不代表模拟了 Node 所有事件循环阶段。

本规则仍然**不是硬实时承诺**：同步代码若阻塞2秒，不能保证10ms时就把结果发给用户，也不能抢占 CPU。某个底层结果实际早已准备好，但完成处理器直到 deadline 后才运行，本课仍按 TIMEOUT 处理。这是保守且明确的选择，不是测量真实网络完成时间。

### 7.5 FIFO 保证到哪里

完成一个工作后，从队首 shift 一个幸存等待者。C>1 时先开始的 A 可以比后开始的 B 更晚完成，FIFO 不保证输出完成顺序。`Promise.all` 返回值仍按传入 Promise 的数组位置排列，与谁先完成不同。

这里没有租户优先级、队列年龄公平性、任务耗时预测或跨进程调度。一个很慢的任务会长期占一个槽。每个进程都建 C=4 的池，四个进程可能合计16个运行任务；这不是全局 C=4。

## 8. 最重要的接口前提：哪个 Promise 才代表工作结束

**不要把上一课“可提前返回 TIMEOUT 的外层 Promise”直接作为这里的 work 返回值。** 那样池看到它已完成仍会提前释放槽，重新制造第1节的错误。

正确接法是让池持有能反映底层完整生命周期的 Promise，把统一 timeout/cancel 的对外结果交给当前池负责。常见陷阱：

- HTTP 请求收到响应头后 Promise 就完成，但响应流仍占连接/内存。适配器应把所需流读取或关闭也包含在工作生命周期内。
- GPU kernel 提交后 CPU 函数就返回，不等于设备任务完成。需要恰当的设备完成信号/事件语义，不能只计 CPU 提交次数。
- 工具函数偷偷启动后台任务并立即返回，池并不知道那些后台任务还活着。
- 一个 work 内部开十个子请求，C=2 只限制两个 work，不能自动限制到两个真实子请求。

本课的 ControlledBackend 在真正 finish/fail 时才结束 Promise，因此可以检验槽位规则。但这个 mock 不会自动证明任意真实 SDK 的生命周期语义。连接真实模型之前，必须审查该 SDK/adapter 的实际完成定义。

若后端永远不结束，它会永远占住一个槽。这是保守的资源安全策略，也可能让服务失去可用性。真实系统需要隔离进程、后端连接回收、运维重启策略或明确的资源终止机制；不能仅凭“已经等够久”谎称资源释放。本课不实现这些生产能力。

## 9. 跑一次确定性演示

在已有 Node v24.22.1 的环境中解压并进入目录：

```bash
node --version
node demo.ts
node --test executor.test.ts
node audit_stress.ts
```

不需要 npm install。演示设置 C=1、Q=1：

1. A 开始，B 等待，C 立即 OVERLOADED。
2. 手动推进到10ms，A 对外返回 TIMEOUT，A 的 aborted=true。
3. 后端 actual active 仍为1，开始过的 ID 仍只有 A。
4. 手动让 A 真正结束，B 才开始。
5. B 成功后 active=0、queued=0、pending timers=0。

完整输出见 DEMO_OUTPUT.txt。重点不是记住输出文字，而是盯着**公共结果、取消信号、真实后端 active**三个量，它们不能相互替代。

## 10. 故障测试：先破坏假设，再看计数器

本轮22项具名测试全部通过，覆盖：

- C/Q 与 timeoutMs 的非法边界；预取消请求不接纳。
- C 运行、Q 等待、再来一个拒绝；Q=0 的准确语义。
- FIFO 开始顺序与不同完成顺序。
- 排队超时和取消立即释放 Q，不启动已移除任务。
- 运行超时或取消后，忽略 abort 的后端仍占槽。
- 晚到成功不覆盖 TIMEOUT，晚到 rejection 有处理器。
- 合作式取消、同步抛错、异步拒绝、正常成功都走正确收尾。
- timer 与 caller listener 清理；公共结果只确定一次。
- timer 延迟时，过期排队任务不启动；恰好在 deadline 观察完成判为超时。
- 回调重入 submit 不能突破并发限制。
- 1000次提交、C=3/Q=7：接纳10个、拒绝990个，backend 峰值为3、queue峰值为7。
- 顺序参考与池的成功结果一致；真实1ms timer仅作连通性 smoke test。

另外提供独立的200个 seed × 100操作随机状态测试。它混合提交、取消、成功、失败、时间推进与合作/不合作后端，显式枚举 C=1..4、Q=0..4 的全部20种组合、每种10次，检查容量边界、执行器与 mock 的 active 一致、结束后的 timer 清理。随机测试不是形式化证明，不覆盖恶意依赖或所有运行时调度。

特别注意：1000次提交的测试**不代表整个应用只分配了10个对象**。调用方仍创建并持有1000个 Promise/结果。池只限制它接纳的工作和队列；外部生产者若在无限循环里创建请求，仍然可以耗尽内存。反压必须在调用链上被真正处理。

## 11. 基准：测量本地等待重叠，不冒充模型提速

运行：

```bash
node benchmark.ts
```

场景是24个本地 timer mock，每个请求等待约3ms，比较 C=1 与 C=4。每组先做1次正确性/预热轮，再交错顺序收集7轮整批墙钟时间。每轮都检查返回值和容量不变量。

本轮 Linux x64 / Node v24.22.1 的结果：

- C=1：整批中位数约75.35ms，范围74.62–76.07ms。
- C=4：整批中位数约19.34ms，范围18.59–20.49ms。

完整原始样本在 BENCHMARK.json。数值会随机器和负载变化；基准重新运行不要求复现相同小数。

它演示了独立异步等待可以重叠。它没有测量模型 token 速度、网络耗时、GPU 利用率、生产吞吐提升或单请求 p95/p99。7轮**整批**样本不是24×7个独立请求延迟样本。单线程 JavaScript 的同步计算也不会因 C=4 自动变成四核并行。

真实上线前应分别测：接纳/拒绝计数、排队等待、底层服务时间、用户可见停留时间、超时后仍活跃任务数、队列长度、资源峰值。不要只报平均响应时间，同时掩盖大量被拒绝的请求。

## 12. 复杂度和下一步权衡

本实现为可读性使用数组：

- 常见接纳与满载判断是 O(1)。
- 从队首 shift、任意位置取消的 indexOf/splice 是 O(Q)。
- 一次完成可能检查 O(Q) 个已过期条目；数组反复 shift 的累计搬移成本最坏为 O(Q²)。
- 执行器保留至多 C 个运行 entry 和 Q 个等待 entry，管理开销为 O(C+Q)，不含 work 内存和调用方对象。

条数上限不等于字节上限。一个等待请求可能带几百MB上下文或图片，Q很小也可能占很多内存。真实系统通常还需要输入大小/token预算和总在途字节预算。

大队列的 O(Q) 取消可能成为事件循环延迟来源。双向链表加 entry 节点能降低删除开销，但增加指针维护与测试成本；环形队列不直接解决任意取消，需要 tombstone 或索引策略。今天先把语义做对，再根据指标更换数据结构。

对本课而言，更多线程、自动重试、持久化任务、LangGraph 节点和 Kubernetes 都不是第一步。首先确保入口和底层 Promise 的资源语义没有说谎。

## 13. 三个小改写练习

### 练习A：别把“忙”自动变成“无限等待”

给本地上游生产者添加处理 OVERLOADED 的策略：停止产生新工作，等待一个明确的恢复信号后再继续。先写出恢复信号与接纳的竞争条件，再设计测试。

约束：不允许在拒绝分支里无限递归重试，不允许每次失败立刻启动新 timer 形成重试风暴。先思考是否真正需要 `waitForCapacity()`，以及信号触发后会不会被别的生产者抢走槽位。本题是设计练习，不要求直接上线。

### 练习B：记录 queueWaitMs

注入同一个单调 Clock，记录接纳时刻和真正启动时刻，为成功启动的任务输出排队时间。取消后从未启动的任务应如何表达？不得把0当成所有“未启动”的默认值。

测试：立即启动、排队9ms后启动、排队超时从未启动、timer回调延迟。公共结果和资源完成事件仍须分开。

### 练习C：增加字节预算

submit 额外接收一个经过上游校验的 `estimatedBytes`。同时满足 C、Q 和总保留字节预算才能接纳。写清字节配额何时释放，特别是“运行超时但仍在执行”的任务。

反例提示：在 public Promise 的 finally 里退还字节配额，和提前释放并发槽是同一种错误。估计值也不是真实峰值内存，请明确估计误差如何处理。

这些是练习，未记录你的答案，也不假设已掌握。先用自己的话解释“为什么 A 超时之后 B 还不能开始”，再动手修改代码。

## 14. 本轮验证与范围

- Node v24.22.1，Linux x64；使用内置模块，无第三方依赖。
- 22项具名测试通过；独立200×100操作随机测试通过；demo与本地timer基准执行成功。
- 所有附带 TypeScript 源文件经过 Node 语法检查；没有运行 tsc 类型检查。
- 无真实模型/API、网络请求、GPU、训练、部署或支付；没有安装软件或修改任何用户仓库。
- 未验证真实 SDK 的生命周期、跨平台兼容、长时间生产压力、分布式配额或CPU抢占。
- ZIP 包含完整核心/参考/mock/测试/基准/配置/说明和验证记录，不包含密钥。

Node 可以直接执行这里的可擦除 TypeScript，并不代表它会检查泛型、readonly或函数类型，也不代表它会读取并执行所有 tsconfig 设置。运行成功和类型检查是两条独立证据。

## 15. 官方资料与阅读范围

以下版本固定为 Node v24.22.1，2026-11-05 已核读相关章节，与本轮执行版本一致；没有将 latest-v24 页面当作相同版本。

1. [Node v24.22.1：AbortController / AbortSignal](https://nodejs.org/download/release/v24.22.1/docs/api/globals.html#class-abortcontroller)：取消信号、aborted、监听器与清理。signal 是通知机制，不是任意代码的强制终止。
2. [Node v24.22.1：setTimeout](https://nodejs.org/download/release/v24.22.1/docs/api/timers.html#settimeoutcallback-delay-args)：延时参数转换、回调时间不作精确保证。基准使用同页的 timers/promises API。
3. [Node v24.22.1：TypeScript](https://nodejs.org/download/release/v24.22.1/docs/api/typescript.html)：类型擦除、不执行类型检查、显式 import type 与 .ts 扩展名等限制。类型擦除在24.12.0起标为稳定，不等于所有TS语法都支持。

4. [Node v24.22.1：performance.now](https://nodejs.org/download/release/v24.22.1/docs/api/perf_hooks.html#performancenow)：相对进程启动的高精度毫秒时间。
5. [Node v24.22.1：timers/promises.setTimeout](https://nodejs.org/download/release/v24.22.1/docs/api/timers.html#timerspromisessettimeoutdelay-value-options)：Promise timer 与 signal 选项，供基准中的合作式 mock 使用。

本课的有界队列是教学代码，不是 Node 官方生产调度器。上述文档支撑使用的 API 语义；容量规则、deadline边界和背压策略是本文明确选择并测试的应用契约。

**下一课计划：稳定 log-sum-exp 与分块/在线 Softmax。** 继续沿模型算子线理解“整行一次算完”如何变成分块组合，再衔接 Attention 数据流；架构线后续再讨论队列等待指标、重试预算与恢复。路线可按学习反馈调整。
