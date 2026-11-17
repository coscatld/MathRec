# 第10课：别把“调用超时”当作“模型执行结束”
## 用生命周期记录排队、执行与调用者等待

日期：2026-11-08（Asia/Shanghai）  
层次：Agent 服务的后端可观测性，不是模型网络层、GPU kernel 或分布式监控平台。  
建议主线学习 40–60 分钟（估计），不假设你已经掌握前课。

衔接：11月3日讨论超时与取消，11月5日实现有界并发与背压；11月7日完成 Attention 的数据流桥接。今天回到同一个“可替换模型后端的 Agent 服务”，只解决一个问题：**系统变慢时，怎样用可核验的事件区分等队列、做工作、等结果？** 本期用离线事件回放，把测量口径先固定；不重新实现调度器，也不连接真实模型。

## 1. 需求与限制：先定义时间边界，再画看板

假设并发上限为1。请求A先进入模型，B排队。B等待5毫秒后调用方放弃，A直到20毫秒才真正结束。若看到B超时就减“正在运行数”，下一请求可能提前启动，突破并发限制。若只统计成功请求的耗时，超时变多时看板反而可能显得更快。

所以记录两个相互关联、但不等同的生命周期：

- **工作生命周期**：入队 → 启动 → 后端真正结束；或入队后未启动就退出。
- **调用者生命周期**：提交 → 得到成功/失败，或超时/取消而停止等待。

调用者结束不自动宣告后端结束。运行中的取消是请求合作，不是已完成证明。今天的 recorder 是测量模型，不负责发送取消信号，也不负责安排下一任务。

前置知识：对象/Map、函数、异常、数组排序；理解 Promise 的 settle 即最终 fulfilled 或 rejected。无需 GPU；Mac M4 可运行本课 Node 代码。不要为本课安装监控服务器。

## 2. 接口契约与公式

以同一进程单调时钟记录四个时刻，单位统一毫秒：

|符号|语义|何时可能不存在|
|---|---|---|
|$a$|进入本课测量边界|本课每条记录都有|
|$s$|工作真正开始|被拒绝、排队时超时/取消|
|$b$|后端工作真正结束|仍在运行或从未启动|
|$c$|调用者停止等待|调用者尚在等待|

对实际启动的工作，排队时间 $Q=s-a$；已结束工作执行时间 $E=b-s$。调用者已结束时，等待时间 $W=c-a$。没有事件就没有该完成时长，不要用0填补“尚未完成”。0是合法的测量值，例如立即启动。

仅对同一条已完成记录，后端总驻留时间满足：

$$R=b-a=Q+E$$

只有当调用者正好在后端结束时返回，才有 $W=R$。超时情况下，$W$ 可能小于 $R$；后端结束之后序列化/回传也可能让 $W$ 更大。这里的“调用者”是服务内部调用包装器，不含用户网络往返或浏览器渲染，不能直接叫用户端到端延迟。

排队中退出的请求另记“排队退出等待”，不能冒充已启动工作的Q样本；运行中但未结束的请求记为运行数，并可另报当前年龄，不能塞进完成执行时间分布。本课accept仅在执行器接纳后调用，不包含过载拒绝样本；接入真实执行器时，立即过载拒绝应在接纳边界外单独计数，避免把拒绝的快速返回混入成功延迟后误判优化。

**观察例子（原创教学数据）**：$a=0,s=10,c=30,b=90$。调用者等30ms，排队10ms，模型执行80ms；不是“执行20ms”，也不是“总耗时30ms”。在时刻40查看时，执行时长尚未完成，运行年龄为30ms。时刻90才生成80ms执行样本。

## 3. 数据流与控制流：事件是事实，指标是派生值

```text
有界执行器（前课）
  接受/拒绝 → 入队 → 启动 → 后端真正 settle
                   ↘ 调用者超时/取消/收到结果
                         ↓
                本课生命周期 recorder
                         ↓
         样本数 + 状态计数 + 三类耗时分布
```

在真实接线中，入队事件来自执行器接纳点；启动事件必须放在调用后端之前；后端结束来自底层Promise的最终分支，而不是Promise.race的超时分支；调用者结束来自对外承诺的最终分支。先验证事件有序、只发生一次，再更新状态。测试版遇到非法转移直接抛错，暴露埋点错误；生产中应隔离监控异常、记录埋点错误计数，不能让监控故障击穿业务。

我们不把用户ID、请求正文、prompt或工具参数作为指标标签。请求标识仅用于本地生命周期关联，聚合标签只用有限枚举；生产需要单独的低采样trace关联，不能把每个request ID放入时序标签。

## 4. 最简单的参考：纸面算清一条请求

```js
// 原创教学例子。对已知完整时间线手算，先不引入框架。
const a = 0, s = 10, c = 30, b = 90;
const queueMs = s - a;       // 10
const executionMs = b - s;   // 80
const callerWaitMs = c - a;  // 30
const residenceMs = b - a;  // 90
console.assert(residenceMs === queueMs + executionMs);
console.assert(callerWaitMs !== residenceMs);
```

只有这段不够：它不能检查重复事件、非法顺序和缺失结束，也不能区分统计窗口中的未完成工作。下面的主实现把这些约束变成代码。请先运行demo，再阅读状态转移，最后改测试。

## 5. 可运行主实现与逐段阅读

代码采用 JavaScript ES modules，省去本课无关的编译步骤；不是 TypeScript 类型检查课程。使用已有 Node 24.x，实测24.22.1，无外部依赖。完整 `metrics.js` 如下：

```js
import { performance } from 'node:perf_hooks';

const CALLER = new Set(['ok', 'error', 'timeout', 'cancelled']);
const BACKEND = new Set(['ok', 'error', 'cancelled']);

/** Exact nearest-rank quantile, not an interpolated or streaming estimate. */
export function quantile(values, p) {
  if (!(p > 0 && p <= 1)) throw new RangeError('p must be in (0, 1]');
  if (values.some(x => !Number.isFinite(x) || x < 0)) throw new RangeError('invalid duration');
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.ceil(p * sorted.length) - 1];
}
export function distribution(values) {
  return { count: values.length, p50: quantile(values, .5),
    p95: quantile(values, .95), p99: quantile(values, .99) };
}

/** One process, one monotonic clock, finite teaching batch; times in ms. */
export class Recorder {
  #clock; #last = -Infinity; #rows = new Map();
  constructor(clock = () => performance.now()) { this.#clock = clock; }
  #now() {
    const t = this.#clock();
    if (!Number.isFinite(t) || t < 0 || t < this.#last) throw new Error('non-monotonic clock');
    this.#last = t;
    return t;
  }
  #get(id) {
    const r = this.#rows.get(id);
    if (!r) throw new Error('unknown request');
    return r;
  }
  accept(id) {
    if (typeof id !== 'string' || !id || this.#rows.has(id)) throw new Error('invalid or duplicate id');
    this.#rows.set(id, { id, acceptedAt: this.#now(), startedAt: null,
      backendAt: null, backendOutcome: null, callerAt: null, callerOutcome: null });
  }
  start(id) {
    const r = this.#get(id);
    if (r.startedAt !== null || r.callerAt !== null) throw new Error('request is not queued');
    r.startedAt = this.#now();
  }
  finishBackend(id, outcome) {
    const r = this.#get(id);
    if (!BACKEND.has(outcome) || r.startedAt === null || r.backendAt !== null)
      throw new Error('invalid backend transition');
    r.backendAt = this.#now();
    r.backendOutcome = outcome;
  }
  finishCaller(id, outcome) {
    const r = this.#get(id);
    if (!CALLER.has(outcome) || r.callerAt !== null) throw new Error('invalid caller transition');
    if ((outcome === 'ok' || outcome === 'error') &&
        (r.backendAt === null || outcome !== r.backendOutcome))
      throw new Error('caller result requires matching completed backend');
    r.callerAt = this.#now();
    r.callerOutcome = outcome;
  }
  snapshot() { return [...this.#rows.values()].map(r => ({ ...r })); }
  report() {
    const r = this.snapshot();
    // Do not put absent measurements into the distribution as zero.
    const started = r.filter(x => x.startedAt !== null);
    const abandoned = r.filter(x => x.startedAt === null && x.callerAt !== null);
    const backend = r.filter(x => x.backendAt !== null);
    const caller = r.filter(x => x.callerAt !== null);
    return {
      accepted: r.length,
      gauges: {
        queued: r.filter(x => x.startedAt === null && x.callerAt === null).length,
        running: r.filter(x => x.startedAt !== null && x.backendAt === null).length,
        callerPending: r.filter(x => x.callerAt === null).length,
      },
      callerOutcomes: Object.fromEntries([...CALLER].map(k => [k, caller.filter(x => x.callerOutcome === k).length])),
      backendOutcomes: Object.fromEntries([...BACKEND].map(k => [k, backend.filter(x => x.backendOutcome === k).length])),
      milliseconds: {
        startedQueueWait: distribution(started.map(x => x.startedAt - x.acceptedAt)),
        abandonedQueueWait: distribution(abandoned.map(x => x.callerAt - x.acceptedAt)),
        execution: distribution(backend.map(x => x.backendAt - x.startedAt)),
        callerWait: distribution(caller.map(x => x.callerAt - x.acceptedAt)),
      },
    };
  }
}
```

阅读顺序：

1. `quantile`先检查范围，再复制排序。不能原地改变调用者数组；nearest-rank采用向上取整，并把1起始排名转换为0起始索引。`distribution`每个分位数分别排序，牺牲常数开销换可读性，生产可排序一次复用。
2. `#now`检查有限非负值及单调性。允许相同时刻的多次事件；调用不合法时不写记录。`#rows`关联请求，`#last`约束这个进程统一时钟，不能把多个进程事件混在一起直接回放。
3. `accept/start`建立工作侧状态。只有未启动且调用者仍在等待的排队请求能启动。`startedAt !== null`保留0ms这个合法时间。
4. `finishBackend`要求已开始、未结束。`finishCaller`是独立终态；普通成功/错误需要已有匹配后端结果，超时/取消可以先出现。这样A的后端后来成功，不会回写A的调用者超时。
5. `snapshot`复制对象，外部分析不能修改内部状态。`report`先确定样本集合，再求差值与分位数。四个分布的count不一定相等，这不是bug，而是不同生命周期的事实。

本课没有同步锁：所有方法是单进程JS同步调用，不在方法中await。这个约束不意味着分布式并发安全；跨线程/跨进程事件汇聚需要另外定义顺序与重复事件处理。本课重复事件抛错，不是幂等远程事件消费者。

## 6. 最小实验、测试与可观察验收

解压ZIP后在目录执行：

```sh
node --test metrics.test.js
node demo.js
node benchmark.js
```

无须 `npm install`。`package.json`提供等价命令。`node demo.js`完整预期见 `EXPECTED_OUTPUT.json`。下面是实际配套的确定性输入，不使用真实sleep，避免系统负载改变测试结论：

```js
import { Recorder } from './metrics.js';
export function replay() {
  let now = 0;
  const recorder = new Recorder(() => now);
  const at = (t, action) => { now = t; action(); };
  // A runs slowly; B leaves while queued; C starts at zero and succeeds.
  at(0, () => { recorder.accept('A'); recorder.accept('B'); recorder.accept('C'); recorder.start('C'); });
  at(10, () => recorder.start('A'));
  at(20, () => recorder.finishCaller('B', 'timeout'));
  at(30, () => { recorder.finishBackend('C', 'ok'); recorder.finishCaller('C', 'ok'); });
  at(40, () => recorder.finishCaller('A', 'timeout'));
  const afterTimeout = recorder.report(); // A still occupies a running slot.
  at(100, () => recorder.finishBackend('A', 'ok'));
  return { afterTimeout, final: recorder.report(), traces: recorder.snapshot() };
}
```

这个例子可同时运行C和A，不是第一节并发上限1的假设案例。它只给出事件事实，不模拟槽位调度。40ms快照应有running=1、callerPending=0；execution只有C的30ms一条。100ms后，execution样本为30和90ms，callerWait样本为20、30、40ms，p95分别为90和40。B的20ms进入abandonedQueueWait，不进入startedQueueWait或execution。

测试不只是确认输出“看起来合理”。关键断言：

```js
import assert from 'node:assert/strict';
import { replay } from './scenario.js';
const { afterTimeout, final } = replay();
assert.equal(afterTimeout.gauges.running, 1);
assert.equal(afterTimeout.milliseconds.execution.count, 1);
assert.equal(final.milliseconds.execution.p95, 90);
assert.equal(final.milliseconds.callerWait.p95, 40);
```

18项完整测试还覆盖：空集合null、0时间戳、错误样本不能漏计、排队与运行取消、重复结束、未知ID、非法状态、时钟回退、snapshot隔离、延迟回传与数学参考、分位数不能相加。`reference.js`提供成功路径数学参考，故意比主实现简单；并非以同一状态机自证正确。

## 7. 从数字到结论：先问分母是谁

本课采用 nearest-rank 分位数：把N个样本升序排列，取第 $\lceil pN\rceil$ 个（从1开始），其中 $0<p\le1$。空集返回null。10个样本的p99就是最大值，不代表已经精确估计了未来99%的请求。不同软件会用不同插值规则，比较时必须注明口径。

不能写 `p95(排队) + p95(执行) = p95(总时长)`。原创反例：两条请求的Q分别为0和100，E分别为100和0，总时长都是100；Q与E的p95之和却是200。应先逐条算Q+E，再求分位数。

也不能把机器A的p95与机器B的p95平均当成全局p95。生产聚合应保留可合并分布（如histogram），而非只保留几个分位数；这是[Prometheus官方直方图说明](https://prometheus.io/docs/practices/histograms/)特别区分的能力。本课使用原始数组和精确排序来保持可读性，不部署Prometheus，不实现它的分桶误差模型。

每张延迟表至少同时看：样本数、结果类别、仍在排队数、仍在运行数、调用者仍在等待数。只看完成请求会遗漏长时间未完成的工作；把超时都剔除会产生幸存者偏差。不要通过人为给未完成工作填“当前年龄”来修补完成分位数，当前年龄是另外一种量。

### 一个可迁移的观察流程

1. 固定口径：本次测的是内部包装器，不是网络端到端；毫秒而不是秒；完成样本而不是所有到达样本。
2. 做预测：若只增加并发请求、服务能力不变，预期Q与队列长度先上涨，而单项E不一定变化。
3. 控制变量：固定mock工作量和输入类别，每次只变并发、排队上限或deadline中的一个。
4. 分层观察：Q上涨优先检查拥塞；E上涨检查后端工作量/资源争用；W很短但超时率高不是变快；运行数持续高而调用者已结束，检查取消是否合作。
5. 留反证：Q高并不自动证明GPU慢，也可能是执行器限制、长任务占槽或事件循环阻塞。记录替代解释，再设计能区分它们的实验。

例：调低deadline后W的p95下降但timeout占比从2%变成20%，不能报告“延迟优化”。应报告“调用者更早放弃，超时增加；后端完成E是否下降尚需另测”。同样，请求混合比例改变会影响分位数，不能在不同工作量下简单横比。

## 8. 设计取舍、故障路径与生产边界

**更简单的方案**：只在函数入口和返回处记两次时钟，适合无排队、无超时、单层同步函数的局部测量。出现队列或caller/backend分离后升级为本课事件记录，而不是马上引入微服务。

**本课方案**：单进程、内存记录、人工回放。Map按请求关联事件；聚合按有限状态分类。记录保留在内存是为了逐条核查，不适合无限期常驻。真实系统要定期导出已完成记录并清理，运行未结束记录设置独立告警/保留策略；不能因caller超时就立即清掉仍运行记录。

**升级条件**：流量较大时用有界窗口/直方图而不是永久保存所有样本；跨进程调用时再加入trace上下文和各节点本地duration。不要用不同机器的单调时钟直接相减，不要用wall clock回拨制造负时长。Node的[performance.now文档](https://nodejs.org/download/release/v24.22.1/docs/api/perf_hooks.html#performancenow)规定其高精度毫秒值以当前进程起点为零；本课用注入时钟让测试完全可重放。

常见故障：

- 重复end导致同一请求计数两次：状态机应拒绝，而非覆盖第一次事实。
- 启动之前就记录backend end：埋点位置错误，不能靠max(0, delta)掩盖。
- 用 `if (start)` 判断有无开始：0会被误判；必须显式判断null/undefined。
- timeout/cancel后仍允许队列中的任务启动：测量器应该暴露不一致，但真正防止它发生需要前课执行器。
- “监控看不到，所以任务结束了”：事件丢失与真正结束不同，应保留缺失/进行中状态。
- 把所有调用失败称作模型错误：过载拒绝、超时、调用者取消与backend error是不同原因。

时间和空间复杂度也属于设计：Map事件查找平均O(1)，快照扫描O(N)，精确分位数排序O(N log N)，记录空间O(N)。频繁全量快照可能比一次事件记录更贵。本课本地基准只测这些JS操作，不是模型吞吐、GPU利用率或网络SLO的证据。重复基准时记录Node版本、样本量与机器负载；先预热，跑多轮，看中位数及波动，不从单次结果宣称加速。

## 9. 三个小改写练习

1. 给报告加入“调用者已经退出、后端仍在运行”的数量与最大运行年龄。为0ms启动和两个不同启动时刻写测试，别把该年龄混入完成execution分布。
2. 设计按backend outcome分组的execution报告。用有限枚举区分success/error；保留每组count。解释为什么caller timeout不能直接当backend error标签。
3. 为已完成记录加入显式清理接口：必须同时满足工作终止且调用者终止才能删除。验证调用者超时但后端仍运行时不能删除。思考清理后哪些累计指标要保留、哪些窗口指标会变化。

验收不是背术语：拿到一条时间线，能说清每个数字的起止事件；给你一个更好看的p95，能追问样本数、失败比例和未完成任务。

## 10. 来源、版本与验证范围

本课代码与教学事件均原创，官方文档用于核验基础API和统计口径。读取日期：2026-11-08（Asia/Shanghai）。

- [Node.js v24.22.1 performance hooks](https://nodejs.org/download/release/v24.22.1/docs/api/perf_hooks.html)：核验performance.now的单位及进程时间基点。固定版本链接，避免latest漂移。
- [Node.js v24.22.1 test runner](https://nodejs.org/download/release/v24.22.1/docs/api/test.html)：核验内置node:test及命令行测试方式。无需第三方测试库。
- [Prometheus Histograms and summaries](https://prometheus.io/docs/practices/histograms/)：动态文档无固定版本号，读取时核验分位数不能直接跨实例聚合以及histogram的用途。本课未调用Prometheus API。

实际验证明细见附件 `VALIDATION.md` 和本讲义下一节。只做本地标准库测试、示例及记录器微基准；没有真实模型、网络服务、GPU或生产负载验证，没有安装依赖、训练或部署。

下一步建议：把观测所得的“拥塞/尾延迟”转化为资源预算和限流策略，再回到模型组件学习。课程计划可调整；本期读过不等于已经掌握。

## 11. 本轮实际验证

Node v24.22.1；18/18测试通过、0跳过；6份JS语法检查通过；demo与完整预期JSON逐字一致。已执行本地记录器微基准。ZIP进行CRC校验并在全新目录解压复测。没有运行真实执行器集成、真实模型、GPU或生产性能测试。
