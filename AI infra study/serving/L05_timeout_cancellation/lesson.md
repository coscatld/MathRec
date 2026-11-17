# 第5课：超时不等于取消，给模型调用加上可解释的等待边界

日期：2026-11-03，星期二（北京时间）。预计主线45–60分钟，可分两次；进阶练习另约15分钟。时间均为估计，不推断你已经掌握前课。

## 1. 今天解决一个问题

上一课用输入/输出契约隔离了模型后端，但如果后端永远不返回，业务入口也会永远等待。今天继续这个最小学习助手，增加两种退出等待的理由：预算用尽与用户主动取消。

三个概念必须分开：
- **停止等待**：调用方不再等这个结果，获得TIMEOUT或CANCELLED。
- **请求停止工作**：向后端传递AbortSignal。后端必须检查或监听它。
- **工作实际结束**：后端停止计算、释放连接等。这可能晚于前两步，也可能根本没发生。

本课不会调用真实模型。我们用可手动完成/拒绝的mock，让这三种状态可观察，而不是用随机sleep碰运气。无需CUDA；Mac M4可运行已有兼容Node环境。GPU学习线下一课计划回到Softmax，再与后端线交替。

## 2. 需求、契约和优先级

沿用第4课：requestId为1–64个ASCII字母数字/下划线/连字符，prompt非空白且≤4096个UTF-16代码单元，输出预算整数1–512。返回值仍须检查ID、文本长度与结束原因。ZIP自带contract.ts，不依赖昨日目录；原始输入先复制并浅冻结。

新增options由可信的应用代码构造：timeoutMs是1..60000整数，signal可省略，若提供必须是真实AbortSignal。这里不将options视作未经校验的HTTP JSON。外部超时参数需要先在路由层验证再传入。

| 情况 | 结果 | 后端是否启动 |
|---|---|---|
| 输入不合法 | INVALID_INPUT | 否 |
| 超时值不合法 | INVALID_OPTIONS | 否 |
| 调用前已取消 | CANCELLED | 否 |
| 正常结果合法 | ok:true | 是 |
| 后端抛错/拒绝 | BACKEND_FAILED | 是 |
| 结果违反契约 | INVALID_OUTPUT | 是 |
| timer回调先被处理 | TIMEOUT，并发出abort | 是 |
| caller取消先被处理 | CANCELLED，并发出abort | 是 |

检查顺序是输入→选项→预取消，所以同时无效时按此顺序返回。运行中的竞争采用**第一个被本包装器处理的完成事件获胜**，不是按推测的远端完成时间排序。一次请求只有一个公开结果。

本课timeout是软等待期限。计时从输入验证完成、设置timer时开始，不包含进入函数前的排队。Node timer回调可能推迟，不能承诺精确N毫秒内返回，也不是跨服务传递的绝对deadline。[Node计时器文档](https://nodejs.org/docs/latest-v24.x/api/timers.html#settimeoutcallback-delay-args)

## 3. 最小完整主实现

下面与ZIP的timeout.ts一致。它依赖的parseRequest/parseReply完整实现也在ZIP；其中没有真实网络调用。

```ts
import { parseRequest, parseReply, type Request, type Reply } from './contract.ts';
export interface CancellableBackend {
  generate(request: Request, signal: AbortSignal): Promise<unknown>;
}
export type Outcome =
  | { ok: true; reply: Reply }
  | { ok: false; code: 'INVALID_INPUT' | 'INVALID_OPTIONS' | 'BACKEND_FAILED'
      | 'INVALID_OUTPUT' | 'TIMEOUT' | 'CANCELLED' };
// Trusted scheduler: callback runs later, never inside schedule(); returns cleanup.
export type Schedule = (callback: () => void, delayMs: number) => () => void;
export const realSchedule: Schedule = (callback, delayMs) => {
  const timer = setTimeout(callback, delayMs);
  return () => clearTimeout(timer);
};
export function answerWithDeadline(
  input: unknown,
  backend: CancellableBackend,
  options: { timeoutMs: number; signal?: AbortSignal },
  schedule: Schedule = realSchedule,
): Promise<Outcome> {
  const request = parseRequest(input);
  if (request === null) return Promise.resolve({ ok: false, code: 'INVALID_INPUT' });
  const { timeoutMs, signal: caller } = options;
  if (!Number.isInteger(timeoutMs) || timeoutMs < 1 || timeoutMs > 60000)
    return Promise.resolve({ ok: false, code: 'INVALID_OPTIONS' });
  if (caller?.aborted) return Promise.resolve({ ok: false, code: 'CANCELLED' });
  const owned = new AbortController();
  return new Promise<Outcome>((resolve) => {
    let settled = false;
    let cancelTimer = () => {};
    const onCallerAbort = () => stop('CANCELLED');
    function finish(result: Outcome): boolean {
      if (settled) return false;
      settled = true;
      cancelTimer();
      caller?.removeEventListener('abort', onCallerAbort);
      resolve(result);
      return true;
    }
    function stop(code: 'TIMEOUT' | 'CANCELLED') {
      // Choose public outcome before cooperative abort can reject the backend.
      if (finish({ ok: false, code })) owned.abort();
    }
    caller?.addEventListener('abort', onCallerAbort, { once: true });
    cancelTimer = schedule(() => stop('TIMEOUT'), timeoutMs);
    // Both handlers remain attached even after the public promise has settled.
    try {
      Promise.resolve(backend.generate(request, owned.signal)).then(
        (raw) => {
          if (settled) return; // discard late value, no further public result
          const reply = parseReply(raw, request);
          finish(reply === null ? { ok: false, code: 'INVALID_OUTPUT' } : { ok: true, reply });
        },
        () => { finish({ ok: false, code: 'BACKEND_FAILED' }); },
      );
    } catch {
      finish({ ok: false, code: 'BACKEND_FAILED' });
    }
  });
}
```

### 3.1 为什么自己创建AbortController？

调用方只把signal交给我们，我们不应取消调用方所拥有的整个控制器。内部owned控制器只管理本次后端工作。外部取消或内部超时时，我们都向这个专属signal发出通知。不同请求各有控制器，不串线。

接口比昨日多了signal参数。已有不接受第二个参数的简单JS函数可能忽略它而继续工作，但那不等于已经支持取消。真实SDK是否传递signal、何时终止网络请求，都必须独立核验。

### 3.2 为什么finish先于abort？

假设timer先触发，我们要返回TIMEOUT；随后owned.abort可能让合作后端立即拒绝Promise。如果先发abort而不先记录结果，错误分类很容易被后端的拒绝覆盖。这里先标settled、清理资源、确定公开结果，再通知后端取消。后端迟来的拒绝看见settled后只被观察，不产生第二个公开结果。

settled是单个调用闭包中的布尔值。Node当前线程中的这些短回调依次执行，所以它足够表达这里的“一次结算”；这不是跨进程锁，也不是数据库事务。

### 3.3 资源清理不能只放在超时分支

正常成功、后端失败、坏响应、取消、超时都通过finish。它清除本次timer并移除caller监听器。once:true只能保证监听器在事件真的触发后自动移除；若请求正常成功、caller一直不abort，仍需主动remove。

后端自己的资源由后端清理。包装器不能替任意SDK关闭连接。本课合作mock监听abort并reject；正常完成时也移除自己的监听器。非合作mock故意忽略signal，用于展示边界。[AbortSignal官方说明](https://nodejs.org/docs/latest-v24.x/api/globals.html#event-abort)

### 3.4 为什么不只写Promise.race？

Promise.race可以选择先完成的Promise，但不会自动取消其余工作，也不替你清除timer。简单等待场景用race并配合finally清理可以更短；当还要区分caller取消与超时、拒绝晚到结果，并将signal传入后端时，需要显式管理这些职责。不要把race本身当成取消机制。

这里同时注册成功与失败处理器，即使公开Promise已超时，也仍观察底层Promise的失败。否则孤立地启动异步工作再遗忘它，晚到reject可能形成未处理拒绝。成功处理器发现已settled直接返回，避免把迟到内容写回已结束的请求。

### 3.5 同步throw与异步reject

调用generate时立即throw，由外层try/catch处理；返回Promise之后的reject，由then的第二个函数处理。不要以为调用处的同步try/catch会自动捕获所有未来Promise错误。

本例信任进程内provider代码、scheduler及真实AbortSignal，输入/输出按普通JSON形状。恶意getter、Proxy、故意抛错的abort监听器、同步调用callback或抛异常的scheduler不在教学契约内。parseReply的白名单不是执行不可信JS的沙箱。生产应隔离不可信代码，并审查清理路径异常；不要为了“看起来不会报错”捕获所有程序bug。

## 4. 用确定性实验看见区别

完整demo如下，和ZIP一致：

```ts
import { answerWithDeadline } from './timeout.ts';
import { manualTimer, controlledBackend } from './fakes.ts';
const timer = manualTimer(), work = controlledBackend(false);
const waiting = answerWithDeadline({ requestId: 'demo', prompt: 'hello', maxOutputChars: 20 }, work.backend, { timeoutMs: 100 }, timer.schedule);
timer.fire();
console.log(JSON.stringify({ result: await waiting, abortRequested: work.signal()?.aborted, workStopped: work.stopped() }));
work.complete({ requestId: 'demo', text: 'late', finishReason: 'stop' });
console.log(JSON.stringify({ workStoppedAfterCompletion: work.stopped() }));
```

输出：
```json
{"result":{"ok":false,"code":"TIMEOUT"},"abortRequested":true,"workStopped":false}
{"workStoppedAfterCompletion":true}
```

第一行是重点：结果已经超时，信号已取消，工作却还没有结束。第二行是我们主动完成测试mock之后的状态，不是包装器强制终止了它。

手动timer不等待真实100毫秒，fire代表“现在处理超时回调”。这样测试只验证逻辑，不受CPU负载和计时抖动干扰。fakes.ts的每个实例只用于一次调用；不要把这个测试夹具当共享生产后端。

## 5. 测试代码如何覆盖故障

下面是ZIP中的两个关键测试，完整13项在timeout.test.ts。

```ts
const timer = manualTimer();
const work = controlledBackend(false);
const result = answerWithDeadline(input, work.backend,
  { timeoutMs: 10 }, timer.schedule);
timer.fire();
assert.deepEqual(await result, { ok: false, code: 'TIMEOUT' });
assert.equal(work.stopped(), false);
work.complete(reply);
assert.deepEqual(await result, { ok: false, code: 'TIMEOUT' });
```

这个测试不仅看到了超时，还明确检查工作没停止，以及晚到成功不会改结果。第二类测试让work.fail在公开结果已返回后发生，并等待一个事件循环回合；Node测试运行器若发现未处理Promise拒绝会报失败。

竞态还需要一个反直觉用例：先调用mock的resolve，再立刻fire手动timer。resolve只是排入Promise微任务，成功回调尚未执行；同步fire会先将结果定为TIMEOUT。此用例证明的是**本包装器的处理顺序契约**，不是模拟真实Node timer与I/O全部调度规则。

验收清单：
- 无效输入/timeout与预取消不启动后端。
- 正常成功、同步throw、异步reject、坏输出都有稳定分类。
- caller取消清timer并移除监听器；内部超时不取消caller的signal。
- 合作后端因abort拒绝，公开结果仍保留最先确定的TIMEOUT。
- 非合作后端晚到成功被丢弃、晚到拒绝被观察。
- 成功已经返回后再caller.abort或timer.fire，不改变原结果。
- 真实timer额外冒烟一次，只验证最终TIMEOUT，不断言精确耗时。

## 6. 环境与运行

基线Linux x64、Node.js24.22.1，零第三方依赖。已有兼容Node即可：
```sh
node --test timeout.test.ts
node demo.ts
node benchmark.ts
```

package.json设type:module，导入路径带.ts。Node类型擦除只运行可擦除语法，不执行tsc检查；本轮没有安装TypeScript编译器，也没有声称静态类型检查成功。macOS/Windows未实测，不能把Linux通过等同跨平台保证。

实际结果：13/13测试通过，0跳过；demo与预期一致；本地基准完成。无GPU、网络、真实模型、训练或付费API，无安装、部署、仓库修改。完整VALIDATION.md随包附上。

## 7. 性能、复杂度与限流陷阱

本课在输入/输出扫描之外，每个请求额外持有一个控制器、一个timer、一个caller监听器和少量闭包，逻辑辅助空间O(1)。文本检查仍随输入输出长度增长，操作系统timer调度内部复杂度不在这个O(1)声明内。

基准1000次预热后测7批，每批5000个顺序立即返回的mock。一次本机观测的批均值中位数约1.611微秒。这个指标包含timer创建/清理等本地开销，不是请求级p50/p95，不是远端模型性能，不代表本课实现带来加速。不要将它与昨日不同路径的数值直接作速度结论。

更重要的容量陷阱：**不能在公开请求超时的一刻，就默认底层工作已释放一个并发槽位。** 若非合作工作继续存在，此时又接收新任务，实际运行数会越积越多。未来并发限制器需要跟踪工作真正结束的生命周期，或使用能证明终止的隔离机制。今天没有实现容量管理，也不声称解决资源泄漏的所有来源。底层Promise若一直未结束且仍被外部资源持有，其处理器可能继续保留请求相关闭包；清timer与caller监听器不等于所有内存都已释放。

如果backend在当前线程执行长同步循环，事件循环无法调度timer，caller取消也不能及时处理。AbortSignal不能抢占CPU。拆块并检查signal、worker或独立进程是不同升级方案；是否值得采用要看测得的阻塞与隔离需求。

生产网络取消还不等于远端事务撤销。例如客户端断开时，远端可能已接受请求或产生副作用；真实支付、订单、工具执行还需要幂等和状态查询。今天只是本地mock，不涉及支付操作，也不提供“超时即可安全重试”的结论。

## 8. 简单方案和升级条件

| 方案 | 适用时机 | 不能省略什么 |
|---|---|---|
| 直接await，无超时 | 本地已知很短、失败边界简单 | 一旦可无限等待就需新策略 |
| race加timer和finally | 只需停止等待、可接受工作继续 | 清timer、观察错误、说明未停止工作 |
| 本课专属controller+单次结算 | 需要caller取消与timeout分类 | 后端必须合作；不能保证硬实时 |
| AbortSignal.timeout/any组合 | 已确认全部底层API接受signal | 仍需处理结果、分类和不合作实现 |
| worker/进程隔离 | 确认存在阻塞或必须强制终止 | 额外通信、恢复、资源清理与安全边界 |

不需要为了一个超时包装器立即引入消息队列或微服务。先把“谁拥有资源、谁停止等待、谁确认结束”说明白。

## 9. 三个小改写练习

1. **补一条取消竞态测试**：caller.abort与timer.fire按两个相反顺序调用；断言先被处理的事件决定结果，另一个事件不会产生第二个结果或残留监听器。
2. **测量取消到停止的距离**：扩展受控mock，记录abort请求与实际complete两个事件顺序。验收非合作mock的两个时点不同；不能用公开结果时间冒充资源释放时间。不必引入真实网络。
3. **为接入点写契约说明**：假设以后接本地推理服务，列出它是否接收signal、断连接是否终止生成、超时后如何确认任务状态。未知项明确写“未验证”，不要随意把真实SDK名写成已支持。

下一课计划回到模型组件线讲Softmax：最大值平移、行规约与数值稳定性。之后再带着本课的生命周期概念进入并发限制与背压。

## 10. 官方来源

核验日期2026-11-03。实际运行版本24.22.1；以下滚动24.x文档当日显示24.22.2，未声称运行最新版。初次直接打开globals/timers遇到工具错误，随后从官方TypeScript页导航成功读取相关正文，未绕过访问限制。
- [Node AbortController/AbortSignal](https://nodejs.org/docs/latest-v24.x/api/globals.html#class-abortcontroller)：取消通知、aborted、once监听与组合信号。
- [Node timers](https://nodejs.org/docs/latest-v24.x/api/timers.html#settimeoutcallback-delay-args)：定时回调不保证精确触发，clearTimeout清理。
- [Node TypeScript支持](https://nodejs.org/docs/latest-v24.x/api/typescript.html#type-stripping)：可擦除语法直接运行，不做类型检查。

未读取用户仓库作为课程依据；没有将旧实验、模拟调用或未执行的验证标为真实模型集成。
