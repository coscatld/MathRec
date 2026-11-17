# 第4课：用接口契约搭起可替换模型后端

日期：2026-11-02，星期一（北京时间）。预计主线40–55分钟，进阶15分钟，均为估计。

## 1. 今天的位置：从算子转到服务边界

前三期依次是残差加法、行求和、RMSNorm。今天按昨日预告，切到并行推进的Agent与后端工程线。已发过不等于你已掌握；今天不要求记住Triton语法。

本课只解决一个问题：**业务代码如何调用模型，而不被某家SDK的请求格式、异常和返回字段绑死？**

渐进项目是一个可替换模型后端的学习助手。今天建立入口与模型适配器边界；后续再加入超时/取消、工具状态机和流式事件。GPU线会回到Softmax等组件，最终再衔接Attention、调度与MTP。模型网络、kernel、服务调度是不同层，本课不硬塞CUDA，也不声称mock模拟了模型智力。

### 学习目标
- 写出输入、输出、错误三个契约，而不是先堆框架。
- 看懂TypeScript接口与JavaScript运行时检查的区别。
- 用依赖注入替换模型实现，并用确定性mock验证业务行为。
- 区分“结果正确”“运行时检查通过”“类型检查通过”“真实集成成功”。

前置：函数、对象、条件判断。若不熟悉async/await，记住：async函数返回Promise，await等待它完成或抛出拒绝原因；不会自动把CPU计算放进新线程。

## 2. 实际需求与限制

同一段业务逻辑将来可能连接本地模型、远程接口或测试假实现。直接在业务函数里调用某家SDK，会让测试依赖网络、密钥、配额与不稳定输出。

今天的最小需求：接收文本，调用一次后端，返回规范化回答；不合法输入必须在调用后端前拒绝，后端坏响应不能直接传给调用方，后端报错不能把内部细节带到公开结果。

默认全离线，无网络服务、API key或GPU。本课不创建支付、训练或部署操作。Mac M4可运行这个Node示例，不需要CUDA。

## 3. 先画清数据契约

| 边界 | 字段与约束 | 为什么这样规定 |
|---|---|---|
| 输入 | requestId：1–64个英文字母/数字/下划线/连字符 | 易于关联请求；不是用户身份，也不是幂等键实现 |
| 输入 | prompt：字符串，trim后非空，原始长度≤4096 | 拒绝空白请求；保留原始空格，不悄悄改写内容 |
| 输入 | maxOutputChars：整数1–512 | 提供明确、易测的输出长度上限 |
| 后端返回 | unknown | 外部数据不能因写了接口就自动可信 |
| 公开成功 | requestId、text、finishReason | 只保留白名单字段，不透传debug字段 |
| 公开失败 | INVALID_INPUT / BACKEND_FAILED / INVALID_OUTPUT | 调用方无需解析任意异常文案 |

**长度单位是JavaScript字符串的UTF-16代码单元。** 名字里的Chars只是教学简写，不等于token数、Unicode码点数或人看到的字符数。例如一个emoji通常占两个代码单元。本课mock直接slice，可能切开代理对；这是明确的教学简化，不可直接当作生产截断策略。生产中应同时考虑模型token预算和Unicode完整性。

本例允许空text，因为模型可能合法地产生空回答；若你的产品要求非空，这是产品契约，应明确改动并测试。未知字段统一丢弃，而不是报错；严格schema也是合理选择，只是兼容性取舍不同。

**输入来源假设为JSON形状的数据或普通对象**，不是带恶意getter/Proxy的进程内对象。网络层若接收JSON，仍需另设请求体字节数上限；这里的prompt长度检查发生在对象已存在之后，不能代替HTTP读入限制。

## 4. 最小完整实现：model.ts

以下就是ZIP中的完整主实现。它不依赖LangChain、LangGraph或模型SDK，先把底层语义看清楚。

```ts
// Teaching boundary: JSON-shaped values, not hostile in-process Proxies/getters.
export type Request = Readonly<{
  requestId: string;
  prompt: string;
  maxOutputChars: number;
}>;
export type Reply = Readonly<{
  requestId: string;
  text: string;
  finishReason: 'stop' | 'length';
}>;
export interface ModelBackend {
  generate(request: Request): Promise<unknown>;
}
export type Result =
  | { ok: true; reply: Reply }
  | { ok: false; code: 'INVALID_INPUT' | 'BACKEND_FAILED' | 'INVALID_OUTPUT' };

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
export function parseRequest(value: unknown): Request | null {
  if (!record(value)) return null;
  const { requestId, prompt, maxOutputChars } = value;
  if (typeof requestId !== 'string' || !/^[A-Za-z0-9_-]{1,64}$/.test(requestId)) return null;
  if (typeof prompt !== 'string' || prompt.trim().length === 0 || prompt.length > 4096) return null;
  if (typeof maxOutputChars !== 'number' || !Number.isInteger(maxOutputChars)
      || maxOutputChars < 1 || maxOutputChars > 512) return null;
  // Copy accepted fields before any await. Unknown fields are intentionally dropped.
  return Object.freeze({ requestId, prompt, maxOutputChars });
}
export function parseReply(value: unknown, request: Request): Reply | null {
  if (!record(value)) return null;
  const { requestId, text, finishReason } = value;
  if (requestId !== request.requestId || typeof text !== 'string') return null;
  if (text.length > request.maxOutputChars) return null;
  if (finishReason !== 'stop' && finishReason !== 'length') return null;
  return Object.freeze({ requestId: request.requestId, text, finishReason });
}
export async function answer(input: unknown, backend: ModelBackend): Promise<Result> {
  const request = parseRequest(input);
  if (request === null) return { ok: false, code: 'INVALID_INPUT' };
  let raw: unknown;
  try {
    raw = await backend.generate(request);
  } catch {
    // Deliberately omit raw provider exceptions from public output.
    return { ok: false, code: 'BACKEND_FAILED' };
  }
  const reply = parseReply(raw, request);
  if (reply === null) return { ok: false, code: 'INVALID_OUTPUT' };
  return { ok: true, reply };
}

// Deterministic mock; no inference, network, randomness, clock, or shared session.
export const mockBackend: ModelBackend = {
  async generate(request: Request): Promise<unknown> {
    const full = `mock:${request.prompt}`;
    return {
      requestId: request.requestId,
      text: full.slice(0, request.maxOutputChars),
      finishReason: full.length > request.maxOutputChars ? 'length' : 'stop',
    };
  },
};
```

### 第一段：类型描述不等于验证

Request的Readonly与接口只提供静态约定。正常TypeScript编译检查时，编译器可发现错误调用；但类型会被擦除，无法阻止外部JSON把数字字段写成字符串。今天使用Node的类型擦除直接运行，**连tsc静态检查都没有执行**，所以尤其不能把“程序跑了”误读成“类型正确”。

unknown强迫我们先检查再使用。不要用`raw as Reply`伪装安全，它只是告诉类型系统相信你，不执行检查。

### 第二段：parseRequest建立一份自己的快照

检查顺序是对象→字段类型→内容约束→复制白名单字段。Number.isInteger同时拒绝NaN、Infinity和小数；没有自动把字符串数字转成整数。

复制避免调用者在await期间修改原对象，Object.freeze进一步限制后端修改这份浅层快照。这里全是字符串与数字，浅冻结够用；以后加入messages数组或metadata对象，必须重新考虑深层复制/不可变数据，不能继续认为浅冻结能保护全部内容。

### 第三段：answer只负责编排

流程是输入解析→一次后端调用→输出解析→规范结果。无效输入直接返回，不消耗后端调用。

try/await/catch同时覆盖后端立即throw和Promise reject。try范围故意集中在后端调用；本课把接入实现视为可信代码，不捕获所有程序bug来伪装业务失败。外部返回数据则通过parseReply校验。

公开错误不带原始异常，有利于避免泄漏内部路径、provider细节或敏感prompt。生产需要受控内部日志与关联ID，但不能默认记录完整学生输入。

### 第四段：后端接口可替换

mockBackend按输入产生固定结果，没有随机数、时钟、共享session，因此同输入可重复验证。它返回Promise，是为了让调用方式贴近未来异步适配器，不是为了假装真实网络请求。

你可以传入另一个含generate方法的对象，answer不需要改。只要后端真正遵守约定，就能替换。**TypeScript结构兼容不等于协议兼容**，所以换真实后端仍须适配字段、检查返回值并做集成测试。

### 第五段：输出验证也是安全边界

返回requestId必须匹配原请求，text必须是字符串且不超过约定上限，结束原因必须在允许集合内。我们不会悄悄截断坏provider响应来伪装成功；超限就是INVALID_OUTPUT。只有mock自己按约定生成受限输出。

finishReason只验证枚举范围，不能凭一个字符串证明远端确实因为长度而停止。真实适配器负责将供应商语义映射到本地语义。

## 5. 数据流与控制流

一次调用中，数据依次经过以下形态：

1. 调用方传入unknown，可能有错误类型或额外字段。
2. parseRequest得到可信、只含三个字段的不可变快照。
3. generate接收快照，返回Promise；业务入口等待结果。
4. 后端结果仍视作unknown，parseReply重新检查。
5. 返回ok为true的规范结果，或ok为false的稳定错误码。

这是一个**调用状态流程**，不是持久化状态机。进程崩溃就没有恢复记录；requestId也没有存入数据库，因此重复requestId会重复执行后端调用。

await允许当前函数暂时让出执行权，但不保证整个应用不会阻塞。若generate内部运行很久的同步计算，事件循环仍会被占用。30个Promise.all请求通过测试只表明请求之间没有串数据，不代表有30条CPU线程。

## 6. 运行与观察结果

环境基线：Node.js **24.22.1**，使用ESM与可擦除类型语法，不是“最新版本”声明。当前官方24.x文档显示24.22.2，固定24.22.1文档链接本轮工具无法读取，以下仅引用已读的24.x官方文档。当前执行环境已运行24.22.1；未在用户Mac/Windows实际运行。

ZIP没有第三方依赖，不需要安装。解压后：

```sh
node demo.ts
node --test model.test.ts
node benchmark.ts
```

package.json中type为module，导入路径明确写`.ts`。我们避开enum、参数属性等需要转换的TypeScript语法，Node只去掉可擦除类型。没有tsconfig，也没有tsc；**本轮不是静态类型检查结果**。有完整工程环境后，再将严格类型检查加入CI，不要以此示例替代它。

最小入口：

```ts
import { answer, mockBackend } from './model.ts';
const result = await answer({ requestId: 'demo-1', prompt: 'hello', maxOutputChars: 64 }, mockBackend);
console.log(JSON.stringify(result, null, 2));
```

预期为ok=true，reply.requestId为demo-1，text为mock:hello，finishReason为stop。没有真实模型在作答。

## 7. 为什么先选简单对象，而不是框架和微服务

**更简单的替代方案：注入一个函数。** 当唯一能力是generate时，直接接收`(request: Request) => Promise<unknown>`完全合理。今天用单方法接口，是为了让“模型后端”有明确命名，也方便下一步增加独立配置与生命周期。

什么时候升级为类？当适配器确实要维护连接、资源清理、明确配置与生命周期时。不是因为“类更高级”。什么时候使用LangChain JS或LangGraph JS？当你需要统一消息模型、工具调用和多步状态流程，且框架的约定能减少实质重复代码时。仍保留自己的业务契约，不将框架对象无差别暴露到所有模块。

今天一个文件足够，不拆微服务、不引入依赖注入容器、不建数据库。真实需求出现时再沿现有边界拆分。

## 8. 故障路径与验收

| 场景 | 可观察结果 | 当前验证 |
|---|---|---|
| 空白prompt、非法ID、超长prompt、非整数预算 | INVALID_INPUT，backend调用次数为0 | 已实测 |
| provider同步throw / 异步reject | BACKEND_FAILED，不返回异常原文 | 已实测 |
| 返回数组/null、错ID、非字符串、过长text、未知结束原因 | INVALID_OUTPUT | 已实测 |
| 正常空text、有额外debug字段 | 允许空串，只公开白名单字段 | 已实测 |
| 调用开始后修改调用方对象 | 后端继续使用原始快照 | 已实测 |
| 同时提交30个不同请求 | 每条结果对应各自ID与prompt | 已实测，仅状态隔离 |
| 后端永不resolve | 当前调用永远等待 | 明确未解决，不声称有超时 |
| 进程崩溃、请求重复 | 当前无恢复/去重 | 未实现 |

测试不能只“调用后打印看看”。ZIP中断言会比较完整规范结果、统计调用次数、测试错误输入并制造异步暂停点。尤其是快照测试用可手动释放的Promise，不靠随机sleep，因此可重复。

## 9. 性能与复杂度：测的究竟是什么

本地一次成功路径包含字段检查、trim扫描、字符串拼接/截断、Promise和少量对象分配。设输入长度为L、输出预算为B，在不计JavaScript引擎字符串内部表示细节的教学模型下，文本处理时间与临时存储规模为O(L+B)；本例L和B都设有上限。这里没有Transformer计算，更没有KV cache。

benchmark先预热1000次，再跑9批，每批顺序等待10000次调用。输出的是**每批平均耗时的中位数与范围**，不是9万次请求的延迟分位数，更不是p95。统计对象不同，不能换名字。

一次本机观测：批均值中位数约0.355微秒/调用。这个数字只表示极小本地mock路径，易受JIT、计时与系统负载影响；不代表真实模型时延，不与网络模型比较，不声称提速。测真实服务时应拆分排队、连接、首token、生成和后处理，并使用真实工作负载与请求级统计。

常见错误：
- 用类型断言代替运行时检查。
- 把mock成功视作第三方集成成功。
- 忘记await导致try/catch抓不到后续reject。
- 将requestId误当已经实现幂等。
- 认为Promise.all等于CPU并行。
- 把UTF-16字符数当作token预算。
- 将批平均值的分位数说成请求p95。

## 10. 三个小改写练习

1. **更换后端，保留业务入口。** 写一个固定返回“请先独立尝试”的后端。预算太小时由适配器生成合法受限文本，不要修改answer。验收：同样的成功契约测试通过，并新增至少一个短预算用例。
2. **修好Unicode截断。** 加一个不会切开代理对的截断函数，但仍以UTF-16代码单元作为预算；测试ASCII、中文、emoji、预算恰好落在emoji中间。明确你是否进一步保证组合字符/字素完整；两者不是一回事。
3. **增加一个产品约束。** 例如禁止空text：写出新的输出规则、修改parseReply与对应测试。先决定应返回INVALID_OUTPUT还是合法空结果，不要让调用方猜。说明这是否改变现有兼容性。

下一课将沿这个后端接口讨论**超时与取消为什么不是同一件事**，继续使用离线可控mock；也会继续穿插模型组件与GPU算子线。

## 11. 实际验证与生产边界

2026-11-02已运行：13项Node测试全部通过，0跳过；demo结果吻合；本地mock基准完成。完整记录见VALIDATION.md。

未执行：tsc类型检查、真实模型/网络集成、GPU、跨平台测试、真实服务并发负载。无安装、付费调用、训练、部署或用户仓库修改。没有验证生产鉴权、租户隔离、超时取消、重试、幂等、日志脱敏与持久化恢复，不能把本课直接部署为公共服务。

## 12. 官方来源与核验日期

读取日期：2026-11-02。以下用于核验API与语义，不是把框架文档当作性能证据。

- [Node.js 24.x TypeScript支持](https://nodejs.org/docs/latest-v24.x/api/typescript.html)：类型擦除、不做类型检查、ESM与显式扩展名、受支持语法。也核对了[当前24.x文档](https://nodejs.org/docs/latest-v24.x/api/typescript.html)，页面版本24.22.2；本课运行仍是24.22.1。
- [Node.js 24.x测试运行器](https://nodejs.org/docs/latest-v24.x/api/test.html)：node:test与异步测试。
- [Node.js 24.x性能API](https://nodejs.org/docs/latest-v24.x/api/perf_hooks.html)：performance.now用于本地计时，单位毫秒。
- [TypeScript Handbook：Everyday Types](https://www.typescriptlang.org/docs/handbook/2/everyday-types.html)：接口、对象类型与类型断言；这是滚动更新文档，不声称本轮安装了某个TypeScript编译器版本。
