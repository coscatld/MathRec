# 第4课：可替换模型后端与运行时契约

## 环境与依赖
- 已实际验证：Linux x64、Node.js v24.22.1。
- 教学兼容基线：Node.js 24.22.1（24.x）；macOS/Windows 同版本预计可运行，未在这两系统实测。
- 仅使用 Node 内建模块，无 npm 依赖，不需要 npm install、GPU、Python、API key 或网络。
- `.ts` 使用可擦除类型语法。Node 直接运行时不做 TypeScript 类型检查；本轮没有 tsc，未宣称类型检查通过。也没有使用 tsconfig 或构建步骤。

## 运行
在解压目录执行：
```sh
node demo.ts
node --test model.test.ts
node benchmark.ts
```
可选已有 npm 时：`npm run demo` / `npm test` / `npm run bench`，它们只调用上述本地命令；不要执行安装。

## 示例与预期输出
输入：`{requestId:'demo-1', prompt:'hello', maxOutputChars:64}`。
`node demo.ts` 输出：
```json
{
  "ok": true,
  "reply": {
    "requestId": "demo-1",
    "text": "mock:hello",
    "finishReason": "stop"
  }
}
```
测试应为13通过、0失败。基准时间随机器/JIT/负载变化，不提供固定通过数值；应正常结束且结果说明是local mock。

## 文件
- `model.ts`：契约、运行时检查、业务入口、确定性mock。
- `model.test.ts`：13项边界/故障/状态隔离测试。
- `demo.ts`：最小入口。
- `benchmark.ts`：顺序mock调用的批均值基准。
- `package.json`：ESM模式、运行命令、基线版本，无依赖。
- `lesson.md`：完整教学。
- `VALIDATION.md`：实际运行及未验证范围。

## 教学边界
这不是LLM，也不是完整Agent或HTTP服务器。输入按JSON形状处理，不是对任意恶意Proxy/getter执行的沙箱。未知字段丢弃，字符串不自动trim。长度预算按UTF-16代码单元，不是token数，也不是用户可见字符；演示截断可能切开emoji代理对，不适合作为生产文本截断。requestId仅用于对应请求，不提供幂等/去重。无超时、取消、重试、鉴权、限流、持久化或真实provider集成。详见讲义。
