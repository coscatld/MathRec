# 第5课：超时不等于取消

已实测 Linux x64 / Node.js v24.22.1。仅Node内建API，没有npm依赖。macOS/Windows的同版Node预计可用，未跨平台实测；不需要GPU。使用可擦除TypeScript，直接运行不进行类型检查；未运行tsc。

解压后在当前目录运行，不需要安装：
```sh
node --test timeout.test.ts
node demo.ts
node benchmark.ts
```
也可使用已有npm的 `npm test` / `npm run demo` / `npm run bench`。

Demo预期：
```json
{"result":{"ok":false,"code":"TIMEOUT"},"abortRequested":true,"workStopped":false}
{"workStoppedAfterCompletion":true}
```
第一行表明调用者已收到超时，但不合作的mock尚未结束；第二行是测试主动完成mock后状态。手动计时器只是测试设施，不代表实际经过100毫秒。

13项测试应全部通过，0跳过。基准数值随环境变化，没有固定耗时验收阈值。

文件：contract.ts是昨日输入/输出验证；timeout.ts是本课完整编排；fakes.ts提供单调用受控mock与手动计时器；timeout.test.ts包含确定性故障测试及真实timer冒烟；demo.ts；benchmark.ts；package.json；讲义；VALIDATION.md。

边界：信任进程内backend/options/AbortSignal/scheduler；不处理恶意Proxy/getter、抛异常的abort监听器或同步回调的错误scheduler。输入按JSON形状。timeoutMs整数1..60000，prompt与输出仍按UTF-16代码单元，不是token。计时从验证后设置timer开始，不保证硬实时；事件循环阻塞时回调会推迟。公开结果已超时不代表计算/网络/扣费已经停止。无重试、限流、容量释放、幂等、持久化或真实模型服务。
