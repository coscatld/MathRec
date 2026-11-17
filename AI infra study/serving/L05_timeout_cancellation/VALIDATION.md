# 实际验证记录
日期2026-11-03（Asia/Shanghai），Linux x64，Node.js v24.22.1。
- node --test timeout.test.ts：13/13 PASS，0skip。
- node demo.ts：与README两行JSON一致。
- node benchmark.ts：1000预热，7批×5000顺序请求，本地立即mock；一次批均值中位数约1.611微秒。不是请求p95、真实模型延迟或加速证据。
- 测试覆盖成功清理、无效输入/options、preabort、调用中取消、timeout先于abort拒绝、非合作继续运行与晚到成功、晚到拒绝无未处理事件、同步throw/异步reject、坏响应、settle-once、微任务竞争、真实1ms timer冒烟（不验证精确时延）。
- 手动scheduler只控制事件顺序，不模拟Node所有队列阶段。fakes是单调用实例，测试每次新建；不能作为共享生产后端。
- 未运行tsc、GPU、真实服务/网络/模型、跨平台、负载压测。未安装依赖、训练、部署或改用户仓库。
