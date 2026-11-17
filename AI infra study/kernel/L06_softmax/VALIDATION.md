# 验证记录 2026-11-04

- 实际执行：Python 3.12.14；标准库测试12项通过；依赖测试6项跳过；6份Python源码 py_compile 通过；demo运行通过。
- 样例概率：[0.090031, 0.244728, 0.665241]；等值行得到约1/3。
- 未执行：PyTorch CPU参考、Triton JIT编译、CUDA数值测试、GPU基准、跨平台验证、backward。
- 原因：当前执行环境没有 torch/triton；没有安装依赖。没有真实模型、网络API、训练或部署。
- 标准库模拟是Python浮点数学与mask语义验证，不是float32归约顺序、硬件exp或GPU编译器测试。
- 测试输出见test_output.txt。不能据此声称GPU正确或比PyTorch快。
- 依赖组合是声明的复现目标，不是本轮已验证兼容矩阵。
