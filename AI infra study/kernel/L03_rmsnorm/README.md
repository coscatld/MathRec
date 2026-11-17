# Lesson 03：RMSNorm

2026-11-01（Asia/Shanghai）。承接残差加法→行求和。新建教学代码，不修改用户项目。

## 零依赖运行
已有 Python 3.10+，解压后在本目录运行：

```sh
python rmsnorm.py
python -m unittest test_stdlib test_torch -v
```

示例输入 `[[3.,4.],[0.,0.]]`、weight=`[1.,2.]`、eps=`1e-5`。
预期打印 `[[0.848528, 2.262741], [0.0, 0.0]]`。
没有PyTorch/CUDA/Triton时相关测试显示SKIP，不等于GPU测试通过。

## 已有GPU环境
声明历史兼容基线：Linux x86_64，Python3.10–3.12，PyTorch2.5.1 CUDA12.1 wheel，Triton3.1.0，兼容NVIDIA驱动；未在本轮实际验证该组合，不宣称最新。requirements-gpu.txt仅记录版本，不是自动安装流程。RTX4060/sm_89可作为目标设备，驱动与系统配置仍需核对。Mac M4不能原生运行CUDA；无需假设有远程GPU。

```sh
python -m unittest test_stdlib test_torch -v
python benchmark.py --rows 4096 --cols 1025 --device 0
```

benchmark先做正确性检查并预热，再输出CUDA event的p50/p20/p80毫秒。包括各方法输出分配，不是孤立kernel成本，不是服务端延迟。此包不执行安装、下载权重、训练、网络或付费API调用。

## 文件
- rmsnorm.py：标准库数学参考、输入检查、PyTorch表达式参考、Triton包装器、示例入口。
- rmsnorm_kernel.py：模块级Triton JIT实现，仅GPU路径导入。
- test_stdlib.py：10项标准库测试。
- test_torch.py：3项CPU PyTorch、3项GPU测试，缺依赖明确跳过。
- benchmark.py：GPU基准脚本。
- requirements-gpu.txt：声明的历史版本。
- VALIDATION.md：实际验证范围。
- lesson.md：完整中文讲义。

## 契约
FP32 dense contiguous x[M,N]、weight[N]，同设备；1<=N<=4096，M<=2^31-1。无autograd，拒绝requires_grad/lazy-negative/conjugate view。eps必须为非bool实数、2^-126<=eps<=1，转为FP32后参与计算。M=0合法返回空，N=0拒绝。输入与权重有限且FP32平方、累加、加eps、最终乘法均不溢出，是调用者负责的数值前置，热路径不扫描。数值容差仅针对测试范围，不是任意输入证明。
