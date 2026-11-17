# 本轮验证记录

日期：2026-11-07。环境：Linux x86_64，CPython 3.12.14。
必需第三方依赖：无。PyTorch/Triton 未安装，没有安装任何软件。

## 实际完成

- `python3 -m unittest -v test_attention audit_checks`：19 项，其中 17 PASS、2 SKIP。
- 原测试 11 PASS，另 2 项可选 PyTorch 对照 SKIP；独立审计 6 PASS。
- 原测试包含固定种子 45 组随机形状 × 6 个 block_size（270 次流式与物化对照），70 位 Decimal 独立计算，非法输入、mask、未来 value 隔离、全屏蔽行/首块和概率不变量。
- 独立审计核对讲义三组手算、故意漏掉分子重缩放的 mutant、公共 score 偏移、乱序绝对位置与 padding/holes、屏蔽极大 score、D 与 Dv 不同的缩放。
- 5 份交付 Python 源码的 py_compile 通过；example.py 通过。
- benchmark.py 在计时之前执行输出容限校验，之后预热和计时。参数 L=16/S=64/D=32/Dv=16，streaming block=8，重复 5 次、各方法预热 1 次。
- ZIP CRC 检查、全部 10 个条目的源文件字节比对、全新目录解压后的 19 项测试和示例、基准、语法检查均通过。
- Markdown 按 UTF-8 解码；讲义中主实现代码块与 attention.py 字节内容一致。没有模型、网络调用或用户仓库改动。

## 解释范围

`TEST_OUTPUT.txt` 保留一轮本地执行记录。基准数字只描述当时 CPython 小尺寸调用的中位耗时，包括校验、分配、循环。物化函数返回 trace，而 streaming 仅返回 output，输出负担不同；不据此宣称通用加速、GPU/HBM 性能或生产请求时延。

## 没有验证

PyTorch CPU 对照未执行；GPU/Triton JIT/FP16/BF16/FP32 kernel、反向传播、MPS/ROCm、其他 Python 版本和操作系统、真实模型与生产负载均未测试。本课不含 Triton/CUDA kernel。
Python 浮点和 Decimal 检查不能代替 GPU 正确性，SKIP 不算 PASS。随机覆盖不是形式化证明。
可选 PyTorch 接口按已读取的官方文档解释，未建立经过实测的版本兼容矩阵或 lockfile。
没有付费调用、训练、下载模型、部署或安装。
