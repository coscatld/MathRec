# 第6课 稳定的行 Softmax

离线标准库入口，Python 3.10+：

```bash
python demo.py
python -m unittest -v test_stdlib test_torch
python -m py_compile softmax.py softmax_kernel.py test_stdlib.py test_torch.py demo.py benchmark.py
```

本次实际环境 Python 3.12.14，未安装 PyTorch/Triton，无 GPU 验证。预计 demo：

```text
row: [1000.0, 1001.0, 1002.0] BLOCK: 4 probabilities: [0.090031, 0.244728, 0.665241] sum: 1.0
row: [-2.0, -2.0, -2.0] BLOCK: 4 probabilities: [0.333333, 0.333333, 0.333333] sum: 1.0
```

可选 GPU 路径需要已有兼容 Linux NVIDIA CUDA 环境及 PyTorch/Triton，不自动安装。参考目标组合 Python 3.11、PyTorch 2.5.1、Triton 3.1.0，未在该组合实测；较新组合也需要重新运行测试，不保证全版本兼容。查看 requirements-gpu.txt。RTX4060/sm_89 可作为学习目标，实际运行需本机驱动/匹配 CUDA PyTorch 构建；Mac M4 不能原生执行此 CUDA/Triton 路径。

已有环境下：

```bash
python -m unittest -v test_torch
python benchmark.py --rows 4096 --cols 1024
```

若 GPU 测试显示 skipped，只表示没有执行；不是 GPU 正确性通过。benchmark.py 需要依赖与 NVIDIA GPU，缺少时会报错，不会将 CPU 数字冒充 GPU 性能。对比 31/32/33、127/128/129、1024/1025 列可观察填充影响。

文件分工：
- softmax.py：标准库参考、尾块数学模拟、Tensor 契约、PyTorch 参考、公共 GPU 包装。
- softmax_kernel.py：内部已验证输入的 Triton launch。
- test_stdlib.py / test_torch.py：12项无依赖测试与6项依赖测试方法。
- demo.py / benchmark.py：离线示例与 GPU 公平计时。
- VALIDATION.md / test_output.txt：实际验证边界。
- lesson.md：完整中文教学。

只支持二维连续行优先输入、1到4096列、FP16/BF16/FP32、有限值、无梯度；允许0行。输出独立分配、shape/dtype相同。NaN、±inf、语义mask、backward、超宽行和非连续张量均不在本课公共契约内。有限值检查在GPU上会同步，基准使用已验证的内部路径排除这一成本。不要把内部 launch 当成通用安全接口。
