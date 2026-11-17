# Lesson 02 — 行求和规约

日期：2026-10-31（Asia/Shanghai）。承接 residual add，下一步 RMSNorm。本目录全部为新建教学样例，不修改用户仓库。

## 文件
- `row_sum.py`：标准库参考/树规约模拟、CUDA Tensor 契约检查、GPU 包装器。
- `row_sum_kernel.py`：模块级 Triton JIT kernel，GPU路径才导入。
- `test_stdlib.py`：不需要任何第三方包的 8 个测试方法。
- `test_torch.py`：可选 PyTorch CPU 3 个、CUDA/Triton 3 个测试方法；环境缺失明确 SKIP。
- `benchmark.py`：仅 GPU，在预分配输出上比较 Triton 与 PyTorch。
- `VALIDATION.md`：本轮实际执行/未执行的验证。
- `requirements-gpu.txt`：历史版本教学基线，不是自动安装脚本。
- `lesson.md`：完整讲义。

## 零依赖路径（Mac M4 / 无 GPU 同样可用）

在解压目录打开终端，已有 Python 3.10+：

```sh
python row_sum.py
python -m unittest test_stdlib test_torch -v
```

示例输出：`reference: [6.0, 3.0]`、`tree simulation: [6.0, 3.0]`。
缺 PyTorch/Triton 时仍可运行，相关测试会显示 SKIP。这不表示 GPU 测试通过。

## 已有兼容 GPU 环境时

声明基线：Linux x86_64、Python 3.10–3.12、PyTorch 2.5.1 + cu121、Triton 3.1.0、兼容 NVIDIA 驱动。沿用首课的历史版本基线，不宣称最新，也未在本轮验证该组合。RTX4060/sm_89 可作为学习目标，不能据此保证本地驱动/WSL配置兼容。Mac M4 不能原生执行 CUDA；不假定你有远程 GPU。

本次不安装依赖、不下载权重。已配好环境者运行：

```sh
python -m unittest test_stdlib test_torch -v
python benchmark.py --rows 4096 --cols 1025 --device 0
```

GPU benchmark 运行前会做一次正确性比较和编译；输出设备、版本、p50/p20/p80 毫秒与逻辑 GB/s。实际性能未知。

## 契约

输入 `[M,N]` 普通 dense contiguous FP32 CUDA Tensor，禁止 requires_grad 和 lazy-negative view；`0 <= N <= 4096`、`M <= 2^31-1`。有限值且中间累加不溢出为调用者前置条件，热路径不扫描。输出 `[M]` FP32，新分配、不改输入。M=0 返回空；N=0 返回全零。后者可触发 PyTorch 的填零工作，但不发射本课自定义 kernel。

不支持反向传播、非连续输入、长行分块、多GPU规约。性能测试内部 `_launch` 不做检查，不应作为外部公共API。
