# 第1课 从残差加法写出第一个 Triton 算子

日期：2026年10月30日，星期五（Asia/Shanghai）。预计学习 45–60 分钟，仅为估计，不表示你已经掌握前置内容。

今天只做一件事：把大模型里常见的 **残差相加** 写成可读、可测试的 Triton kernel，并把索引、数据搬运和尾块逐步对应起来。它不是新的网络结构，也不承诺比 PyTorch 更快。源码附件 `lesson_01_residual_add.py` 与本讲义中的完整代码一致。

## 1 学习目标和前置知识

读完并改写后，应能解释：

1. 一个 Triton program 处理哪些元素，为什么 program 不等于一个 CUDA thread。
2. `tl.load`、局部计算和 `tl.store` 分别在做什么；哪些地方并没有显式使用 shared memory。
3. 为什么最后一块要给 load 和 store 都加 mask，空张量为什么不发射 kernel。
4. 如何分别检查语义正确、尾块正确与性能，而不是只看程序没有报错。

前置只需要 Python 函数、数组下标、张量 shape、FP32，以及 GPU kernel 是由 CPU 发起的异步工作。暂时不需要理解 Tensor Core、warp shuffle 或 FlashAttention。

## 2 它在大模型的哪里

一个简化的残差子层可写为：

$$
y=x+F(x).
$$

这里 $F$ 可以是 attention 或 MLP 子层。本课假定 $r=F(x)$ 已经算好，只实现：

$$
y_i=x_i+r_i,\quad 0\le i<N.
$$

- **网络层**决定 $F$ 是什么、残差放在哪里。
- **PyTorch 参考**表达 `torch.add(x, r)` 的数学语义。
- **本课 kernel**负责两个已存在张量的逐元素相加。
- **推理调度**决定何时执行、在哪个 stream 或哪个请求批次执行；本课不实现服务调度。

不要把 `x+r` 叫作多算子融合：这里只有一次相加。PyTorch 的加法本身通常也由一个算子实现；改写成 Triton 并不会自动获得“融合加速”。

### 本课的严格输入输出契约

| 项目 | 约定 |
|---|---|
| shape | 两输入完全相同，如 `[B,T,H]`；不支持 broadcasting |
| dtype | 两输入均为 `torch.float32`；输出仍为 float32 |
| layout | 两输入都是普通 dense、默认 contiguous，且无 lazy-negative 标记；不处理转置 view 或任意 stride |
| device | kernel 路径要求同一 CUDA device；CPU 参考可在 CPU 运行 |
| 输出 | 新分配、与输入同 shape，不做原地修改，不与输入共享输出存储 |
| 训练 | 教学推理算子，不注册 backward；输入 requires_grad=True 时明确拒绝 |
| 空输入 | 输出同 shape 空张量，不启动零 grid |

例如 `[2,3,7]` 有 $N=42$ 个元素。只因它是连续存储，我们可以按一维地址处理，再让返回张量保留原来的三维 shape。**shape 与地址步长是两回事**；不能直接把本实现套到转置后的张量。

## 3 环境和实际验证范围

**GPU 参考基线，不是“最新版”声明：**Linux x86_64、PyTorch 2.5.1 的 CUDA 12.1 构建、Triton 3.1.0，以及与该 CUDA 构建兼容的 NVIDIA 驱动。PyTorch 官方历史安装页列出该 cu121 构建，v2.5.1 官方源码中的 Triton pin 为 3.1.0。[来源1][来源2]

RTX 4060 可作为你的目标示例：官方 compute capability 为 8.9，即常写的 `sm_89`。[来源3] 本课不假定你已经有远程 GPU，也不替你安装依赖或配置驱动。Windows 用户应使用已经配置好的兼容 Linux／WSL2 CUDA 环境来尝试本基线，不把它当成已验证的原生 Windows Triton 安装方案。

**Mac M4 不能原生执行这份 CUDA kernel。** 可以在 Mac 跑纯 Python 索引模拟；已有 CPU PyTorch 时也可跑语义参考。MPS／Metal 是不同后端，不是把 `device='cuda'` 改成 `'mps'` 就能运行这份 Triton kernel。[来源4]

本期实际完成的验证：

- Python 语法编译检查通过。
- 标准库索引模拟的 6 个长度用例通过：0、1、255、256、257、1025；同时检查每个有效元素恰好写一次。
- 当前验证环境没有 PyTorch、Triton。因此 **PyTorch CPU 测试、GPU 编译与正确性测试、性能测量均未执行**，没有 GPU 加速比或带宽结果。
- Python 整数模拟只验证索引覆盖与尾块逻辑，不验证 GPU 指令、float32 舍入、内存事务或 JIT 编译。

在已有相应环境中运行：

```bash
python lesson_01_residual_add.py          # 无第三方依赖也能跑索引模拟
python lesson_01_residual_add.py --gpu    # 另跑真正的 Triton/CUDA 测试
python lesson_01_residual_add.py --bench  # GPU 测试通过后，另测预分配输出版本
```

默认命令在 PyTorch 存在时会额外跑 CPU 参考测试；不存在时明确打印 SKIP。GPU 模式缺环境会报错退出，**不会悄悄改成 CPU 并称作 GPU 通过**。脚本会打印已安装版本与实际 GPU，便于记录复现实验环境。

## 4 先看最小语义 再看核心 kernel

PyTorch 语义只需 `y = torch.add(x, r)`。完整脚本中的 `torch_reference` 先做本课契约检查，避免 broadcasting、dtype promotion 等框架便利特性掩盖自己 kernel 的边界。

核心 kernel 是这几行：

```python
pid = tl.program_id(axis=0)
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
valid = offsets < N
x = tl.load(X + offsets, mask=valid, other=0.0)
r = tl.load(R + offsets, mask=valid, other=0.0)
y = x + r
tl.store(Y + offsets, y, mask=valid)
```

正式可执行定义和 wrapper 见后面的完整源码；这里不省略 `@triton.jit` 的必要性。

### 4.1 program 和逻辑元素

本课 `BLOCK=256`，grid 为 $\lceil N/256\rceil$。program $p$ 的逻辑索引是：

$$
\mathrm{offsets}=256p+[0,1,\ldots,255].
$$

`tl.arange` 产生一组逻辑元素索引，不是创建 256 个 CUDA thread。普通配置 `num_warps=4` 表示 4 个 warp，即 128 个线程协作处理这个 program；256 个逻辑元素如何分配给线程、寄存器与指令由编译器布局决定。**不要由此固定推断“线程0一定处理0和128”或“一个program就是一个thread”。**[来源5][来源6]

### 4.2 N=257 时最后一块

- program 0：offsets 为 0–255，全部有效。
- program 1：offsets 为 256–511，仅第一个位置有效。
- `load(..., mask=valid, other=0.0)` 只读取有效地址，无效 lane 的逻辑值用 0 填充。
- `store(..., mask=valid)` 只写入有效地址；**只保护 load、忘记保护 store 仍会越界写。**

`N=0` 时，wrapper 仍返回空输出，但直接跳过 launch；无需依赖零 grid 的后端行为。

### 4.3 数据怎样移动

`X/R/Y` 是设备内存地址，`X + offsets` 表示按元素类型偏移的地址集合，不是 Python 列表切片。连续索引有利于合并访存，但实际事务数量还依赖地址对齐、cache 和编译后的访问形式，不能仅由源码断言“只发一个128B事务”。

`tl.load` 通过设备内存层次读取输入，逻辑值参与加法，`tl.store` 写回输出。局部值通常由编译器放入寄存器，但是否溢出及最终指令应以生成代码／分析工具为准。本 kernel **没有显式 shared memory tile**，不能把 `tl.load` 翻译成“先装到 shared memory”。

每个输出元素只依赖对应的两项输入，program 间没有通信，也没有归约，所以无需手写 barrier、原子加或跨 program 同步。此结论只适用于当前独立逐元素工作；之后的规约和 transpose 会有不同的数据依赖。

### 4.4 数值和自动求导边界

本课读入、运算、写出都限定 float32，没有低精度转换或归约重排。随机有限值用 `rtol=1e-6, atol=1e-6` 对照 PyTorch，整数构造用零容差；另提供 NaN／Infinity 用例并启用 `equal_nan=True`。这不是对所有设备上的 NaN payload、signed zero 或 subnormal 行为作 bitwise 一致保证。

没有 backward 的自定义 kernel 不应悄悄混入训练。因此 wrapper 拒绝 requires_grad 输入，而不是给出一个看似正常却断开梯度的输出。未来若用于训练，需要明确接入 autograd 并测试梯度，不能删除检查就算支持训练。

## 5 完整可运行源码

以下代码与同名 `.py` 附件逐字一致。先读 kernel 和 wrapper，再读测试；测试代码比加法本身长是正常的，因为它在声明正确性边界。

```python
"""Lesson 01: contiguous float32 residual add, inference only.

No installation is performed. Default: stdlib simulation + optional Torch CPU tests.
--gpu: also run Triton CUDA tests. --bench: also benchmark preallocated outputs.
"""
import argparse
import math

try:
    import torch
except ImportError:
    torch = None
try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

BLOCK = 256

if triton is not None:
    @triton.jit
    def _residual_add_kernel(X, R, Y, N, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = offsets < N
        x = tl.load(X + offsets, mask=valid, other=0.0)
        r = tl.load(R + offsets, mask=valid, other=0.0)
        y = x + r
        tl.store(Y + offsets, y, mask=valid)


def _validate(x, r, *, cuda):
    if torch is None:
        raise RuntimeError("PyTorch is required for tensor tests/kernels")
    if not isinstance(x, torch.Tensor) or not isinstance(r, torch.Tensor):
        raise TypeError("x and r must be tensors")
    if x.shape != r.shape:
        raise ValueError("shapes must match; this lesson does not broadcast")
    if x.dtype != torch.float32 or r.dtype != torch.float32:
        raise TypeError("only float32 is supported in this lesson")
    if x.device != r.device:
        raise ValueError("devices must match")
    if not x.is_contiguous() or not r.is_contiguous():
        raise ValueError("only contiguous tensors are supported")
    if x.is_neg() or r.is_neg():
        raise ValueError("lazy negative tensor views are not supported")
    if x.requires_grad or r.requires_grad:
        raise ValueError("inference only; no autograd implementation")
    if cuda and x.device.type != "cuda":
        raise ValueError("the Triton kernel requires CUDA tensors")


def torch_reference(x, r):
    _validate(x, r, cuda=False)
    return torch.add(x, r)


def _launch(x, r, out):
    # Internal entry: inputs and output are already validated/preallocated.
    n = x.numel()
    if n == 0:
        return
    grid = (triton.cdiv(n, BLOCK),)
    with torch.cuda.device(x.device):
        _residual_add_kernel[grid](x, r, out, n, BLOCK_SIZE=BLOCK, num_warps=4)


def residual_add(x, r):
    _validate(x, r, cuda=True)
    if triton is None:
        raise RuntimeError("A compatible Triton installation is required")
    out = torch.empty_like(x)
    _launch(x, r, out)
    return out


def simulate_indexing(x, r, block=BLOCK):
    """CPU indexing model, NOT a Triton interpreter or float32 GPU test."""
    if len(x) != len(r) or block <= 0:
        raise ValueError("invalid lengths/block")
    n = len(x)
    out, writes = [None] * n, [0] * n
    for pid in range((n + block - 1) // block):
        for lane in range(block):
            i = pid * block + lane
            if i < n:  # masked loads and store
                out[i] = x[i] + r[i]
                writes[i] += 1
    assert writes == [1] * n
    return out


def test_indexing():
    for n in [0, 1, 255, 256, 257, 1025]:
        x, r = list(range(n)), [-2 * i for i in range(n)]
        assert simulate_indexing(x, r) == [-i for i in range(n)]
    print("PASS: 6 stdlib indexing/tail/unique-write cases (not GPU execution)")


def expect_error(fn, kind):
    try:
        fn()
    except kind:
        return
    raise AssertionError(f"expected {kind.__name__}")


def test_tensors(device, use_kernel=False):
    operation = residual_add if use_kernel else torch_reference
    shapes = [(0,), (1,), (255,), (256,), (257,), (1025,), (3, 5), (2, 3, 7), ()]
    for shape in shapes:
        n = math.prod(shape)
        x = torch.arange(n, dtype=torch.float32, device=device).reshape(shape)
        r = (-2 * torch.arange(n, dtype=torch.float32, device=device)).reshape(shape)
        expected = (-torch.arange(n, dtype=torch.float32, device=device)).reshape(shape)
        y = operation(x, r)
        torch.testing.assert_close(y, expected, rtol=0, atol=0)
        assert y.shape == x.shape and y.dtype == x.dtype and y.device == x.device
        if n:
            assert y.data_ptr() != x.data_ptr() and y.data_ptr() != r.data_ptr()
    generator = torch.Generator(device=device).manual_seed(7)
    x = torch.randn((1025,), device=device, dtype=torch.float32, generator=generator)
    r = torch.randn((1025,), device=device, dtype=torch.float32, generator=generator)
    torch.testing.assert_close(operation(x, r), x + r, rtol=1e-6, atol=1e-6)
    a = torch.tensor([float("inf"), -float("inf"), float("nan"), 0.0], device=device, dtype=torch.float32)
    b = torch.tensor([-float("inf"), 1.0, 1.0, -0.0], device=device, dtype=torch.float32)
    torch.testing.assert_close(operation(a, b), a + b, equal_nan=True)
    expect_error(lambda: operation(x, r[:1]), ValueError)
    expect_error(lambda: operation(x.half(), r.half()), TypeError)
    t = torch.zeros((3, 4), device=device).t()
    expect_error(lambda: operation(t, t), ValueError)
    expect_error(lambda: operation(x.clone().requires_grad_(), r), ValueError)
    if use_kernel:
        expect_error(lambda: residual_add(x, r.cpu()), ValueError)
        torch.cuda.synchronize(device)
    print(f"PASS: tensor cases on {device}; kernel={use_kernel}")


def benchmark():
    import triton.testing
    device = torch.device("cuda", torch.cuda.current_device())
    n = 2**22
    x = torch.randn(n, device=device, dtype=torch.float32)
    r = torch.randn_like(x)
    yt, yp = torch.empty_like(x), torch.empty_like(x)
    _validate(x, r, cuda=True)
    _launch(x, r, yt)  # compile before measurement
    torch.testing.assert_close(yt, x + r, rtol=1e-6, atol=1e-6)
    fns = {"triton_preallocated": lambda: _launch(x, r, yt),
           "torch_add_out": lambda: torch.add(x, r, out=yp)}
    for fn in fns.values():
        for _ in range(20):
            fn()
    torch.cuda.synchronize(device)
    for name, fn in fns.items():
        median_ms, p20_ms, p80_ms = triton.testing.do_bench(
            fn, warmup=25, rep=100, quantiles=[0.5, 0.2, 0.8])
        gbps = (12 * n) / (median_ms * 1e6)
        print(f"{name}: median={median_ms:.6f} ms, "
              f"p20={p20_ms:.6f}, p80={p80_ms:.6f}, logical_GB/s={gbps:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--bench", action="store_true")
    args = parser.parse_args()
    test_indexing()
    if torch is not None:
        print(f"torch={torch.__version__}; bundled CUDA={torch.version.cuda}")
        test_tensors(torch.device("cpu"))
    else:
        print("SKIP: PyTorch CPU reference tests (PyTorch not installed)")
    if args.gpu or args.bench:
        if torch is None or triton is None or not torch.cuda.is_available():
            raise SystemExit("GPU path unavailable: compatible torch+Triton+CUDA GPU required")
        device = torch.device("cuda", torch.cuda.current_device())
        print(f"triton={triton.__version__}; GPU={torch.cuda.get_device_name(device)}; "
              f"capability={torch.cuda.get_device_capability(device)}")
        test_tensors(device, use_kernel=True)
        if args.bench:
            benchmark()


if __name__ == "__main__":
    main()
```

## 6 怎样测试 才知道没有漏掉尾块

完整脚本按三层分开验证：

1. **无依赖索引模型**：验证每个有效下标写一次、不访问无效下标。长度恰好等于 block 的用例，不能代替 block+1 的用例。
2. **CPU PyTorch 语义**：检查 shape/dtype、手算整数结果、标量和 2D/3D，以及错误输入的拒绝行为。这一层不证明 Triton 正确。
3. **真实 GPU 路径**：同样的用例换成 `residual_add`，与 PyTorch 对照；最后同步设备，让异步执行错误暴露。CPU/GPU 输入混用也要拒绝。

小张量不够代表所有情况；下一步可以加入大于多个 block 的随机长度、不同 contiguous shape 和多个 CUDA device。不要因为当前样例通过，就宣称任意 stride、广播、FP16 或 backward 已受支持。

## 7 性能测量 不用“看起来更快”作结论

本例每个元素逻辑上读取两个 FP32，写入一个 FP32，理想显式流量为：

$$
\mathrm{Bytes}_{\mathrm{logical}}=N(4+4+4)=12N.
$$

假设实测中位延迟为 $t_{\mathrm{ms}}$，有效带宽为：

$$
\mathrm{GB/s}_{\mathrm{logical}}=\frac{12N}{t_{\mathrm{ms}}\times10^6}.
$$

这是十进制 GB/s。它不是 Nsight 统计的实际 DRAM 字节数；cache 命中、事务放大和测量方法都会影响解释。一次相加约 1 FLOP，因此理想算术强度仅 $1/12$ FLOP/byte；大输入常受数据搬运约束，小输入还可能主要受 launch 开销影响，不能从这个比值直接推导实测瓶颈。

`--bench` 的设计：

- 先编译 kernel、跑正确性，再显式 warmup 和 synchronize；不把首次 JIT 时间混入稳定执行延迟。
- 两边输入相同，输出均预分配。比较内部 `_launch` 与 `torch.add(..., out=...)`，不让一边计分配、一边不计。
- `do_bench` 显式使用 `quantiles=[0.5,0.2,0.8]`，返回值依这个顺序对应 p50、p20、p80；`warmup=25`、`rep=100` 是**毫秒预算，不是迭代次数**。[来源7]
- 这里使用 CUDA event 对预分配后的调用路径计时；`_launch` 仍有 Python／device-context 包装，不能视为纯硬件 kernel 延迟。短 kernel 的结果可能受 CPU 发射和上下文开销影响。分配与公开 wrapper 的参数验证均未计入，也不代表首次编译耗时或完整 Transformer 端到端速度。若要比较端到端，就另建两边都包含分配的测量，分开命名。
- 换 256／512／1024 元素块时，先跑正确性，再重复多轮测量。不要只保留最好一次，记录版本、GPU、shape、dtype、温度／其他负载和统计分位数。

本期没有实际 GPU 性能数字。`do_bench` 的具体实现随版本变化，所以讲义给出参考版本并显式选 quantiles，不依赖“默认究竟是 mean 还是 median”的文档表述。[来源7]

## 8 最常见的五个错误

- 把 `BLOCK=256` 当成 256 个线程：它是逻辑元素数，num_warps 是另一项配置。
- 忘记 store mask：读没越界不代表写安全。
- 为“方便”默默调用 `.contiguous()`：这会引入拷贝与额外内存，改变 API 成本；本课选择显式拒绝。
- 用 CPU `time.time()` 包住一次异步 launch：很可能主要测到提交工作，而非 GPU 完成。
- 看到 Triton 输出存在就用于训练，或宣称它比 PyTorch 必然快：本课既没有 backward，也没有实测加速证据。

## 9 三个小改写练习

1. **换 block，保留契约。**将 BLOCK 改为 128、512，补长度 BLOCK−1、BLOCK、BLOCK+1、3×BLOCK+7。先预测每个 grid 和最后一块有效元素数，再运行。提示：不需要随 BLOCK 机械同比增加 num_warps。
2. **写一个真正的两步融合。**实现 $y=x+\alpha r$，把 Python 标量 $\alpha$ 传给 kernel；参考用 FP32 PyTorch。提示：乘加是否被编译为 FMA 可能影响舍入，因此先写清比较容差，不承诺与“先乘再加” bitwise 相同；性能基线也要包含相同数学工作。
3. **把拒绝 stride 改成显式支持一维 stride。**仅支持同长度 1D 输入，传入各自 `stride(0)`，读地址改为 `X+offsets*s_x`、`R+offsets*s_r`，输出保持连续。提示：mask 仍按逻辑元素个数判断；用 `base[::2]` 测试，并在性能结果里单独标注非连续输入。

这是实现改写练习，不是30题试卷，也不需要空白答题卡。未交回答案前不会假定你已经完成或掌握。

## 10 从这一课到 MTP 的学习顺序

- **当前：**残差加法，建立 program、地址、mask、测试与计时习惯。
- **接下来：**行内 sum/max 规约；再做 RMSNorm／LayerNorm、稳定 Softmax 与 online softmax 状态。
- **数据布局与计算复用：**transpose、scan；GEMM 的块划分与复用；Embedding gather、输出 projection。
- **网络组件：**RoPE、SwiGLU；Attention／GQA，再逐步拆解 FlashAttention 的分块、数值和资源权衡。
- **稀疏执行与状态：**MoE router、token dispatch/combine，KV 的读写、布局与调度边界。
- **MTP：**先用 PyTorch 理解多 token 预测模块、目标对齐、mask 与损失，再找可优化的 GEMM／norm 等 kernel；推理时如何生成候选并由主模型验证单独讲。**MTP 是训练／模型设计，不是一个 kernel，也不天然等同投机解码。**DeepSeek-V3 论文与作者仓库将 MTP 用于训练，并讨论其可用于投机解码；不能据此断言本课实现了该模型，或 MTP 必然加速。[来源8]

这是课程顺序，不是已完成清单；复杂组件会拆多日。下一课可从“每行求和”进入规约，复用今天的 shape、尾块与测试方法。

## 来源与版本记录

以下官方网页／源码于 **2026-10-30** 核对。`main` 文档为滚动页面，不冒充固定发行版；可复现代码基线使用下面的版本链接。本期为通用组件教学，不引用用户仓库实现。

[来源1]: https://pytorch.org/get-started/previous-versions/#v251 "PyTorch 官方历史版本 2.5.1 CUDA 12.1 构建"
[来源2]: https://raw.githubusercontent.com/pytorch/pytorch/v2.5.1/.ci/docker/triton_version.txt "PyTorch v2.5.1 官方 Triton pin 3.1.0"
[来源3]: https://developer.nvidia.com/cuda/gpus "NVIDIA 官方 GPU compute capability"
[来源4]: https://docs.nvidia.com/cuda/archive/10.2/cuda-toolkit-release-notes/index.html "CUDA 10.2 为最后支持 macOS 的 CUDA 发行版 不适用于现代 Apple Silicon CUDA 执行"
[来源5]: https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html "Triton 官方向量加法教程"
[来源6]: https://triton-lang.org/main/python-api/generated/triton.Config.html "Triton Config num_warps"
[来源7]: https://github.com/triton-lang/triton/blob/v3.1.0/python/triton/testing.py#L78-L150 "Triton 3.1.0 do_bench 实现"
[来源8]: https://arxiv.org/abs/2412.19437 "DeepSeek-V3 技术报告 v1 2024-12-27 v2 2025-02-18"

- Triton 指令参考：[program_id](https://triton-lang.org/main/python-api/generated/triton.language.program_id.html)、[arange](https://triton-lang.org/main/python-api/generated/triton.language.arange.html)、[load](https://triton-lang.org/main/python-api/generated/triton.language.load.html)、[store](https://triton-lang.org/main/python-api/generated/triton.language.store.html)。
- PyTorch 语义与测试：[torch.add](https://docs.pytorch.org/docs/stable/generated/torch.add.html)、[torch.testing.assert_close](https://docs.pytorch.org/docs/stable/testing.html#torch.testing.assert_close)。
- [DeepSeek-V3 作者仓库](https://github.com/deepseek-ai/DeepSeek-V3)：用于后续 MTP 阅读定位，不是本期代码依赖。

- [PyTorch v2.5.1 Tensor.is_neg](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/_tensor_docs.py#L2676-L2682)：本课拒绝 lazy-negative view，避免原始指针读取遗漏视图语义。
