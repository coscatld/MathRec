# 第2课 从逐元素计算走向行求和规约

日期：2026年10月31日，星期六（Asia/Shanghai）。主线预计 40–60 分钟，数值与基准部分可分次完成；这是估计，不表示你已经掌握前置内容。

昨天的 residual add 是“每个位置独立算一个结果”；今天改成“同一行的很多位置共同产生一个结果”。本期聚焦模型算子层。它是后续 RMSNorm、Softmax、Attention 的前置，也用于练习通用架构习惯：**先定义契约，再分离参考实现、优化实现与测试。**

交付：本讲义 + `Coding_架构每日学习_2026-10-31.zip`。全部源码、测试、基准、版本说明都在 ZIP 中；本文核心代码与文件一致。

## 1 学什么，处在系统哪一层

目标：
1. 说清一个 program 负责哪一行、每个逻辑元素对应哪个地址。
2. 区分逐元素计算和 reduction，正确选择尾块的中性元素。
3. 能区分数学正确、浮点误差、实际GPU测试与纯CPU模拟。
4. 用相同输入输出契约比较简单基线和优化版本，而不是把“用了 Triton”当作性能证明。

前置：Python、二维数组、shape、FP32、上一课 program/load/store/mask。暂不要求会写 CUDA warp shuffle。

在模型中，隐藏状态常为 `[B,T,H]`。本课概念上把 B×T 个位置看作 M 行、H 个通道看作 N 列，规约最后一维。例如 RMSNorm 需要先对每行平方值求和，再除以 N；Softmax 也需要行求和。**今天只求和，不实现 RMSNorm，也不是直接对模型隐藏状态求和就完成一个网络层。**

层次边界：网络结构规定求什么；PyTorch 表达语义；Triton 实现数据搬运和规约；推理服务决定何时调度。跨请求并发和 Agent 状态属于后续课程，不由这份 kernel 解决。

## 2 输入输出契约比优化更早

数学定义：

$$
y_r=\sum_{c=0}^{N-1}x_{r,c},\qquad x\in\mathbb{R}^{M\times N},\quad y\in\mathbb{R}^{M}.
$$

| 项目 | 本课约定 |
|---|---|
| shape | 仅二维 `[M,N]`；输出 `[M]`，不保留长度1的维度 |
| dtype | 输入/输出 FP32；规约也使用 FP32 |
| layout | 普通 dense strided 且 contiguous；不隐式调用 contiguous() 掩盖复制成本 |
| device | CUDA wrapper 只接收 CUDA Tensor；CPU 另有纯 Python 参考 |
| 数值前置 | 输入为有限FP32可表示值，累加不溢出；wrapper 不逐次扫描这些条件 |
| 长度 | `0 <= N <= 4096`、`M <= 2^31-1`；N上限是教学上限，不是 Triton 通用极限 |
| 自动微分 | 不实现 backward，requires_grad=True 明确拒绝 |
| 状态 | 新分配输出，不改输入；不支持外部指定out以避免本课展开别名规则 |
| 空输入 | M=0 返回空向量；N=0 返回 M 个0，避免 next_power_of_2(0) 和无意义发射 |

数值前置不扫描是有意取舍：一次 `isfinite(...).all().item()` 会引入额外GPU工作和主机同步。调试入口可加扫描，热路径与计时不应偷偷包含它。若生产接口必须接收任意 NaN/Inf/极端值，则要另行定义传播、溢出和报错策略，不能沿用本课的有限值正确性结论。

### 最小 PyTorch 语义参考

```python
import torch
x = torch.tensor([[1., 2., 3.], [-1., 0., 4.]], dtype=torch.float32)
y = torch.sum(x, dim=1)  # [6., 3.]，shape [2]
print(y)
```

PyTorch CPU 参考无需 CUDA，但本轮环境没有安装 torch，所以这段没有实际执行。ZIP 的标准库入口提供同样例子，可在无GPU环境阅读/运行。

## 3 先用 3×5 的矩阵想清楚索引

有3行，每行5个数：发射3个 program。program 0算第0行，program 1算第1行，program 2算第2行。

对每个 program，令 BLOCK=8，为一行构造8个**逻辑位置**：0、1、2、3、4、5、6、7。只有前5个地址有效，后三个位置填0，然后一起规约成一个数。

连续行的地址通常是 `X + row*N + col`；代码传 `x.stride(0)`，正确表达行起点间距，包括某些退化维度的合法stride。列连续，所以列偏移就是 `col`。

- load mask：`cols < N`，不读取越界位置。
- 填充值：0，因为 `a+0=a`；不能照搬 Softmax 最大值规约的负无穷。
- store：每个 program 仅写 `Y[row]`。grid恰好M个program，没有多余行，因此这里不需要store mask。
- `num_warps=4` 在 NVIDIA 上表示 program 的线程配置，不表示“4行”，也不表示 BLOCK=128。BLOCK 是逻辑元素数量，编译器把逻辑值分配给线程。

## 4 完整实现：先读kernel，再看wrapper

以下是 ZIP 中的完整 `row_sum.py`，既支持零依赖参考演示，又延迟导入GPU依赖。`_launch` 是内部已验证路径，不是可以随便传入任意out的公共接口。

```python
"""Lesson 02: row reduction; strict FP32 inference contract. No third-party imports on CPU path."""
import math
import struct

MAX_COLS = 4096


def f32(x):
    return struct.unpack('f', struct.pack('f', x))[0]


def reference_rows(rows):
    """High-accuracy Python reference, NOT a GPU implementation."""
    return [math.fsum(row) for row in rows]


def simulate_row(row):
    """One possible FP32 tree, NOT Triton's actual instruction order.

    Requires FP32-representable values and no intermediate overflow."""
    n = len(row)
    if n > MAX_COLS:
        raise ValueError('teaching limit: N <= 4096')
    if any(not math.isfinite(x) for x in row):
        raise ValueError('simulation requires finite values')
    if not n:
        return 0.0
    block = 1 << (n - 1).bit_length()
    values = [f32(row[i]) if i < n else 0.0 for i in range(block)]
    while len(values) > 1:
        values = [f32(values[i] + values[i + 1]) for i in range(0, len(values), 2)]
    return values[0]


def _validate(x):
    import torch
    if not isinstance(x, torch.Tensor):
        raise TypeError('expected torch.Tensor')
    if x.layout != torch.strided or x.ndim != 2:
        raise ValueError('expected dense strided [M,N]')
    if x.dtype != torch.float32:
        raise TypeError('only float32 is supported')
    if not x.is_contiguous() or x.is_neg() or x.is_conj():
        raise ValueError('expected ordinary contiguous tensor')
    if x.requires_grad:
        raise ValueError('inference only; backward is not implemented')
    if x.shape[0] > 2**31 - 1:
        raise ValueError('row count exceeds CUDA grid limit')
    if x.shape[1] > MAX_COLS:
        raise ValueError('teaching limit: N <= 4096')
    if x.device.type != 'cuda':
        raise ValueError('GPU wrapper requires CUDA; use reference_rows on CPU')



def _launch(x, out):
    """Internal unchecked path for benchmark: validated M>0,N>0 and matching out."""
    import triton
    from row_sum_kernel import kernel
    kernel[(x.shape[0],)](
        x, out, x.stride(0), x.shape[1],
        BLOCK=triton.next_power_of_2(x.shape[1]), num_warps=4)


def row_sum(x):
    """[M,N] -> [M]. Finite inputs, no overflow assumed; NOT runtime-scanned.

    Metadata is checked. No autograd, no dtype conversion, no implicit contiguous copy.
    """
    import torch
    _validate(x)
    m, n = x.shape
    with torch.cuda.device(x.device):
        if m == 0:
            return torch.empty((0,), dtype=x.dtype, device=x.device)
        if n == 0:
            return torch.zeros((m,), dtype=x.dtype, device=x.device)
        out = torch.empty((m,), dtype=x.dtype, device=x.device)
        _launch(x, out)
        return out


if __name__ == '__main__':
    rows = [[1., 2., 3.], [-1., 0., 4.]]
    print('reference:', reference_rows(rows))
    print('tree simulation:', [simulate_row(row) for row in rows])

```


GPU kernel 放在独立 `row_sum_kernel.py`，由wrapper延迟导入。Triton 3.1.0解析模块级全局名字；不要把tl作为局部import藏在嵌套jit函数闭包中。

```python
"""Imported only by the GPU launch path; globals are visible to Triton 3.1.0 JIT."""
import triton
import triton.language as tl

@triton.jit
def kernel(X, Y, ROW_STRIDE, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    vals = tl.load(X + row * ROW_STRIDE + cols, cols < N, other=0.0)
    total = tl.sum(vals, axis=0)
    tl.store(Y + row, total)
```

### 逐段看数据流

1. `row = tl.program_id(0).to(tl.int64)`：先提升为int64再计算行地址，避免大矩阵行偏移的int32乘法溢出；当前program唯一负责一行，program之间互不累加同一输出，所以没有atomic_add。
2. `tl.arange(0, BLOCK)`：生成逻辑列索引。BLOCK在编译期确定为不小于N的2次幂，N=1025时BLOCK=2048。Padding只存在于计算逻辑，不是把输入张量真的复制扩容。
3. `tl.load(..., other=0)`：有效位置从GPU全局内存加载；无效位置贡献0。这里没有显式分配共享内存数组。
4. `tl.sum(vals, axis=0)`：把一维逻辑向量变为标量。编译器完成线程间所需的规约实现与同步；我们没有手写barrier，不代表硬件上没有通信或同步。不能把这个Python式表达理解成CPU for循环。
5. `tl.store(Y+row,total)`：每行写一个FP32结果。多个program可按任意次序执行，不影响各行独立结果。
6. wrapper：先拒绝不支持的输入，再处理空shape，再分配并发射。调用通常异步返回Tensor；需要值或计时终点时才同步。N=0时的`torch.zeros`可能有PyTorch填零工作，仅表示不发射本课自定义kernel。

本代码沿用 Triton **3.1.0** 签名：`tl.sum(vals, axis=0)`，没有传新版文档可能出现的dtype参数。因为输入本来就是FP32，无需借该参数提升精度。[固定版本实现](https://github.com/triton-lang/triton/blob/v3.1.0/python/triton/language/standard.py#L242-L247)

## 5 为什么加法也会算出不同结果

实数加法可结合，浮点加法会舍入。用四个数说明消去：

$$
10^8+1-10^8+1=2\quad\text{（精确实数结果）}.
$$

如果FP32树先算 `(1e8+1)` 和 `(-1e8+1)`，两个1都可能因精度损失消失，最后得到0。另一种顺序可能得到不同结果。因此不同GPU、编译器、PyTorch和Triton规约顺序不保证逐位一致。

`simulate_row` 使用一种明确的FP32相邻配对树，能展示这件事；它**不是 Triton 内部顺序的模拟器**，不能拿它的具体结果要求GPU完全一致。

验收分两层：
- 普通测试数据：同时对比FP64参考的绝对误差和PyTorch FP32结果；既检查语义又接受有限的舍入差异。
- 大幅消去数据：单独诊断绝对误差，不因参考接近0就使用纯相对误差，也不宽松到只检查没有崩溃。

附件普通随机输入范围为[-1,1]或标准正态，GPU测试使用与行绝对值和相关的容差；极端消去用例专门打印诊断。**这组阈值是教学测试约定，不是所有有限FP32输入的通用误差证明。** 有精度强约束时，可用更高精度累加、补偿求和或重新设计分块，但要测成本。

## 6 测试先于 benchmark

在ZIP解压目录运行：

```sh
python row_sum.py
python -m unittest test_stdlib test_torch -v
```

| 场景 | 要观察什么 | 本轮结果 |
|---|---|---|
| 空行、空矩阵 | 输出shape与0的定义 | 标准库通过；GPU未测 |
| N=31/32/33、255/256/257、1025、4096 | 2次幂边界和尾块中性元素 | 标准库通过；GPU未测 |
| 全1、负数、固定随机种子 | 逐行求和，不混行；误差受控 | 标准库通过 |
| 正负大数消去 | 顺序导致误差，不把FP32当实数 | 标准库示例得到0，对比精确2 |
| FP16、转置、1D、requires_grad、过宽输入 | 明确拒绝，不偷偷复制或降精度 | GPU测试代码已提供，未执行 |
| CPU Tensor误传CUDA wrapper | 清楚报错 | PyTorch未安装，未执行 |

GPU测试主体在 `test_torch.py`；核心检查相当于：

```python
x = torch.randn((7, 257), device="cuda", dtype=torch.float32)
y = row_sum(x)
ref64 = x.double().sum(dim=1)
tol = 2e-6 * x.double().abs().sum(dim=1) + 1e-5
assert bool(((y.double() - ref64).abs() <= tol).all())
torch.testing.assert_close(y, x.sum(dim=1), rtol=1e-4, atol=1e-4)
```

本轮实际执行：**8个标准库测试方法通过；6个依赖测试跳过；5份Python源码语法检查通过。** 未安装torch/triton，没有执行PyTorch CPU计算、Triton JIT、GPU访问测试或性能测量。SKIP不是PASS，Python语法通过不是GPU编译通过。详细记录见 `VALIDATION.md`。

注意这里“消去”指正负数相减导致的浮点问题，不是取消请求。GPU kernel一旦提交，本课没有服务层取消机制；未来讲Agent超时/取消时，需要区分停止等待与真正停止设备计算。

## 7 怎么测，才不把分配/JIT算成算子速度

仅在已有兼容GPU环境下：

```sh
python benchmark.py --rows 4096 --cols 1025 --device 0
```

基准先对两条路径分别做与行绝对值和相关的FP64误差检查，并完成首次编译，预分配两个输出；计时对比 `_launch(x,y)` 与 `torch.sum(x, dim=1, out=ref)`。公共wrapper的校验与输出分配被两边共同排除。它是**预分配执行路径比较**，不是端到端API耗时，更不是完整模型/Agent请求时延。

使用 `do_bench(..., warmup=25, rep=100, quantiles=[.5,.2,.8])`：前两个参数单位是毫秒，不是循环次数；明确输出p50、p20、p80时间。基于Triton3.1.0源码，该工具包含同步和缓存清理策略。它不是简单的CPU秒表，也不等同于真实服务的热缓存工作负载；记录设备、版本和shape，跨方案使用同一方法。[计时实现](https://github.com/triton-lang/triton/blob/v3.1.0/python/triton/testing.py#L78-L150)

逻辑数据量只计算读输入和写输出：

$$
\mathrm{bytes}=4MN+4M,\qquad
\mathrm{GB/s}=\frac{4MN+4M}{t_{\mathrm{ms}}\times10^6}.
$$

它不是测得的HBM总流量，不含缓存清理等操作的全部流量，不能用它直接声称达到硬件峰值。理论工作量为O(MN)，输出空间O(M)；每个program逻辑临时向量O(BLOCK)，实际寄存器/共享内存占用要看编译结果。M很小会并行度不足，N很大会增加单program资源压力，N刚超过2次幂时padding工作增加。

本轮没有GPU数字，因此**没有加速倍数结论**。下一步可测 `(M,N)=(1,4096),(128,257),(4096,1025)`，观察不同瓶颈，而不是只挑最好看的shape。

## 8 本课的架构思维：把边界写进代码

- **契约层** `_validate`：哪些输入支持、哪些拒绝。好处是错误位置明确，而不是任由kernel产生错误结果。
- **实现层** `_launch`/kernel：只关心已满足约定的数据。测试可直接比较同一语义的PyTorch参考和Triton实现。
- **验证层**：零依赖模拟、CPU参考、GPU正确性、性能分别记录，避免一个“全绿”掩盖环境缺失。

简单替代方案始终是 `torch.sum`：支持面更广、维护成本低。只有实际profile显示这部分值得优化，或它能与后续操作融合降低访存，才有理由承担自定义kernel维护成本。

什么时候升级？N超过4096时，先走PyTorch fallback或明确拒绝；需要支持超宽行时再拆分为“分块局部和→第二阶段规约”。不能直接把多个program同时写同一个Y而不协调，也不应在没有需求时先引入复杂调度。

未来映射到后端同样适用：公共接口负责契约，执行器负责工作，测试负责可观察保证。这里只建立方法，不宣称已实现服务架构。

## 9 三个小改写练习（不是试卷）

1. **sum改mean**：N>0时除以N。先决定N=0返回NaN、0还是拒绝，并写成契约和测试；不要把除0留给偶然行为。检查输出shape和dtype不变。
2. **sum改sum-of-squares**：在load后做平方再规约，对照PyTorch `(x*x).sum(1)`；说明有限输入为何仍可能平方溢出。它是下一课RMSNorm的直接积木。
3. **加一个显式fallback入口**：对N>4096走PyTorch，保留dtype/layout/device约定；打印或返回所用backend供测试验证。不要在底层偷偷复制转置Tensor。验收两条路径输出契约一致、分派边界4096/4097有测试。

下一课计划：从sum-of-squares构建RMSNorm，讲epsilon、权重广播与FP32累加；后续逐步把算子接入模型接口、推理服务和有状态Agent。课程范围已扩大，但每次只推进一个主要能力，不假设你已掌握本课。

## 10 环境与来源

声明兼容基线沿用首期：Linux x86_64、Python3.10–3.12、PyTorch2.5.1/cu121、Triton3.1.0，需兼容NVIDIA驱动。不是“2026最新版本”，也不是本轮实际测试过的组合。Mac M4可跑标准库路径，不能原生CUDA；不假设你已连上远程GPU。不安装软件、不下载模型、不调用付费API、不训练/部署/修改你的仓库。

2026-10-31实际读取的官方资料：
- [Triton官方Fused Softmax教程](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)：行式program与规约组织的参考；本文不复制其速度结论，其persistent循环实现也不是本课单program单行实现。
- [tl.sum当前文档](https://triton-lang.org/main/python-api/generated/triton.language.sum.html)：用于对照当前API；执行代码以[3.1.0固定实现](https://github.com/triton-lang/triton/blob/v3.1.0/python/triton/language/standard.py#L242-L247)为准。
- [Triton3.1.0 testing.py](https://github.com/triton-lang/triton/blob/v3.1.0/python/triton/testing.py#L78-L150)：计时单位、分位数和同步/缓存策略。

- [PyTorch2.5.1 sum/out声明](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/native_functions.yaml#L5189-L5207)与[空归约实现](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/ReduceOps.cpp#L1108-L1131)。
- [PyTorch2.5.1浮点精度说明](https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/notes/numerical_accuracy.rst#L4-L16)：浮点加法不满足结合律、跨实现不保证逐位一致。

本轮PyTorch2.5 sum网页访问失败，使用上述固定源码替代。所有GPU正确性和性能仍待实际运行验证。
