# 第6课 稳定的行 Softmax 与融合规约

日期：2026-11-04（Asia/Shanghai）

今天回到 GPU 算子线：把第2课行规约和第3课 RMSNorm 中的“一行一个 program”用于 Softmax。第4、5课的模型接口与超时取消属于服务层；本课属于模型内部的数值计算层，两者可以独立测试。预计主线40–60分钟，进阶计时另需15分钟，均为估计，不意味着你已经掌握前课。

本课要解决一个具体问题：**给一行分数，得到非负、总和接近1的概率，同时避免指数溢出，并尽量减少中间张量写回显存。** 先运行离线参考，再读 kernel；没有GPU也能完成数学、索引和测试设计部分。

## 1 从模型需求到接口契约

Attention 中，某个 query 对多个 key 的分数需要沿 key 维归一化；输出头也可以把词表 logits 转为概率。它们都可能用到 Softmax，但维度、mask、宽度和调用方式不同。本课不实现完整 Attention、采样器或 FlashAttention，只实现二维张量的行 Softmax。

假设 $X$ 的 shape 是 $[M,N]$。每个行号代表一个独立的归一化问题，每个列号是一项候选。输出 $Y$ 仍为 $[M,N]$。

契约如下：

| 项目 | 本课约束 |
|---|---|
| shape | 二维，$M\ge0$，$1\le N\le4096$ |
| dtype | FP16、BF16、FP32；中间计算转FP32，写回原dtype |
| layout | 连续row-major，地址为 `base + row*N + col` |
| 输入值 | 全部有限；不接收NaN、正负无穷 |
| autograd | 仅forward，拒绝 `requires_grad=True` |
| 输出 | 新分配、与输入同shape/dtype，不覆盖输入 |

4096是教学范围，不是Triton的理论极限。限定宽度使单program行规约更容易理解，也避免把一个简单实现冒充所有形状都适用的生产方案。0行返回空输出；0列拒绝，因为没有可归一化项。

特别区分两种 mask：

- **存储尾块 mask**：例如3列填充到4个槽位，第4项根本不存在，不能读取/写入。
- **模型语义 mask**：例如因果Attention不允许读取未来位置。它改变有效候选集合。

本课只实现第一种。全语义mask行如何处理需要单独设计，不能靠“尾块代码已经有mask”就认为支持了。

## 2 为什么必须先减最大值

直接计算为：

$$
y_j = \frac{\exp(x_j)}{\sum_{k=0}^{N-1}\exp(x_k)}.
$$

输入 `[1000,1001,1002]` 会让朴素指数出现溢出。令 $m=\max_j x_j$，则：

$$
y_j = \frac{\exp(x_j-m)}{\sum_k\exp(x_k-m)}.
$$

分子分母同除 $\exp(m)$，实数数学下不改变概率。现在所有指数输入不大于0，且至少有一项为0，因此至少一个指数值为1，分母不为0。极小概率可能下溢为0，这是浮点表示的限制，不等于需要加一个任意epsilon来改变概率公式。

例子：减最大值后是 `[-2,-1,0]`，得到约 `[0.090031,0.244728,0.665241]`。三个分数都加100，不改变数学上的结果；浮点输入先发生舍入时，极端平移未必保持逐bit相同。

与 RMSNorm 对比：RMSNorm除的是平方均值的平方根；Softmax除的是指数的行和。两者都要规约，但中性填充值与数值风险不同，不要机械套用前课的零填充。

## 3 先建立可检查的参考实现

下面来自配套 `softmax.py`，只用标准库。它采用Python浮点，不模拟GPU的FP32计算顺序。

```python
"""Lesson 06: finite-input row softmax. No dependencies for scalar reference."""
import math

def reference_rows(rows):
    """Python float reference, not a float32/GPU emulator."""
    if not rows:
        return []
    n = len(rows[0])
    if n == 0 or any(len(r) != n for r in rows):
        raise ValueError('nonempty rectangular rows required')
    if any(not math.isfinite(v) for r in rows for v in r):
        raise ValueError('only finite logits supported')
    result = []
    for row in rows:
        maximum = max(row)
        numer = [math.exp(v - maximum) for v in row]
        denom = math.fsum(numer)
        result.append([v / denom for v in numer])
    return result
```

观察控制流：先验证输入，再逐行找最大值、求指数、求分母，最后归一化。输出只由同一行决定，因此不同行之间没有通信需求。

PyTorch语义参考很短：

```python
def torch_reference(x):
    import torch
    validate_tensor(x)
    return torch.softmax(x, dim=-1, dtype=torch.float32).to(x.dtype)
```

`dtype=torch.float32`让参考归一化在FP32路径完成，然后转回原dtype。官方文档明确区分沿哪个dim归一化以及dtype转换；不要不写dim，让代码读者猜测。[PyTorch softmax API](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.softmax.html)

检查函数也属于正确实现：非连续输入若仍按 `row*N+col` 寻址，可能读取错误元素。本课选择报错，而不是悄悄 `.contiguous()` 引入复制和额外成本。完整契约检查已在ZIP里，公共GPU包装会在启动kernel前检查shape/dtype/layout/finite/grad。

## 4 Triton主实现

本课采用单行单program的教学版，和官方教程当前展示的persistent多行循环调度不同；不引入occupancy调度或autotune。基础API和稳定计算思路参考官方教程，但本课有独立契约、测试与简化边界。[官方 Fused Softmax 教程](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)

```python
"""Prevalidated, preallocated internal launch; caller owns all contracts."""
import torch
import triton
import triton.language as tl

@triton.jit
def _row_softmax(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    offset = row * N + col
    valid = col < N
    logits = tl.load(X + offset, mask=valid, other=-float('inf')).to(tl.float32)
    peak = tl.max(logits, axis=0)
    weights = tl.exp(logits - peak)
    total = tl.sum(weights, axis=0)
    probs = weights / total
    tl.store(Y + offset, probs, mask=valid)

def launch(x, out):
    """Private path: finite contiguous CUDA input; disjoint matching output."""
    if x.shape[0] == 0:
        return
    n = x.shape[1]
    with torch.cuda.device(x.device):
        _row_softmax[(x.shape[0],)](x, out, n, triton.next_power_of_2(n), num_warps=4)
```

### program与列索引

`tl.program_id(0)`是本program负责的行，代码先转成int64，避免大矩阵的 `row*N` 发生32位索引溢出。grid大小为M。启动时进入 `torch.cuda.device(x.device)` 上下文，避免输入在非当前GPU时启动设备不一致。`col=tl.arange(0,BLOCK)`是一组逻辑列索引，BLOCK取不小于N的最小2次幂。例如N=33，BLOCK=64。

**BLOCK不是CUDA线程数量。** `num_warps=4`在NVIDIA上对应一个program使用4个warp的配置，逻辑元素如何分配给线程由编译器决定，不能把64个逻辑元素直接等同为64个线程。一个线程可能负责多个元素。

### 从显存读入并做尾块保护

`offset=row*N+col`以元素为单位，不需要手动乘dtype字节数；指针类型会决定实际字节地址。连续列对应连续地址，通常有利于合并访问，但具体事务和缓存行为仍需硬件测量。

`tl.load(...,mask=valid,other=-inf)`对不存在的列不读内存，返回填充值。将结果转为FP32后做最大值、指数和求和。`tl.load`的mask保护内存访问，而不是在越界读取之后才修补结果。[load API](https://triton-lang.org/main/python-api/generated/triton.language.load.html)

为什么填 `-inf`？因为它不影响有效有限值的最大值，而且减最大值后指数为0，不污染分母。若填0，真实行 `[-4,-3,-2]`补出来的0会变成最大值，并在分母里占一份概率；最后只写回真实3列，其概率和小于1。配套测试专门捕获这个错误。

### 规约与同步

`tl.max`和`tl.sum`分别把一行逻辑向量变成标量。编译器负责实现program内部需要的线程协作与数据交换，不是每个线程各算自己的分母，也不是Python循环串行求和。[sum API](https://triton-lang.org/main/python-api/generated/triton.language.sum.html)

本实现的各program读写不同的行，不需要跨program同步；没有用户手写的全局原子累加或barrier。真正硬件采用寄存器交换、共享内存或其他策略依编译结果而定，不能仅凭这段源码断言所有中间值永远只在寄存器。

### 指数与写回

`tl.exp`属于快速数值路径，不能要求与Python数学库或PyTorch逐bit一致。最终 `tl.store` 写到输出指针时会转换为输出dtype；FP16/BF16舍入后，概率和只能要求接近1，而非严格等于1。N=1时结果应为1。

NaN和±inf不属于输入契约。特别是整行 `-inf` 会出现 `-inf - (-inf)`，产生NaN。我们选择在公共接口拒绝，而不是暗中定义全mask行为。

## 5 数据流和融合为什么可能有用

一个朴素分步张量程序可能把减最大值后的矩阵、指数矩阵写回显存，再读回来做下一步。本课kernel在一个program中完成这些步骤，逻辑上只加载一次输入并写一次输出，减少中间全矩阵物化。

对dtype字节数为$s$的矩阵，理想主输入输出流量为：

$$
B_{\mathrm{ideal}}=2MNs.
$$

这只是有效数据量模型，不是硬件实际DRAM流量测量。寄存器溢出、缓存命中、填充开销、launch开销都会改变性能。实际中PyTorch自己的Softmax也可能已经高度优化，不能从“融合”二字推导这份教学kernel一定更快。

复杂度约为 $O(MN)$；每个program的逻辑中间空间为 $O(BLOCK)$。N从1024变1025，BLOCK从1024变2048，可能显著改变资源需求。不是输入只增加一个元素，运行成本就必定只增加1/1024。

简单替代方案是直接用 `torch.softmax`。需要通用shape、反向传播、成熟调度时优先用它；只有已测出瓶颈并且可约束输入时，才值得维护专用kernel。超宽行可考虑分块/多阶段归约；把Softmax与Attention的后续乘V融合则是更大范围的问题，不是简单把本函数换个名字。

## 6 先证明行为，再讨论速度

离线运行：

```bash
python demo.py
python -m unittest -v test_stdlib test_torch
```

预计demo输出：

```text
row: [1000.0, 1001.0, 1002.0] BLOCK: 4 probabilities: [0.090031, 0.244728, 0.665241] sum: 1.0
row: [-2.0, -2.0, -2.0] BLOCK: 4 probabilities: [0.333333, 0.333333, 0.333333] sum: 1.0
```

测试分三层：

1. 数学与契约：等值行、单元素、大正值、大负值、空batch、空列、ragged行、非有限值。
2. 索引与填充：1、3、31、32、33、127、257、4096列，对比原始行与补 `-inf` 的数学模拟，并检查填0会失败。
3. 真实张量与GPU：CPU PyTorch参考、GPU的多shape/dtype、有限值校验、行和、误差容限。依赖缺失时明确skip。

GPU断言的核心如下，完整矩阵在 `test_torch.py`：

```python
x = torch.randn(m, n, device='cuda', dtype=dt) * 3
out = softmax(x)
torch.testing.assert_close(out, torch_reference(x), atol=tol, rtol=tol)
```

本课初始容限：FP32为2e-6、FP16为1e-3、BF16为8e-3，绝不是所有输入分布的通用保证。若失败，先看误差大小、分布和实现，再决定是否调整容限；不能为了“绿灯”一味放宽。只检查行和也不够，因为错误的概率排列仍可能加起来等于1。

**本次实际验证**：Python 3.12.14下12项标准库测试通过、6项依赖测试跳过、6份Python源码语法检查通过、demo通过。没有安装依赖，没有执行PyTorch参考、Triton JIT、GPU数值测试或性能测量。语法检查不等于kernel编译成功，CPU数学模拟不等于GPU正确性证明。

## 7 性能测试要先声明测量边界

公共 `softmax(x)` 会扫描有限值并用 `.item()`取得检查结果。在GPU上这会增加kernel与同步开销。为了教学可诊断性，默认保留检查；不能把它的时延混进“纯kernel性能”再归因给规约算法。

配套 `benchmark.py`先验证输入、分配输出、执行正确性对比，再比较两个预分配输出的FP32路径：

```python
def custom():
    launch(x, a)

def baseline():
    torch.softmax(x, dim=-1, out=b)

custom(); baseline(); torch.cuda.synchronize()
torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)
ms = triton.testing.do_bench(custom, warmup=25, rep=100,
                           return_mode='median')
```

这里warmup与rep按毫秒预算，而不是次数；显式选择median，避免依赖默认返回策略。[do_bench API](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html)

已有GPU环境后运行：

```bash
python benchmark.py --rows 4096 --cols 1024
```

脚本输出设备、compute capability、torch/triton/CUDA版本、shape、dtype与测量边界。第一轮编译不算稳态延迟。理想有效带宽是 $B_{ideal}/t$，不是实际测得的DRAM吞吐。此处包含调用/launch流程，排除了校验与分配；不是完整Agent请求延迟，也不代表所有dtype性能。

至少比较多种行列规模，交错运行顺序并多次重复，关注设备温度/负载；不要只选一组最快数字。若预分配路径快，而公共接口慢，应该先分析扫描和同步开销，而不是声称kernel失败。

## 8 用本课练习基本实验观察

用三行记录，而不是写“感觉变快了”：

- 事实：输入shape/dtype、版本、校验范围、正确性误差、重复运行的时延。
- 假设：例如N跨过2次幂导致BLOCK翻倍，资源压力增大；也可能是launch或测量噪声。
- 验证：固定M与dtype，只改变N=1024/1025；重复运行并查看编译资源信息。这里是待验证假设，不是本次已测结论。

比较优化时，一次先改一个因素，例如num_warps；不要同时改dtype、输入规模和基准边界。保持同一输入有利于公平对照，但单一随机seed也不足以覆盖所有数值情况。

## 9 三个小改写练习

1. 给标准库参考增加平移不变性测试，使用温和数值与几个固定seed；再解释为什么极大偏移可能先损失输入有效数字。验收是数值接近，不要求逐bit相等。
2. 修改kernel支持输入的独立行stride，同时仍要求列stride=1。测试 `x=base[:, :N]` 的有间隙行视图。不要只删除连续性检查；必须修改地址计算并保留输出stride契约。
3. 设计语义mask接口，明确“全mask行”要报错还是输出全0。先写参考与测试，再修改kernel；不允许未经定义就把NaN替换成0。此练习是下一层设计，当前交付实现不支持该接口。

## 10 环境和下一步

无依赖路径只需Python 3.10+。GPU教学目标可用Linux NVIDIA RTX4060/sm_89；参考组合Python3.11、PyTorch2.5.1与Triton3.1.0列于依赖文件，未在本轮实测。CUDA驱动和PyTorch构建必须匹配，文档版本也不等于运行环境版本。Mac M4可以跑标准库和已有PyTorch的CPU参考，不能原生执行此CUDA路径。本课不安装任何依赖。

下一课计划回到服务线，讲“有界并发与背压”：超时不代表后台工作结束，并发槽位必须跟真实工作生命周期绑定。算子线后续再把Softmax与Attention的数据流连起来，不直接跳到完整FlashAttention。

### 来源与读取范围

以下官方页面于2026-11-04读取，核验API与关键语义；未运行其完整教程或复现性能图：

- [Triton Fused Softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)，main文档，无固定发布版本。本课不采用其persistent调度。
- [Triton load](https://triton-lang.org/main/python-api/generated/triton.language.load.html)，核对mask与other。
- [Triton sum](https://triton-lang.org/main/python-api/generated/triton.language.sum.html)，核对规约axis/dtype。
- [Triton do_bench](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html)，核对计时预算和return_mode。
- [PyTorch softmax](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.softmax.html)，此次stable跳转到2.15；核对dim、dtype、输出语义，不代表本课已在2.15运行。

完整代码、测试、示例、README、依赖声明和实际验证记录均在同一ZIP中。讲义包含核心代码，ZIP不是替代讲解。
