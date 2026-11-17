# 第8课 稳定 log-sum-exp 与分块在线 Softmax

日期：2026-11-06（北京时间）

**今天的目标：把“一整行一次算完”改成“每次只看一块，还能得到同样的归一化结果”。** 第6课已经实现减最大值的行 Softmax；昨天的有界并发属于服务层，今天回到模型数值计算层。预计主线45–60分钟，GPU实验另计。时间只是学习建议，不假设你已经完成前面的练习。

先理解一个核心关系：**归一化分母会随最大值改变而换尺度，旧的累计结果也必须一起换尺度。** 配套代码先给标准库参考与可运行测试，再给可选 PyTorch / Triton 路径。没有 GPU 也可以完成主线。

## 1 先分清三个输出

给定非空有限实数行 $x_0,\ldots,x_{N-1}$：

$$
m=\max_j x_j,\qquad \ell=\sum_j\exp(x_j-m).
$$

我们关心三个不同的东西：

1. **归一化状态** $(m,\ell)$。它是可以组合的中间表示。
2. **log-sum-exp**，简称 LSE：$L=m+\log\ell$。它是一个标量，不是概率向量。
3. **Softmax**：$p_j=\exp(x_j-m)/\ell$。它是与输入等长的概率向量。

由于最大元素对应的指数为1，其余指数在0到1之间，实数数学下 $1\le\ell\le N$。我们避免直接生成可能溢出的 $\exp(m)$，但最终 LSE 仍要承受返回 dtype 的舍入。

模型里它们各有用途：概率向量用于归一化权重；LSE用于对数概率、损失和某些反向实现的辅助信息；$(m,\ell)$是实现过程中保留的数值状态。PyTorch 的 `torch.logsumexp(x, dim=-1)` 沿最后一维规约，并使用稳定计算；`keepdim=True`会保留长度为1的规约维度。[PyTorch logsumexp API](https://docs.pytorch.org/docs/2.15/generated/torch.logsumexp.html)

对普通数值，数学上 $\log p_j=x_j-L$。但代码应注意运算顺序：更稳妥的形式是 $(x_j-m)-\log\ell$。如果先将很大的 $m$ 和小的 $\log\ell$ 相加，再相减，小量可能已经丢失。后面会用真实可运行反例说明。

## 2 为什么每块各做 Softmax 再拼起来是错的

设一行拆为 A 与 B 两块。分别归一化后，每块的概率和都是1，拼起来的总和就是2。它们没有共同的分母。

更隐蔽的错误是：每块先减各自的最大值，再把各块指数和直接相加。A 的指数和以 $e^{m_A}$ 为尺度，B 以 $e^{m_B}$ 为尺度，不能像同一种单位的数量那样直接相加。

手算一个短例子：

- A为 `[1,2]`，$m_A=2$，$\ell_A=1+e^{-1}\approx1.367879$。
- B为 `[3]`，$m_B=3$，$\ell_B=1$。
- 全行最大值为3。A原有的指数总和要乘 $e^{2-3}=e^{-1}$。
- 全行 $\ell=1.367879\times e^{-1}+1\approx1.503215$。
- 最终概率约为 `[0.090031,0.244728,0.665241]`。

如果写成 $\ell_A+\ell_B$，第二步就已经不再表示任何正确的全行分母。一个特别有效的测试是让**最大值在最后一块出现**，这样旧状态一定要经历重新缩放。

## 3 推导可合并状态

用 $S_A=(m_A,\ell_A)$ 表示A块；用 $S_B=(m_B,\ell_B)$ 表示B块。合并后的最大值：

$$m=\max(m_A,m_B).$$

把A块的每个指数改写到新尺度：

$$\sum_{j\in A}e^{x_j-m}=e^{m_A-m}\sum_{j\in A}e^{x_j-m_A}=e^{m_A-m}\ell_A.$$

因此合并规则为：

$$
S_A\oplus S_B=
\left(\max(m_A,m_B),\;\ell_Ae^{m_A-m}+\ell_Be^{m_B-m}\right).
$$

这是 online normalizer 的关键。原始论文给出了逐元素更新及可并行组合的二元操作；本课用块摘要来实现它，不照搬论文的硬件性能数字。[Online normalizer calculation for softmax，§3与§3.1](https://arxiv.org/html/1805.02867v2)

可以把单元素 $x$ 看成状态 $(x,1)$，于是同一公式既支持逐个元素，也支持按块更新。无论之前看过多少块，只需保留当前 $(m,\ell)$；当前块处理完就可以丢弃其临时指数值。

### 结合律与浮点误差

在精确实数运算下，先合并A/B再合并C，或先合并B/C再合并A，结果都表示同一个集合的最大值和指数和。因此可以顺序折叠，也可以树形合并。

在浮点下，加法、指数与乘法都有舍入。不同块大小、规约树和执行顺序可能产生小误差，不能承诺逐bit相同。测试使用误差容限，同时检查概率非负、概率和以及最大值位置，而不是只比一个“看起来接近”的LSE标量。

这是一个通用架构思路：先定义“状态到底代表什么”，再允许更换调度方式。类似的分层也出现在前几课的后端接口与执行器中，只是这里的不变量是数学量，不是队列容量。

### 空状态不等于可以随便计算负无穷

把空状态约定为 $(-\infty,0)$ 很方便，但直接将两个空状态代入公式会出现 $-\infty-(-\infty)$，产生NaN。标准库实现明确识别空状态，合并空状态时直接返回另一侧。

非空块全部为有限数，所以其最大值有限。GPU实现按真实列范围循环，每次迭代至少含一个有效元素，最后一块只有部分槽位被填成负无穷，不会把一个全空块送进规约。

本课**不支持用户输入NaN、正负无穷，也不实现语义mask**。Attention的全mask行不是这里的“空batch”；它必须单独规定输出和状态，不能顺手套用本课逻辑。

## 4 输入契约与代码分层

| 路径 | 约束与返回值 |
|---|---|
| 标准库输入 | 可迭代的有限 `numbers.Real`，能转为Python float；拒绝bool、复数、NaN、±inf，不是任意精度算术 |
| 标准库空集合 | `block_state([])`与空块序列返回单位元；空LSE为负无穷；空Softmax拒绝 |
| 标准库块参数 | 正整数，默认128；输入为迭代器时由 `two_pass_softmax`保存副本以便第二遍读取 |
| GPU shape | 连续row-major二维 `[M,N]`，M≥0，1≤N≤65536；0行返回空输出，0列拒绝 |
| GPU dtype | FP16、BF16、FP32，硬件/运行时须支持所选dtype；中间FP32，概率写回原dtype |
| GPU数值 | 全部有限，已表示输入满足绝对值≤1e30；此上限避免FP32减法溢出，不保证小差异保真 |
| GPU范围 | NVIDIA CUDA；不接收 `requires_grad=True`，无反向；不支持ROCm/MPS/CPU执行这个kernel |
| GPU块参数 | 128、256（默认）、512或1024；这是教学选择，并非Triton的硬限制 |
| GPU输出 | 新分配、与输入同shape/dtype的概率，不覆盖输入；本课GPU不输出LSE |

GPU的有限值与幅度校验会扫描输入，并通过 `.item()`让主机等待设备结果。它是公共接口的实际成本，不应在端到端比较中偷偷忽略。内存布局不符时报错，不自动调用 `.contiguous()`引入隐藏复制。65536是教学范围，不是性能或最大宽度承诺。

标准库代码用于数学和边界测试，不模拟GPU线程、FP32规约顺序或快速指数。可选PyTorch路径提供CPU数值参考和GPU基准对照；自定义张量实现位于Triton模块，其shape、dtype、layout和forward约束见上表。把错误挡在公共接口之外，能避免一个“本来不支持”的输入落进kernel后被误认为算法错误。

包内代码按职责拆分：

- `online_softmax.py`：有限值检查、不可变State、块摘要、状态合并、LSE与两遍Softmax。
- `triton_online_softmax.py`：可选GPU kernel及受检查公共包装；启动时切到输入所属CUDA device，退出后恢复调用方设备。
- `example.py`：可直接运行的手算示例和大共同偏移案例。
- `tests/test_reference.py`：标准库单元与固定种子属性检查。
- `tests/test_optional.py`：PyTorch float64参考、GPU shape/dtype/块大小/错误输入测试。
- `audit_checks.py`：独立高精度与边界验证，具体检查见验证记录。
- `benchmark.py`：两种清楚分离的GPU计时范围。
- `requirements-optional.txt`：可选依赖范围提示，不是已验证lockfile。
- `README.md`、`VALIDATION.md`：运行入口、输入契约、已测与未测边界。

没有自动安装逻辑。缺依赖会报告跳过或说明原因；不会因为GPU不可用，就把CPU执行冒称GPU验证。

## 5 从可执行参考开始

下面的实现直接来自包内源码，完整契约、其他函数和测试都在ZIP中。

```python
"""Dependency-free finite-real, forward-only reference (Python binary64)."""
from dataclasses import dataclass
import math
from numbers import Real


def finite_values(values):
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError('inputs must be real numbers, not booleans')
        x = float(value)
        if not math.isfinite(x):
            raise ValueError('inputs must be finite')
        result.append(x)
    return result


@dataclass(frozen=True)
class State:
    """m=max(x), l=sum(exp(x-m)); (-inf, 0) is the empty identity."""
    m: float = -math.inf
    l: float = 0.0

    def __post_init__(self):
        if not ((self.m == -math.inf and self.l == 0.0) or
                (math.isfinite(self.m) and math.isfinite(self.l) and self.l >= 1.0)):
            raise ValueError('invalid normalizer state')

    @property
    def logsumexp(self):
        return -math.inf if self.l == 0 else self.m + math.log(self.l)


def block_state(values):
    xs = finite_values(values)
    if not xs:
        return State()
    m = max(xs)
    return State(m, math.fsum(math.exp(x - m) for x in xs))


def merge(a, b):
    # Branch before subtracting maxima: (-inf)-(-inf) would be NaN.
    if a.l == 0:
        return b
    if b.l == 0:
        return a
    m = max(a.m, b.m)
    return State(m, a.l * math.exp(a.m - m) + b.l * math.exp(b.m - m))


def online_normalizer(blocks):
    """Consumes blocks once; retains O(1) state (plus current block)."""
    state = State()
    for block in blocks:
        state = merge(state, block_state(block))
    return state


def stable_logsumexp(values):
    """Empty input returns -inf, the logarithm of an empty exponential sum."""
    return block_state(values).logsumexp


def two_pass_softmax(values, block_size=128):
    """Materializes a reusable input, normalizes blocks, then emits probabilities.

    This is not a one-pass streaming output algorithm: earlier probabilities
    depend on later inputs. Empty softmax is undefined and rejected.
    """
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
        raise ValueError('block_size must be a positive integer')
    xs = finite_values(values)
    if not xs:
        raise ValueError('softmax needs at least one value')
    state = online_normalizer(xs[i:i + block_size] for i in range(0, len(xs), block_size))
    # Do not reconstruct exp(x-lse): m+log(l) can lose log(l) for large offsets.
    return [math.exp(x - state.m) / state.l for x in xs]
```

阅读顺序：先看状态包含哪些量，再看块摘要，最后看合并。`block_size`控制的是每次处理多少列，不改变最终归一化的目标。它不是批大小，也不是CUDA线程数。

参考实现为了输入校验、可重复读取和教学清晰，可能保存规范化的Python输入副本。不要把“归约状态只需常数空间”误解成“整个函数不占输入或输出空间”。最终Softmax向量本身就需要 $O(N)$ 输出空间。

## 6 Online 不是每读一个分数就能永久输出一个概率

第一遍扫完之前，未来仍可能出现更大的分数，也可能增加总概率质量。已经看到的第一个元素，其最终概率仍未知。

本课完整Softmax采用两阶段：

1. **第一遍**按块读取输入，维护最终 $(m,\ell)$。
2. **第二遍**重新读取输入，用统一的 $m,\ell$ 生成所有概率。

这里只求LSE时，第一遍结束即可，不需要概率向量。若输入流不可回放，又不能存下原数据，就不能凭一个常数大小的状态，最后恢复每一个元素对应的概率。它保存的是归一化摘要，不是原始序列的可逆编码。

这也说明为什么算法论文中的访问次数模型不能直接套到任何实现：第6课的短行kernel把一行逻辑向量留在program内部，已经可以单次主输入加载后输出；今天为了限制每次活跃的块大小，会重新读输入。分块是资源权衡，不是名字里有online就自动更快。

## 7 Triton里如何把状态跨块带下去

```python
"""Optional educational CUDA/Triton implementation; importing CPU code needs none of this."""
import torch
import triton
import triton.language as tl

MAX_WIDTH = 65536
MAX_ABS = 1.0e30  # Prevent FP32 subtraction overflow; not an accuracy guarantee.


@triton.jit
def _kernel(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)  # Avoid 32-bit row-offset overflow.
    lanes = tl.arange(0, BLOCK)
    m = tl.full((), -float('inf'), tl.float32)
    ell = tl.full((), 0.0, tl.float32)
    # N>0, so each visited block has at least one valid element.
    for start in range(0, N, BLOCK):
        offsets = start + lanes
        x = tl.load(X + row * N + offsets, offsets < N, other=-float('inf')).to(tl.float32)
        bm = tl.max(x, 0)
        new_m = tl.maximum(m, bm)
        # First block: exp(-inf-finite)=0; never evaluates -inf-(-inf).
        ell = ell * tl.exp(m - new_m) + tl.sum(tl.exp(x - new_m), 0)
        m = new_m
    for start in range(0, N, BLOCK):
        offsets = start + lanes
        x = tl.load(X + row * N + offsets, offsets < N, other=-float('inf')).to(tl.float32)
        p = tl.exp(x - m) / ell
        tl.store(Y + row * N + offsets, p, offsets < N)


def validate(x, block_size=256):
    if not isinstance(x, torch.Tensor):
        raise TypeError('x must be a torch.Tensor')
    if x.ndim != 2 or not (1 <= x.shape[1] <= MAX_WIDTH):
        raise ValueError('expected [M,N], M>=0 and 1<=N<=65536')
    if x.device.type != 'cuda' or torch.version.hip is not None:
        raise ValueError('requires NVIDIA CUDA tensor (ROCm is outside this lesson)')
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError('supported dtypes: float16, bfloat16, float32')
    if not x.is_contiguous():
        raise ValueError('contiguous input required; no implicit copy')
    if x.requires_grad:
        raise ValueError('forward only; requires_grad must be False')
    if type(block_size) is not int or block_size not in (128, 256, 512, 1024):
        raise ValueError('block_size must be 128, 256, 512 or 1024')
    if not bool(torch.isfinite(x).all().item()):
        raise ValueError('finite inputs required')
    if x.numel() and float(x.abs().max().item()) > MAX_ABS:
        raise ValueError('absolute input value must be <=1e30')


def _launch_validated(x, out, block_size):
    """Private benchmark path: caller already validated input/output shape and storage."""
    if x.shape[0]:
        with torch.cuda.device(x.device):
            _kernel[(x.shape[0],)](x, out, x.shape[1], block_size, num_warps=4)
    return out


def softmax(x, block_size=256):
    """Checked public API; validation synchronizes the GPU and scans the input."""
    validate(x, block_size)
    out = torch.empty_like(x)
    return _launch_validated(x, out, block_size)
```

一行仍由一个program负责；行之间独立。第一段循环的 `m` 和 `l` 是循环携带的状态，下一块依赖上一块更新后的值。块内最大值与求和则由 `tl.max`、`tl.sum`完成。

请注意五件事：

- 列偏移是 `start + tl.arange(0, BLOCK)`，有效条件是 `col < N`。尾块用 `-inf`读入填充，使它不贡献最大值或指数和；写回也要带mask。
- 每次取到新最大值，都要修正旧分母。改变最大值却忘了修改分母，会让结果依赖分块边界。
- 中间值用FP32计算，写回概率时转为输入dtype。本课GPU接口只返回概率，不返回LSE；LSE由标准库状态和参考函数展示。低精度输入已经丢失的信息不能靠FP32中间值找回来。
- `BLOCK`是逻辑块宽度，`num_warps`是执行配置，两者不等价。没有证据时不要画成“一个元素对应一个线程”。
- 这份代码在一个program里串行遍历块，不是多program共同处理一行的并行归约，没有实现跨program同步。

本课kernel使用JIT函数中的普通 `range`写循环，由Triton编译；需要传入额外编译属性时还有 `tl.range`接口。本课不启用warp specialization，也不根据文档中的新属性假设旧版本都支持。[Triton range API](https://triton-lang.org/main/python-api/generated/triton.language.range.html)

从Python语法可解析，到Triton JIT编译，再到GPU数值正确，是三层不同的验证。当前环境没有PyTorch/Triton，下面的GPU路径已提供测试入口，但**本轮未JIT、未在GPU运行、未测GPU速度**。

## 8 一个比指数溢出更隐蔽的反例

看两项相等的大数 `[1e20,1e20]`。Python浮点中，$\log2$远小于这个数量级的间隔，于是计算 `1e20 + log(2)`会舍入回 `1e20`。

如果先算LSE，再做：

```python
[math.exp(v - lse) for v in row]
```

两个值都可能成为1，概率和为2。没有NaN，也没有overflow，但结果完全错误。

保留状态再做：

```python
[math.exp(v - m) / l for v in row]
```

可以得到 `[0.5,0.5]`。这就是本课同时保留 $m$ 和 $\ell$，而不只保留LSE的理由。配套测试真的覆盖该反例。

对FP32也有类似现象，且出现于更小的共同偏移。LSE输出“数值看起来很准”不能证明由它重建的概率也准；用相对误差比较极大LSE时尤其容易掩盖这个问题。

再区分另外一种损失：如果创建FP32输入时，`1e20`和`1e20+1`已经被表示成同一个数，那么之后任何kernel都无法恢复原来那1的差别。应比较**相同已量化输入**上的参考与实现，不能让参考偷偷拥有额外精度的信息。

## 9 测试应该刻意破坏哪些假设

不要只测随机正态输入。随机输入经常“很正常”，反而绕过最容易写错的路径。

在解压后的目录运行：

```bash
python example.py
python -m unittest discover -s tests -v
python audit_checks.py
```

原始测试包含7个具名方法，本轮4个标准库方法通过、3个可选方法跳过。标准库随机方法实际生成132组输入（11种长度×12轮），每组检查4种块大小、随机分块、空块、块顺序、平移、概率范围/总和和近似结合律。最大长度1025，包含127/128/129、255/256/257等边界。

可选GPU代码另包含M=0、N=1、N=65536、4种块宽和3种dtype（硬件支持时）、大共同偏移、输入拒绝和多GPU设备上下文案例。这些测试本轮没有运行成功的证据，不能把“写了测试”计为“通过测试”。不支持BF16或只有一张GPU时，对应子案例不执行。

测试的判断关系包括：

- 分块结果接近整行参考；不同块大小结果在容限内一致。
- 合并状态的两种顺序、左折叠和树形组合，在容限内一致。
- 等值行应均匀，单元素行概率为1；最大值晚到时需要重缩放。
- 合理平移下Softmax不变、LSE加上相同平移；极端平移会先改变输入表示，所以不宣称无限精度的平移不变性。
- 尾块不能额外贡献概率；负分数的离线填充模拟会抓到错误零填充。该检查是数学模拟，不是GPU执行。
- 非法shape、dtype、layout、grad、非有限值或块参数必须在公共接口拒绝。

随机属性检查能扩大覆盖面，不是形式证明。标准库全部通过也不能替代真实设备上的dtype、编译器、内存访问与数值测试。

## 10 如何分析性能而不被漂亮数字误导

对 $M$ 行、每行 $N$ 列，输入/概率dtype占 $s$ 字节，忽略缓存、对齐和额外校验：

- 本课完整两遍Softmax，主数据逻辑流量约 $3MNs$：两次读X、一次写P。GPU接口不额外写LSE。
- 若只实现LSE，主数据理论上可以只读一遍X，再写每行标量；本课没有独立GPU LSE-only入口，不要把概率输出函数当成该优化。
- 第6课短行融合版本理想主流量约 $2MNs$，因此不能根据online论文的三遍到两遍结论，就断言本课一定快于第6课。

以上是**逻辑访问模型**，不是实测DRAM字节数。第二遍可能命中缓存；寄存器溢出也可能增加隐含流量。公共接口的有限值扫描、`.item()`同步、输出分配和调用开销也会增加端到端时间。

分块的候选收益是较小的活跃向量规模：每个program的主要临时向量由整行宽度变为BLOCK，能减少某些资源压力。成本包括重新读取、循环依赖、重复指数和块内规约；行级并行度仍受M限制。M很小时，即使N很大，也可能没有足够program把设备占满。

已有兼容GPU环境时，可以运行：

```bash
python benchmark.py --rows 256 --cols 4097 --warmup 20 --repeats 100
```

本课不要求你现在安装依赖，也没有自动安装。基准先用同一已表示输入的FP64 PyTorch输出检查正确性，再预热，最后报告多轮中位数，分两组：

1. **unchecked/preallocated**：Triton与PyTorch都复用已分配输出，都排除输入校验；以CUDA events计时。事件间隔仍可能含主机launch供给不足造成的空闲，不是纯指令延迟。
2. **checked/allocated**：两边都使用同一输入校验，分配输出，并在主机秒表前后同步。它衡量所选包装调用范围，不是一般PyTorch API必然承担的检查成本。

只在组内比较。基准使用固定BLOCK=256和重复驻留输入，可能有热缓存；没有模拟多请求、冷缓存、真实Attention或服务吞吐。本轮运行该命令明确报告缺少torch/triton并跳过，**没有任何GPU耗时或加速比**。

想做可信比较，先固定同一个问题：都只输出Softmax，还是还要输出LSE？本课基准的两边都只输出Softmax。是否都包含输入校验和分配？两边输入的dtype、shape与layout是否相同？GPU异步launch不能直接用未同步的CPU秒表当作kernel耗时。

建议实验一次只变一个因素：固定M/N，改变BLOCK；或固定BLOCK/M，改变N。先写假设，再看正确性、资源和时间证据。没有实测，不写“提升了多少倍”。

## 11 从Softmax接到Attention的下一块拼图

Attention通常不需要把整个概率矩阵作为最终结果返回，它需要的是概率加权后的V：

$$o=\sum_j p_jv_j.$$

这时可以把状态扩展为 $(m,\ell,a)$，其中：

$$a=\sum_j\exp(x_j-m)v_j.$$

两个块合并时，分子向量也按同样尺度修正：

$$a=a_Ae^{m_A-m}+a_Be^{m_B-m},\qquad o=a/\ell.$$

如果每个分数块和对应V块都在处理时可用，就不必为了最终加权结果而保存所有概率。这里的a大小取决于V的特征维度，不再是一个标量。真正Attention还要产生QK分数、处理缩放/语义mask、设计Q/K/V分块、精度与反向；本课没有完成这些工作。

官方Triton Fused Attention示例使用了对最大值、分母和输出累加器的同步缩放，并采用 `exp2` 路径与相应尺度转换。它保存的辅助量要按源码里的对数底解释，不能直接当成本课自然对数LSE。[Triton Fused Attention教程](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)

**online normalizer是一块数学基础，不等于完整FlashAttention实现。** 本课的独立Softmax仍输出整张概率张量，不能把它的显存行为说成避免物化Attention概率矩阵。

## 12 三个小改写练习

### 练习A 找出少掉的缩放因子

把合并中的旧分母缩放故意删除。先预测 `[1,2]`与`[3]`、`[3]`与`[1,2]`两种块顺序的结果，再运行测试。写出为什么“最大值晚到”是重要反例。

验收：你能指出哪一行破坏了状态不变量，而不仅是看到一个断言失败。练习改动放在副本里，不要覆盖已经验证的原文件。

### 练习B 实现树形合并

先把输入拆为块摘要，再每次两两合并直到剩一个。处理奇数个块和零个块，比较树形与左折叠结果。

验收：同一已表示输入上的结果在容限内一致；能够解释为何数学结合律不等于浮点bitwise结合律。构建全部摘要需要额外空间，不要宣称整个实验仍为常数额外空间。

### 练习C 只输出加权结果

用标准库实现一维value版本：每个分数 $x_j$ 配一个有限的标量 $v_j$，维护 $(m,\ell,a)$，最后返回 $a/\ell$。与“先完整Softmax，再点乘V”的参考比较。

验收：最大值在末块、value正负相抵、等值分数、长度不匹配都要测试；分母和分子必须同时缩放。先保持小规模有限数值，不借此声称解决了所有Attention精度或全mask问题。

今天只需把这句话讲清：**状态保存的是“以当前最大值为共同尺度的指数和”，最大值变了，旧累计量的单位也要一起换。** 练习是否完成仍以你的实际反馈为准。

## 13 本轮验证与参考资料

本轮在Linux x86_64、CPython 3.12.14完成标准库测试、示例、独立审计与源码语法检查。PyTorch与Triton未安装，可选测试及GPU基准跳过；没有JIT编译、GPU数值运行、性能数据、训练或部署。详细的实际命令、通过/跳过数量和独立审计结果见同包 `VALIDATION.md` 与 `TEST_OUTPUT.txt`。

`compileall`检查Python语法，不会完成Triton编译。没有验证Mac/MPS、ROCm、其他Python版本、反向传播或生产负载。可选依赖文件是版本范围提示，不代表这些组合已验证。

资料核读日期：2026-11-06。论文采用固定v2；Triton使用main滚动文档；PyTorch stable链接本轮跳转到2.15。文档版本与可选依赖参考组合并不等于本轮执行环境，兼容性必须由实际环境测试确认。

- [Online normalizer论文v2](https://arxiv.org/html/1805.02867v2)：读取§3、§3.1的状态更新/合并与访问模型；不复用其历史硬件速度作为本课成绩。
- [PyTorch 2.15 logsumexp](https://docs.pytorch.org/docs/2.15/generated/torch.logsumexp.html)：核对dim、keepdim和稳定计算语义。
- [Triton range](https://triton-lang.org/main/python-api/generated/triton.language.range.html)与[exp](https://triton-lang.org/main/python-api/generated/triton.language.exp.html)：核对JIT循环和指数API。
- [Triton Fused Attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)：仅用于后续连接，重点读分母/累加器重缩放及exp2尺度，未运行完整官方示例。

**下一课计划：用小规模CPU参考把QKᵀ、mask、Softmax与V连起来，再对照可流式合并的Attention数据流。** 后续继续穿插后端指标、重试预算、记忆系统与Agent架构，不假设某一条线已经学完。
