# 第3课 RMSNorm：把行规约变成模型组件

日期：2026年11月1日，星期日（Asia/Shanghai）。主线预计40–60分钟，数值与性能部分可选加练15分钟，均为估计。

前两课分别讲残差加法与行求和。今天只解决一个问题：**如何把一行隐藏状态的整体尺度归一化，并乘上可学习的逐通道权重？** 继续保留“参考语义→契约→优化实现→测试”这条主线，不因为前两课已发送就假定你已掌握。

交付：本Markdown讲义和 `Coding_架构每日学习_2026-11-01.zip`。ZIP含全部实现、测试、基准、README与版本/验证记录。核心代码直接展示在正文。

## 1 需求、位置与前置知识

学习目标：
1. 从公式推导平方、行规约、倒平方根与逐元素缩放的数据流。
2. 说清一个program负责什么，以及mask为什么不能替代正确的除数。
3. 理解 `_validate` 为什么存在，哪些条件检查了、哪些只是调用者前置条件。
4. 用独立参考与边界测试判断正确性，区分可运行教学实现和已验证生产算子。

前置：Python函数/列表、Tensor shape、逐元素乘法、上一课 `program_id/arange/load/sum/store`。不要求先会手写CUDA同步。

模型隐藏状态常写成 `[B,T,H]`，其中B是批次、T是token位置、H是每个位置的通道数。把前两维展成M=B×T、N=H，则本课输入为 `[M,N]`，每行独立归一化，输出仍是 `[M,N]`。展平只是在允许的连续布局下重解释；真实系统若需复制，应单独计入成本。

RMSNorm可放在Transformer子层前后，位置由模型结构决定。例如前置归一化的示意为 `h + Attention(RMSNorm(h))`，FFN也可有独立归一化。**不是所有Transformer都采用这个结构，也不能把已有LayerNorm无条件替换掉。** 本课只实现一个前向组件，不实现Attention、完整模型、训练或推理调度。`weight[N]`是网络的可学习参数；本课拿既定值做推理，没有开展学习。

## 2 从公式看数据流

对第r行：

$$
s_r=\frac{1}{N}\sum_{c=0}^{N-1}x_{r,c}^{2},\qquad
q_r=\frac{1}{\sqrt{s_r+\epsilon}},\qquad
y_{r,c}=(x_{r,c}q_r)\gamma_c.
$$

$\gamma$就是代码里的 `weight`，shape为 `[N]`，所有行共享；$\epsilon$本课显式取 `1e-5`，放在平方根内。不同模型需遵从其配置，不能因名字相同就使用相同epsilon。PyTorch2.5.1的 `eps=None` 对FP32采用机器精度约 $1.1920929\times10^{-7}$，并不是本课 `1e-5`。[固定版本实现](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/layer_norm.cpp#L248-L286)

例子：x=`[3,4]`，gamma=`[1,2]`。
- 平方得到 `[9,16]`，均方为12.5。
- 每个位置共享 `1/sqrt(12.5+eps)`。
- 缩放后约 `[0.848528,1.131370]`，再乘gamma得到 `[0.848528,2.262741]`。

与LayerNorm的关键差异：本课不减行均值，不计算中心化方差，也没有额外bias。均值不必变成0。归一化后又乘gamma，输出的均方也不必等于1。epsilon非零时，即使gamma全1，“输入乘任意正比例后输出严格不变”也不成立。原论文的重缩放讨论要连同假设阅读，而不是直接当作实现断言。[原论文第4节](https://arxiv.org/html/1910.07467v1#S4)

## 3 先写清接口契约

| 项目 | 本课约定 |
|---|---|
| 输入/输出 | x `[M,N]`，weight `[N]`，输出 `[M,N]`，不修改输入 |
| dtype | 全部FP32，平方和规约同为FP32 |
| layout | 普通dense strided且contiguous；不隐式做复制 |
| 设备 | x和weight相同；PyTorch参考可CPU，Triton必须CUDA |
| 大小 | `1 <= N <= 4096`，`0 <= M <= 2^31-1`，是教学上限不是通用硬件极限 |
| 空输入 | M=0返回空Tensor，不发射本课kernel；N=0拒绝，因为均方没有定义 |
| 自动微分 | x或weight的requires_grad为True就拒绝；没有backward |
| epsilon | 实数且非bool，范围 `2**-126 <= eps <= 1`，先舍入为FP32；默认1e-5 |
| 数值前置 | 输入/权重有限，平方、平方和、加eps及最终乘法不溢出；热路径不扫描 |

epsilon限制是教学设计：仅“Python浮点数大于0”不够，像 `1e-100` 转FP32会变0；最小正常FP32下界还避免本课展开次正规数的flush-to-zero差异。这里不是建议所有模型必须采用这个范围。允许更广数值范围时，应重新定义行为和测试。

### `_validate(x, weight, *, cuda, eps)` 到底在做什么？

它是函数入口的检查员，不参与RMSNorm数学计算。
- `x, weight` 是待处理数据；`cuda`说明当前实现是否必须用CUDA；`eps`是数值配置。
- 参数里的 `*` 表示后面的参数必须写名字，例如 `_validate(x,w,cuda=True,eps=1e-5)`，防止一串位置参数弄混。
- 检查失败就 `raise`，及时终止，不继续启动可能读错地址的kernel。
- 例如同shape仍可能一份在CPU一份在GPU，因此shape检查不能代替device检查。
- `requires_grad`检查不是“帮你关闭梯度”，而是拒绝一个本课没有实现反向传播的请求。真正使用 `nn.Parameter` 推理时，需在明确推理接口处提供不要求梯度的权重视图；不要误以为本包装器是可训练的 `nn.Module`。
- 检查dtype、shape等只读元数据；检查每个元素是否有限则要额外数据访问，GPU `.item()`还可能同步主机，所以本课只把它写成调用者数值前置条件。

**架构习惯：入口拒绝不支持的输入，比内部偷偷转换后把额外成本藏起来更容易维护。** 若产品必须接受任意布局，最简单替代方案是显式转换/使用PyTorch，并把复制与fallback策略放在更高层；无需一开始堆复杂调度器。

## 4 参考实现：先知道答案应该怎么算

最小PyTorch数学表达式（完整文件中的参考还会调用契约检查）：

```python
mean_square = (x * x).mean(dim=1, keepdim=True)  # [M,1]
inv_rms = torch.rsqrt(mean_square + eps)         # [M,1]
y = (x * inv_rms) * weight                      # [M,N]
```

`keepdim=True`保留长度1的列维，让行标量广播回各通道；weight按最后一维广播。本课完整参考只允许同设备FP32二维输入，不把PyTorch支持的全部广播行为暴露成自己的契约。

没有PyTorch的Mac/普通CPU也可运行ZIP中的 `reference_rows`：用 `math.fsum` 计算Python双精度平方和。它是数学对照，不是GPU实际规约顺序或 `rsqrt` 指令的模拟。GPU验收还需要真正运行Tensor版本，不能拿标准库PASS当作kernel已验证。

## 5 Triton核心实现与逐段解释

文件 `rmsnorm_kernel.py`：

```python
"""Import only on GPU path; tl is module-global for Triton JIT resolution."""
import triton
import triton.language as tl


@triton.jit
def kernel(X, W, Y, N: tl.constexpr, EPS, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    mask = col < N
    offset = row * N + col
    x = tl.load(X + offset, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + col, mask=mask, other=0.0).to(tl.float32)
    mean_square = tl.sum(x * x, axis=0) / N
    inv_rms = tl.rsqrt(mean_square + EPS)
    y = (x * inv_rms) * w
    tl.store(Y + offset, y, mask=mask)
```

### 5.1 谁负责哪一行？

grid为 `(M,)`，`program_id(0)`得到行号；不是每个program只处理一个元素，也不是一个program等同一条CUDA线程。`num_warps=4`给出实现配置，逻辑BLOCK个元素怎样分布到线程/寄存器由编译器决定。行号先转int64，再乘N，避免大矩阵行偏移先在int32里溢出。

### 5.2 从哪里搬数据？

连续二维输入的线性地址是 `row*N+col`。weight是所有行共享的通道向量，所以地址只有 `col`，不能加 `row*N`。每个program读取自己一行x与weight，并只写自己一行y；行与行没有共享可写结果，因此不需要跨program同步或atomic。

Triton内部规约需要的线程通信由编译器生成。没有手写barrier不等于硬件“没有通信”，也不表示以后实现多program协作时可以忽略同步。

### 5.3 尾块为什么补0，为什么除N？

BLOCK取不小于N的2次幂。N=33时BLOCK=64，col33..63不读取真实地址，由 `other=0` 填补。平方后的0不改变平方和，但真正元素只有33个，所以要除33，不是64。除BLOCK会把均方压低，使输出幅度偏大。

加载weight也加mask，存y也加mask；只对load加mask、忘记store，是常见越界错误。N=0在入口已拒绝，避免next_power_of_2(0)和除0。

### 5.4 为什么先转FP32？

平方前就确定FP32计算，规约也是FP32。本课输入已经FP32，显式转换仍强调数值位置；以后扩展FP16/BF16时，不能先在低精度里平方溢出后再转FP32。

`tl.rsqrt`返回倒平方根，不是任意精度数学函数；加法顺序、近似指令和舍入可能使结果与另一实现略有不同。测试用容差，不要求逐位相等。[Triton3.1 sum](https://github.com/triton-lang/triton/blob/v3.1.0/python/triton/language/standard.py#L243-L248)、[rsqrt](https://github.com/triton-lang/triton/blob/v3.1.0/python/triton/language/math.py#L155-L160)

### 5.5 epsilon能解决什么，不能解决什么？

全零行加正epsilon后分母不为0，输出仍为0。**epsilon不防止平方溢出**：有限FP32的1e20平方也可能为Inf；许多单项平方不溢出，平方和仍可能溢出。最终乘巨大weight也可能溢出。PyTorch官方同样提醒中间结果溢出问题。[数值精度说明](https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/notes/numerical_accuracy.rst#L36-L47)

本课不声称处理这些极端值；要支持更大范围，可研究先按行最大绝对值缩放、分阶段累加或更高精度参考，但会改变运算、成本及epsilon处理，需独立测试。不能简单把eps调大来“修好”溢出。

## 6 完整公共入口与CPU参考

下列就是ZIP中 `rmsnorm.py` 的完整代码，支持只用标准库运行示例。Triton模块只在GPU路径导入；`tl`保持在kernel文件模块级，避免JIT无法解析局部导入。

```python
"""Lesson 03: FP32 inference RMSNorm. CPU reference needs only Python."""
import math
import numbers
import struct

MAX_COLS = 4096


def validate_eps(eps):
    if isinstance(eps, bool) or not isinstance(eps, numbers.Real):
        raise TypeError('eps must be a real scalar, not bool')
    eps = float(eps)
    # Require a positive NORMAL FP32 epsilon: avoids flush-to-zero surprises.
    if not math.isfinite(eps) or not (2**-126 <= eps <= 1.0):
        raise ValueError('teaching contract: 2**-126 <= eps <= 1')
    return struct.unpack('f', struct.pack('f', eps))[0]


def reference_rows(rows, weight, eps=1e-5):
    """Python double reference; not a FP32 GPU simulation.

    Numerical domain: finite inputs and no intermediate/output overflow.
    Python double arithmetic is a diagnostic, not support for every finite double."""
    eps = validate_eps(eps)
    n = len(weight)
    if not 1 <= n <= MAX_COLS:
        raise ValueError('weight width must be 1..4096')
    if any(len(row) != n for row in rows):
        raise ValueError('each row must match weight width')
    if any(not math.isfinite(v) for v in weight) or any(
            not math.isfinite(v) for row in rows for v in row):
        raise ValueError('reference requires finite inputs')
    result = []
    for row in rows:
        inv = 1.0 / math.sqrt(math.fsum(v*v for v in row) / n + eps)
        result.append([(v * inv) * w for v, w in zip(row, weight)])
    return result


def _validate(x, weight, *, cuda, eps):
    import torch
    eps = validate_eps(eps)
    if not isinstance(x, torch.Tensor) or not isinstance(weight, torch.Tensor):
        raise TypeError('x and weight must be tensors')
    if x.layout != torch.strided or weight.layout != torch.strided:
        raise ValueError('dense strided tensors required')
    if x.ndim != 2 or weight.ndim != 1 or x.shape[1] != weight.shape[0]:
        raise ValueError('expected x[M,N], weight[N]')
    if x.dtype != torch.float32 or weight.dtype != torch.float32:
        raise TypeError('FP32 only')
    if x.device != weight.device:
        raise ValueError('devices must match')
    if any(not t.is_contiguous() or t.is_neg() or t.is_conj() for t in (x, weight)):
        raise ValueError('ordinary contiguous tensors required')
    if x.requires_grad or weight.requires_grad:
        raise ValueError('inference only: no backward')
    if not 1 <= x.shape[1] <= MAX_COLS or x.shape[0] > 2**31-1:
        raise ValueError('requires 1 <= N <= 4096 and M <= 2**31-1')
    if cuda and x.device.type != 'cuda':
        raise ValueError('CUDA required for Triton path')
    return eps


def torch_reference(x, weight, eps=1e-5):
    """Unfused semantic baseline; accepts CPU or CUDA, same inference contract."""
    import torch
    eps = _validate(x, weight, cuda=False, eps=eps)
    return (x * torch.rsqrt((x*x).mean(dim=1, keepdim=True) + eps)) * weight


def _launch(x, weight, out, eps):
    """Internal unchecked entry. Caller ensures validated nonempty inputs/output."""
    import triton
    from rmsnorm_kernel import kernel
    kernel[(x.shape[0],)](x, weight, out, x.shape[1], eps,
        BLOCK=triton.next_power_of_2(x.shape[1]), num_warps=4)


def rmsnorm(x, weight, eps=1e-5):
    """New output [M,N]. Finite values/no FP32 overflow are NOT runtime scanned."""
    import torch
    eps = _validate(x, weight, cuda=True, eps=eps)
    with torch.cuda.device(x.device):
        out = torch.empty_like(x)
        if x.shape[0] != 0:
            _launch(x, weight, out, eps)
        return out


if __name__ == '__main__':
    print([[round(v, 6) for v in row] for row in
           reference_rows([[3., 4.], [0., 0.]], [1., 2.])])
```

入口顺序：先检查与归一化eps→进入对应CUDA设备上下文→分配输出→非空时发射kernel→返回Tensor。CUDA执行对主机通常异步；返回Tensor不表示GPU已完成，后续依赖由框架/stream语义管理，计时要使用正确同步方法。

内部 `_launch`不重复检查，是包装器的实现细节，不能作为无条件安全的外部API。这与后端服务中“先校验请求，再进入执行器”是同一种分层思路。

## 7 正确性测试和故障路径

解压后在目录内运行：

```sh
python rmsnorm.py
python -m unittest test_stdlib test_torch -v
```

示例预期：`[[0.848528, 2.262741], [0.0, 0.0]]`。

测试代码集中在 `test_stdlib.py`、`test_torch.py`，以下展示GPU测试的核心模式：

```python
torch.manual_seed(13)
for m, n in ((0,3), (1,1), (3,31), (3,33), (7,1025), (2,4096)):
    x = torch.randn(m, n, device="cuda")
    w = torch.randn(n, device="cuda")
    expected = torch_reference(x, w)
    torch.testing.assert_close(rmsnorm(x,w), expected,
                               rtol=2e-5, atol=2e-6)
```

| 场景 | 通过条件/要防止的错误 |
|---|---|
| M=0,N=3 | 返回shape[0,3]，无自定义kernel发射 |
| N=1 | 标量一行，负输入保留符号，gamma正确作用 |
| N=31/33/1025 | 与参考在容差内，尤其捕捉除BLOCK错误 |
| 全零、1e-8、正负常数 | 无NaN/Inf，eps对小值有可观察影响 |
| weight不同/负/为0 | 逐通道缩放正确，不能误用行权重 |
| N=0、shape不匹配、非连续、dtype不符、requires_grad | 报明确错误，不启动不支持路径 |
| CPU传给Triton入口 | 报CUDA要求；CPU参考仍可独立使用 |
| eps=0/负/NaN/Inf/1e-50/bool | 拒绝，避免隐含转0或错误类型 |

容差是本期测试输入范围的验收目标，不是任意FP32值的误差证明。随机测试不能代替边界测试。CPU参考与GPU实现都写错同一公式时可能一致，因此还用手算[3,4]、双精度对照及PyTorch内置 `functional.rms_norm`交叉检查。

**本轮实际验证**：标准库10PASS；PyTorchCPU3项和GPU3项均SKIP，环境未安装torch/triton；5个Python文件语法检查。没有运行GPU编译、数值测试、内存检查或benchmark，没有测得任何加速。全部记录见 `VALIDATION.md`。

## 8 如何测性能，不被“融合”二字误导

在已有兼容GPU环境下：

```sh
python benchmark.py --rows 4096 --cols 1025 --device 0
```

脚本先比较正确性并预热，再用 `triton.testing.do_bench(..., warmup=25, rep=100, quantiles=[0.5,0.2,0.8])`打印p50/p20/p80，warmup和rep预算及输出单位均为毫秒。它对比本课Triton包装器、PyTorch分解表达式与PyTorch内置RMSNorm。三个入口均创建输出，但工作量、临时Tensor数量和检查逻辑并不完全相同，因此这是**这些具体前向路径**的比较，不是单纯比较编译器kernel的公平冠军赛。[do_bench固定实现](https://github.com/triton-lang/triton/blob/v3.1.0/python/triton/testing.py#L78-L150)

需要分清：
- JIT首编译不纳入稳态结果，部署冷启动要另测。
- CUDA事件测量GPU时间区间；分配/主机提交可能影响区间，不能当作纯计算时间，也不是HTTP请求p99。
- 若要仅比较kernel，应为两侧实现同等预分配out契约，并独立报告；不要只给Triton去掉分配却给PyTorch保留。
- 输入与输出至少各读/写MN个FP32，weight逻辑上每行读取N个，但物理HBM流量受缓存复用影响。不要拿逻辑字节数冒充实际DRAM计数。
- 时间复杂度O(MN)，输出空间O(MN)，每program的逻辑工作空间O(BLOCK)。行太长可能增加寄存器压力，行太少可能GPU利用率不足；不能只看一次大矩阵。
- 建议测试M=1/32/4096、N=32/1025/4096，报告设备/版本/形状/容差/分位数，不先承诺速度提升。这里未测任何性能数值。

## 9 设计取舍：何时值得升级？

最简单可靠的应用层方案通常是现有框架算子。手写kernel的价值在于可测瓶颈、可控契约与学习，不是看到一个公式就必然重写。

| 选择 | 适用条件 | 代价/升级条件 |
|---|---|---|
| PyTorch表达式 | 先验证语义，CPU也可读/测 | 可能有中间Tensor；真正热点再优化 |
| PyTorch内置RMSNorm | 模型集成，依赖框架支持的布局/自动微分 | 先核实具体版本和eps，不假定与教学接口同能力 |
| 本课单program行融合 | 连续FP32、短中行、纯前向教学 | 不支持训练、低精度和任意stride，尚未GPU验证 |
| 分块/多阶段或专用融合 | 行长/带宽/发射已被测量为瓶颈 | 更多同步、中间结果、精度与维护成本；不提前堆复杂度 |

把残差加法与RMSNorm融合是合理下一步练习方向，但必须明确是归一化 `x+r` 还是 `RMSNorm(x)+r`，两者一般不同。输出是否还要保存残差也改变I/O契约和内存成本。

## 10 三个小改写练习

1. **看见契约失败**（约10分钟）：先读懂已有eps边界测试 `2**-126` 和1，再在已有PyTorch时补充weight为非连续切片的拒绝测试。要求合法边界能用、不支持布局明确拒绝，不能偷偷复制。说明 `*` 与关键字参数为什么减少误调用。
2. **故意写错再定位**（约10分钟）：仅在本地副本把除N改成除BLOCK。用N=32与33比较，解释为什么32可能过、33失败。最后恢复正确实现；验收是非2次幂测试能稳定捕捉错误，不把错误版本混入benchmark。
3. **显式fallback策略**（进阶约15分钟）：新增上层 `normalize_hidden`，对本课支持的CUDA输入选择Triton，否则调用框架参考，并返回/记录使用的backend。先定义哪些输入仍必须拒绝（例如无可用依赖、shape错误），不要把所有异常一概吞掉。用mock执行器测试路由，分别记录转换/计算成本。

下一课计划切到“可替换模型后端的最小接口与确定性mock”，把今天的契约/参考/实现分层用于Agent服务；随后回到归一化、激活、矩阵乘法、Attention，再逐步进入网络级MTP。服务接口、状态机、工具调用、队列/取消/幂等和算法按依赖穿插，避免把所有知识塞进同一天。

## 11 环境与来源

历史教学基线：Linux x86_64、Python3.10–3.12、PyTorch2.5.1 cu121、Triton3.1.0、兼容NVIDIA驱动。不是最新版本推荐，也不是本轮验证过的环境。Mac M4不能原生运行CUDA，标准库路径可用，不假定已有远程GPU。未安装软件、下载权重、训练或调用付费服务。

本期读取日期：**2026-11-01**。直接来源：
1. [PyTorch2.5.1 RMSNorm原生公式与epsilon](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/layer_norm.cpp#L248-L286)。采用实际实现确认公式，不直接照抄旧版文档中可能歧义的RMS记号。
2. [PyTorch2.5.1 nn.RMSNorm参数与forward](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/nn/modules/normalization.py)：weight初始化为1，forward调用functional.rms_norm，无额外bias。
3. [Triton3.1 sum](https://github.com/triton-lang/triton/blob/v3.1.0/python/triton/language/standard.py#L243-L248)、[rsqrt](https://github.com/triton-lang/triton/blob/v3.1.0/python/triton/language/math.py#L155-L160)。按固定版本使用API，不混用最新版sum的可选参数。
4. [Triton3.1官方LayerNorm教程](https://github.com/triton-lang/triton/blob/v3.1.0/python/tutorials/05-layer-norm.py)：参考行program、masked load和规约模式；本课RMSNorm不减均值，不能直接声称复制了同一算子。
5. [Triton3.1测试计时器](https://github.com/triton-lang/triton/blob/v3.1.0/python/triton/testing.py#L78-L150)、[PyTorch数值精度说明](https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/notes/numerical_accuracy.rst#L36-L47)。
6. [Root Mean Square Layer Normalization原论文](https://arxiv.org/abs/1910.07467)，2019-10-16提交，NeurIPS2019；[第4节](https://arxiv.org/html/1910.07467v1#S4)。本文借用定义与条件讨论，不引用论文速度数字作为本实现的收益。

没有读取或使用用户仓库源码作为依据；没有把旧报告当作本轮GPU实测。教学简化与未执行验证已在上文逐项说明。
