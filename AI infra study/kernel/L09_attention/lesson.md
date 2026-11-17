# 第9课 把 Attention 写完整：QKᵀ、mask、Softmax 与 V

日期：2026-11-07（北京时间）

**今天的目标：从一行 Softmax，走到一个可运行、可检查中间值的单头 Attention，再把它改成不保存完整分数/概率矩阵的分块计算。**

昨天已经解释了在线归一化状态 `(m, l)`，以及最大值变化时为什么必须重缩放。今天新增的是：分数从哪里来、mask 到底屏蔽什么、为什么最后输出的是 V 的加权和，以及如何让加权分子跟分母一起更新。预计主线 45–60 分钟，练习另计；不假设你已完成前课练习。

这是一节连接模型语义与 GPU 算子设计的课。主线只依赖 Python 标准库，附可选 PyTorch 对照。**今天没有新 Triton/CUDA kernel，也不把 Python 分块循环称为 FlashAttention 实现。** 先固定问题和正确性，再谈矩阵分块、线程映射和融合，能避免把 mask 错误藏在高性能代码里。

## 1 把输入和输出画成形状

本课只做一个 batch、一个 head，Q/K/V 已经给定：

- `Q[L, D]`：L 个 query，每个 query 有 D 个特征。
- `K[S, D]`：S 个 key，特征维度必须同为 D。
- `V[S, Dv]`：每个 key 对应一个 value，数量必须同为 S；Dv 不必等于 D。
- `keep[L, S]`：布尔矩阵，`True` 表示允许 query i 读取 key/value j。
- `O[L, Dv]`：最终输出，每个 query 得到一个 Dv 维向量。

计算路径是：

```text
Q[L,D] × Kᵀ[D,S] -> scores[L,S]
                 -> 缩放与允许关系
                 -> 对每个 query 沿 S 做 Softmax
                 -> probabilities[L,S] × V[S,Dv]
                 -> output[L,Dv]
```

其中：

$$s_{ij}=\frac{\sum_{r=0}^{D-1} Q_{ir}K_{jr}}{\sqrt D}.$$

对第 i 行允许的 key 集合 $A_i$，非空时：

$$m_i=\max_{j\in A_i}s_{ij},\quad
l_i=\sum_{j\in A_i}\exp(s_{ij}-m_i),$$

$$p_{ij}=\begin{cases}\exp(s_{ij}-m_i)/l_i & j\in A_i\\0 & j\notin A_i,\end{cases}
\qquad O_{it}=\sum_j p_{ij}V_{jt}.$$

Q、K 决定“读多少”，V 决定“读到什么”。如果只把 V 改掉，分数和概率不变，输出通常会变。这里的 `Kᵀ` 是数学上的转置，在 Python 标准库代码里不需要真的建立一个转置副本，只需按两个行向量做点积。

缩放分母用 D，不是 S，也不是 Dv。经典的直觉是在各维独立、均值为 0、方差为 1 的简化假设下，点积方差随 D 增长；除以 `sqrt(D)` 用于控制尺度。真实网络向量不一定满足这些假设，它是解释动机，不是对所有输入的保证。标准缩放点积定义见原论文 [Attention Is All You Need，§3.2.1](https://arxiv.org/html/1706.03762v7)。

本课没有实现 Q/K/V 的线性投影、多头拼接、输出投影、RoPE、dropout、反向传播、KV cache 管理或 MTP。一个 Attention 核心函数与完整 Transformer block 是不同范围。

## 2 先用三个数手算

令 `D=1`，一个 query 为 `[1]`，三个 key 为 `[0]、[log(2)]、[log(3)]`，value 分别为 `[10]、[20]、[40]`。

不加 mask 时，三个指数的相对比例是 `1:2:3`，因此：

```text
概率 = [1/6, 2/6, 3/6]
输出 = 10/6 + 40/6 + 120/6 = 170/6 ≈ 28.333333
```

如果中间 key 被屏蔽，参与归一化的是第 0、2 个位置：

```text
keep = [True, False, True]
概率 = [1/4, 0, 3/4]
输出 = 10/4 + 120/4 = 32.5
```

**错误写法：先算 `[1/6, 2/6, 3/6]`，再把第二项置零，直接乘 V。** 剩下的权重和只有 `4/6`。如果不重新归一化，得到的不是指定允许集合上的 Attention。

另外，不能把禁用位置的 score 写成 0。score 为 0 仍然产生 `exp(0)=1`，在全为负数的有效分数中甚至可能非常显著。实现可以用 `-inf` 表示禁用位置，也可以显式只遍历允许位置；关键是它们在分母和分子中都贡献 0。

## 3 两类 mask 不要混在一起

### 3.1 存储边界 mask

假设一个 GPU tile 宽度是 128，而本行只剩 3 个 key，剩余 125 个逻辑位置没有合法内存。这类 mask 保证不越界读取、写入。它不能替代模型的访问规则。

### 3.2 语义 mask

padding、因果关系、滑动窗口、文档边界都属于“内存存在但这次不能读”的规则。实际计算允许的元素必须同时满足存储边界和语义条件。

本课把语义约定写进参数名 `keep`：`True = 保留/允许`。不要只写一个模糊的 `mask` 然后凭印象猜真假方向。PyTorch SDPA 的布尔 mask 也是 True 参与，但 `MultiheadAttention.key_padding_mask` 的 True 表示屏蔽，两者方向相反。[PyTorch SDPA API](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html)

组合规则时，用逻辑与：

```text
允许 = key 不是 padding
    AND key 的绝对位置 <= query 的绝对位置
    AND key 与 query 属于允许关联的文档
```

这里的文档条件是扩展示意；配套实现只负责传入的布尔 keep 与显式绝对位置 causal mask，没有推断 padding 或文档身份。

## 4 非方阵 causal：shape 不能告诉你时间位置

常见但容易错的写法是：只要想 causal，就做 `j <= i`。

在完整 prefill 中，query 和 key 都来自位置 `[0,1,2,3]`，这没问题。可是在 decode 中，query 只有最新一个位置 `[3]`，key 为 `[0,1,2,3]`：

```text
局部行号 i = 0
query 的实际位置 = 3
正确 keep = [True, True, True, True]
错误的 j <= i = [True, False, False, False]
```

本课使用直接的定义：

$$\text{causal\_keep}_{ij}=(\text{key\_positions}_j\le \text{query\_positions}_i).$$

例如 query 位置 `[3,4]`、key 位置 `[0,1,2,3,4]`：

```text
[True, True, True, True, False]
[True, True, True, True, True ]
```

对连续且 query 位于 key 序列末尾的特例，这对应右下角对齐，偏移为 `S-L`。官方 `causal_lower_right` 明确使用这个偏移。[PyTorch causal_lower_right](https://docs.pytorch.org/docs/2.15/generated/torch.nn.attention.bias.causal_lower_right.html)

本课选择显式位置，不让函数从 `(L,S)` 猜场景。它也能表达稀疏保留的 key 位置；但调用者必须保证这些位置真实正确。位置允许非连续，不代表 key/value 已自动按位置排序、也不代表缓存已正确更新。

**可选 PyTorch 对照一律传入显式布尔 mask，`is_causal=False`。** 官方文档中的非方阵 `is_causal=True` 采用左上对齐，不是上述 decode 的末尾对齐。`dropout_p=0.0` 也明确传入。[PyTorch SDPA 参数说明](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html)

## 5 全屏蔽行必须定义，而不是等 NaN 出现

如果某行没有任何允许位置，普通公式的分母为 0。把整行设成 `-inf` 后直接减最大值，会出现 `-inf - (-inf)`，接着传播 NaN。

本课为教学接口做出明确选择：

- 全屏蔽行的概率全部为 0。
- 全屏蔽行输出为 Dv 维零向量。
- 它的概率和为 0，而不是 1。
- 这是一项接口约定，不是空集合 Softmax 本来就存在的概率分布。

实现先检查有效集合是否为空，再进入稳定 Softmax。可选 PyTorch 路径也显式处理空行，不把不同版本或后端的默认行为当成我们的契约。

注意区分“某个分块暂时全屏蔽”和“整行全屏蔽”。前一个分块可能没有可读 key，后一个却有；在线算法此时应保持空状态，继续扫描，不能提前返回零。

## 6 第一版：让中间值可见

`attention.py` 中的物化实现返回 scores、masked_scores、probabilities、output，适合逐行打印和定位错误。核心函数的完整代码会在后文附出，与 ZIP 中源码保持一致。

它按四步完成：

1. 校验输入，确保形状和数值范围明确。
2. 给每个 query/key 对生成缩放后的分数。
3. 在每一行的允许集合上计算稳定概率。
4. 对每个 value 维度，累加 `p[j] * V[j][t]`。

这里 `math.fsum` 用于更稳健地求 Python 浮点和，但输入乘法和指数仍是浮点运算，不能称为精确实数计算。手工 Decimal 参考测试提供额外检查，也不等于证明所有输入。

为什么需要一份慢但可读的参考？因为优化版可能同时改变循环顺序、内存布局和数值类型。如果参考中间值可查，发现错误时就能按顺序问：分数对吗？允许集合对吗？概率对吗？最后加权对吗？这比只看到最终输出相差 0.1 更容易定位。

### 校验函数在做什么

校验不是 Attention 的数学公式，它是接口边界。形状不对先报错，比在内层点积时得到错误广播或截断更清楚。本课故意用小规模、有限且有界的 Python 数据，拒绝空维度、ragged 列表、NaN/Infinity 和数值冒充布尔 mask。

数值校验是 O(输入大小)，dense keep 校验还要遍历 L×S 个布尔值。当前实现扫描已有输入，不建立 Q/K/V 的防御性副本；调用时不得由其他线程并发修改输入，也不把任意对象子类或恶意迭代行为当作受支持输入。后面的复杂度与基准会把它们算清楚，不能一边做完整输入扫描，一边把整个公开接口称为“只有常数内存、只读一遍”。具体契约是 L/S/D/Dv 与 block_size 均在 1..256；数值仅严格的 int/float（拒 bool），有限且绝对值不超过 10^6；位置是 0..10^9 的整数。_validate 不修改输入。

## 7 第二版：把概率乘 V 融进在线状态

对某一个 query，令当前已经读过的允许 key 集合为 A。定义：

$$m=\max_{j\in A}s_j,\quad l=\sum_{j\in A}e^{s_j-m},\quad
 a_t=\sum_{j\in A}e^{s_j-m}V_{jt}.$$

最终只需要：

$$O_t=a_t/l.$$

`m`、`l` 是两个标量，`a` 是 Dv 长度的向量。它不是 S 长度的概率向量。

当读入新块 B，新的最大值为 `m_new`。先把旧状态换到新单位：

$$\alpha=e^{m-m_{new}},\quad l_{new}=\alpha l+\sum_{j\in B}e^{s_j-m_{new}},$$

$$a_{new,t}=\alpha a_t+\sum_{j\in B}e^{s_j-m_{new}}V_{jt}.$$

最后令 `m = m_new`。**分母和向量分子必须使用同一个 alpha。** 如果只缩放分母，旧 value 的贡献保留了错误单位，最后除法无法修复。

### 一次很小的重缩放例子

令两个 score 分别为 `0、log(2)`，value 为 `10、20`，每块一个元素。

```text
第一块：m=0，l=1，a=10
第二块：m_new=log(2)，alpha=1/2
更新：l=1/2 + 1 = 1.5
      a=10/2 + 20 = 25
输出：25/1.5 = 50/3 ≈ 16.666667
```

如果漏掉旧分子缩放，就变成 `a=10+20=30`，错误输出为 20。这类“最大值在后面”的输入能直接暴露错误，不要只测最大值在第一块的情况。

### 空状态的处理

初始没有允许 key，定义 `l=0`、`a=0`，`m` 没有有效含义。实现可以用 `m=-inf` 作为标记，但必须避免计算两个负无穷相减。全屏蔽块跳过；第一个非空块建立状态；最终 `l==0` 时遵循零输出契约。

浮点下分块顺序可能改变最后几个有效位，所以测试用容限比较，不要求 bitwise 相同。这里保留了所有允许 key 的贡献，没有做 KV 采样、Top-K 近似或稀疏化。计算组织变了，不代表注意力的定义变了。

## 8 为什么这里能流式，而昨天完整 Softmax 需要二遍

完整 Softmax 的目标是返回每一个 `p[j]`。后来的最大值和分母会改变早期概率，因此通常要保留分数或再次读取。

今天最终目标是 `sum(p[j] * V[j])`。早期贡献可以聚合为一个 Dv 维向量，并在尺度变化时整体修正。你不再需要逐项恢复早期 `p[j]`，于是可以只保留聚合状态。

这给出一个通用架构判断：**先看下游最终要什么，再决定是否必须物化中间结果。**

- 调试要求返回完整 scores/probabilities：物化参考很有用。
- 推理只需要 Attention 输出：可以改写计算顺序，避免保存两张完整中间矩阵。
- 业务又要求输出完整注意力图：输出本身就有 L×S 个元素，不能继续声称该功能只有线性输出空间。

这不是只要把变量命名为“streaming”就会省显存。当前实现仍需要 Q/K/V、输出和调用者提供的 dense keep；我们只消除了完整 scores/masked_scores/probabilities 的保留。

## 9 复杂度与 GPU 数据搬运如何连接

忽略 mask 稀疏性带来的少算，单头稠密 Attention 的主要算术量约为：

```text
QKᵀ：2 × L × S × D 次浮点操作
P V：2 × L × S × Dv 次浮点操作
Softmax：O(L × S)
```

乘加按两次操作计数只是惯例。真实硬件指令数、吞吐和时间不能由这几行直接推出。

物化参考保存 scores、masked_scores 和 probabilities，共三张 `L*S` 中间矩阵。若只做超出本教学尺寸上限的理论 FP32 紧密数组估算，L=S=4096 时每张约 64 MiB，三张约 192 MiB。Python list/float 的真实占用远大于这个理想数组估计，不能把它当作本程序 RSS。

逐 query 的分块核心只维护当前块分数、两个标量和 Dv 向量。若块大小是 B，**核心额外工作空间**约为 O(B+Dv)，再加输出 O(L×Dv)。公开接口会扫描已有输入而不复制 Q/K/V，但输入本身、位置列表和可选 dense keep 的 O(L×S) 空间仍然存在。完整程序总内存不是 O(B+Dv)。

映射到 GPU 时，一个 query tile 会和多个 K/V tile 交互。想减少高带宽内存的中间写回，可以将分数块与当前累加状态在更靠近计算的位置处理；同时必须权衡寄存器、共享内存、占用率和并行度。FlashAttention 的重要出发点是这种 IO-aware 分块，而不是改变 Softmax 定义。[FlashAttention 原论文摘要](https://arxiv.org/abs/2205.14135)

官方 Triton Attention 教程可对照阅读：QK 分数块产生后，局部指数同时更新分母与输出 accumulator；后续新最大值会重缩放旧 accumulator。其当前示例还包含 TensorDescriptor、精度转换、硬件特化和反向等内容，本课没有复刻这些完整功能。[Triton Fused Attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)

注意官方示例的 `exp2`：若分数本来基于自然指数，需要把尺度乘 `log2(e)`。不要把 `exp` 直接换成 `exp2` 而不改尺度，也不要把不同对数底的辅助统计混为一谈。

## 10 测试应该针对不变量，而不是只比一组随机数

配套测试分为几组：

- **形状/输入边界**：ragged、维度不匹配、空维、非有限值、错误 keep 类型、非法 block。
- **数值小例子**：手算权重、均匀分数、只有一个允许 key、全屏蔽行。
- **因果对齐**：非方阵末尾 query、非连续绝对位置、改变未来 value 不影响过去输出。
- **在线状态**：首块全屏蔽、后块才有最大值、最后一块不足 B、多个 block 大小。
- **性质测试**：每个非空允许行的概率和约为 1；禁用位置概率严格为 0；无 dropout 时输出每个坐标落在允许 V 对应坐标的最小值和最大值之间。
- **独立参考**：小规模 Decimal 检查，避免两个实现共同复制同一个错误公式而“互相证明”。

坐标范围只是一项必要性质：错误的输出也可能落在区间内；要结合已知答案和独立计算。全屏蔽行不适用允许 V 的 min/max，因为集合为空。

“改未来 value 不影响过去输出”还需要保持 Q/K/位置不变。如果你把上游整个模型的输入都改了，Q/K 可能一起改变，那就不是这个局部函数的同一个实验。

## 11 怎样运行与读结果

解压 ZIP 后进入同名目录，先读 `README.md`。主线执行：

```bash
python3 -m unittest -v test_attention audit_checks
python3 example.py
python3 benchmark.py
```

无需 pip 安装。`requirements.txt` 只说明 PyTorch 对照路径，缺依赖时明确 SKIP；SKIP 不计为通过。CPU 标准库数值结果不验证 GPU 的 FP16/BF16/FP32 舍入、Tensor Core 累加、Triton JIT、显存访问或性能。

基准只用于观察 Python 实现的工作量，计入公开调用的校验、分配和 Python 循环开销。先做正确性检查，再 warmup，再用单调高分辨率时钟计时；代码会输出实际参数和重复次数。物化版本还返回调试中间值，流式版本只返回输出，两者返回内容并不完全对等，因此这个小基准尤其不能被解释为某个生产算法的普遍加速比。

如果将来做 GPU 对照，应先固定输出要求、dtype、layout、mask、是否包含分配/校验、后端选择和编译预热，再处理异步执行计时。内核吞吐、单请求时延、排队时延和端到端 Agent 时延是不同层次，不能混用一个数字。

本轮具体 PASS/SKIP 数量、环境、命令和结果见后附验证摘要及 ZIP 中 `VALIDATION.md`。没有 GPU 性能结论。

## 12 三个改写练习

### A. 让 causal 与滑动窗口同时成立

用绝对位置增加条件 `query_pos - W + 1 <= key_pos <= query_pos`，并与已有 keep 取交集。

验收：W=1、窗口大于历史、key 位置不连续、全屏蔽行都测试。先写出允许矩阵再跑数值函数；不要在 kernel 中凭局部行号猜位置。

### B. 故意只缩放分母

复制流式实现，删除旧分子乘 alpha 的一步。用第7节两块反例确认测试失败，再修复。

验收：解释错误发生在哪个不变量，而不只是说“结果不对”。同时比较“最大值先到”和“最大值后到”，理解单个简单例子为何可能漏检。

### C. 给 trace 加开关，设计公平比较

增加一个只返回 output 的物化路径，或让 trace 可选。保持所有方法输入/输出契约一致，分别测“包括校验”和“已校验核心”的耗时。

验收：明确哪些分配和扫描被移出了计时区；正确性检查在计时之前完成；报告环境与运行参数，不用 Python 数字推断 GPU 加速。若返回完整概率矩阵，单独报告它的空间成本。

练习放在副本里，保留经过测试的原版。是否完成练习以你的实际反馈为准。

## 13 读完以后应能回答

1. 为什么输出最后一维是 Dv，而不是 S？
2. 为什么全 mask 行需要单独约定？
3. L=1、S=4 为什么不能自动推出 `[True,False,False,False]`？
4. 为什么 a 与 l 必须同步重缩放？
5. 为什么不保存 P 不等于整个程序没有 L×S 内存？

本课留下的核心句子是：**Attention 最终需要的是概率加权后的向量；如果保留正确的尺度与累加不变量，就可以改变中间数据的保存方式，同时保持允许集合上的计算含义。**

下一课计划穿插后端可观测性：把排队时间、执行时间和调用者等待时间分开记录，继续第7课的并发/背压主线。之后再回到 GEMM 分块与 Attention kernel；不把每条主线都塞进同一天。

## 14 资料与版本边界

核读日期：2026-11-07。以下是概念和 API 语义的来源，不是本轮软件安装记录：

- [Attention Is All You Need，固定 v7](https://arxiv.org/html/1706.03762v7)：§3.2.1 的缩放点积定义、§3.2.3 的因果屏蔽。本文手算、代码与测试为教学实现。
- [PyTorch 2.15 SDPA](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html)：核对布尔 mask 方向、非方阵 causal 对齐、dropout 参数与后端数值差异。stable 链接本次跳转至 2.15；本轮未安装或执行 PyTorch。
- [PyTorch 2.15 causal_lower_right](https://docs.pytorch.org/docs/2.15/generated/torch.nn.attention.bias.causal_lower_right.html)：核对连续后缀 query 的 `S-L` 对齐示意；代码直接用绝对位置，不依赖该辅助 API。
- [FlashAttention，固定 v2 摘要页](https://arxiv.org/abs/2205.14135)：仅支持 IO-aware 分块的背景，不引用历史性能作为本课结果。
- [Triton main Fused Attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)：核读 forward 的分母/向量重缩放和 exp2 尺度；滚动文档，未运行官方示例，不声称与某个历史 Triton 版本完整兼容。


## 15 主实现完整源码

下面代码与 ZIP 中 `attention.py` 完全一致。先跟着 `attention_materialized` 看每层中间值，再读 `attention_streaming` 的状态更新。测试、示例和基准在同一 ZIP，无需复制网页代码才能运行。

```python
"""Single-head attention: transparent CPU reference, Python 3.10+, stdlib only.

Keep mask: True means allowed. Fully masked rows produce exact zeros.
Teaching limits prevent nonfinite intermediate arithmetic, not resource exhaustion.
"""
from dataclasses import dataclass
import math

MAX_DIM = 256
MAX_ABS = 1e6


@dataclass
class Trace:
    scores: list[list[float]]          # Q K^T / sqrt(D), BEFORE mask
    masked_scores: list[list[float]]   # forbidden scores replaced with -inf
    probabilities: list[list[float]]
    output: list[list[float]]


def _matrix(x, name):
    if not isinstance(x, (list, tuple)) or not 1 <= len(x) <= MAX_DIM:
        raise ValueError(f'{name}: require 1..{MAX_DIM} rows')
    width = None
    for row in x:
        if not isinstance(row, (list, tuple)) or not 1 <= len(row) <= MAX_DIM:
            raise ValueError(f'{name}: require nonempty bounded rows')
        if width is None:
            width = len(row)
        if len(row) != width:
            raise ValueError(f'{name}: ragged matrix')
        for value in row:
            if type(value) not in (int, float):
                raise ValueError(f'{name}: only int/float scalars, not bool')
            # Compare before conversion so arbitrarily large Python ints reject safely.
            if not -MAX_ABS <= value <= MAX_ABS or not math.isfinite(value):
                raise ValueError(f'{name}: require finite values within +/-{MAX_ABS}')
    return width


def _validate(q, k, v, keep):
    d = _matrix(q, 'Q')
    if _matrix(k, 'K') != d:
        raise ValueError('Q and K widths differ')
    dv = _matrix(v, 'V')
    if len(k) != len(v):
        raise ValueError('K and V row counts differ')
    if keep is not None:
        if not isinstance(keep, (list, tuple)) or len(keep) != len(q):
            raise ValueError('keep must have shape [L,S]')
        for row in keep:
            if (not isinstance(row, (list, tuple)) or len(row) != len(k)
                    or any(type(b) is not bool for b in row)):
                raise ValueError('keep must have shape [L,S] with strict bool entries')
    return d, dv


def causal_keep(query_positions, key_positions):
    """Absolute positions only: key_pos <= query_pos. No implicit L/S alignment.

    Decode example: query_positions=[5], key_positions=list(range(6)).
    Position order may be arbitrary; combining padding uses Boolean AND outside.
    """
    for positions in (query_positions, key_positions):
        if not isinstance(positions, (list, tuple)) or not 1 <= len(positions) <= MAX_DIM:
            raise ValueError('positions must be nonempty bounded sequences')
        if any(type(p) is not int or not 0 <= p <= 10**9 for p in positions):
            raise ValueError('positions must be nonnegative bounded integers')
    return [[kp <= qp for kp in key_positions] for qp in query_positions]


def _score(qrow, krow, scale):
    return math.fsum(a * b for a, b in zip(qrow, krow)) * scale


def attention_materialized(q, k, v, keep=None):
    """Return all intermediate matrices, for inspection rather than memory savings."""
    d, dv = _validate(q, k, v, keep)
    scale = 1.0 / math.sqrt(d)
    scores, masked, probs, outputs = [], [], [], []
    for i, qrow in enumerate(q):
        row = [_score(qrow, krow, scale) for krow in k]
        allowed = [True] * len(k) if keep is None else keep[i]
        masked_row = [s if ok else -math.inf for s, ok in zip(row, allowed)]
        if not any(allowed):
            p = [0.0] * len(k)  # Never evaluate -inf - (-inf).
        else:
            m = max(masked_row)
            weights = [math.exp(s - m) if ok else 0.0
                       for s, ok in zip(row, allowed)]
            denominator = math.fsum(weights)
            p = [w / denominator for w in weights]
        output = [math.fsum(p[j] * v[j][c] for j in range(len(k)))
                  for c in range(dv)]
        scores.append(row)
        masked.append(masked_row)
        probs.append(p)
        outputs.append(output)
    return Trace(scores, masked, probs, outputs)


def attention_streaming(q, k, v, keep=None, block_size=4):
    """Return O without an L*S score/probability allocation.

    Per row state after processed allowed keys:
      m = maximum score; l = sum exp(score-m)
      acc[c] = sum exp(score-m)*V[key,c]; O[c] = acc[c]/l.
    Rescale BOTH l and acc whenever the running maximum changes.
    Workspace excludes input, mask and output: O(block_size + Dv).
    """
    d, dv = _validate(q, k, v, keep)
    if type(block_size) is not int or not 1 <= block_size <= MAX_DIM:
        raise ValueError(f'block_size must be an integer in 1..{MAX_DIM}')
    scale = 1.0 / math.sqrt(d)
    outputs = []
    for i, qrow in enumerate(q):
        m, l = -math.inf, 0.0
        acc = [0.0] * dv
        for start in range(0, len(k), block_size):
            indices = [j for j in range(start, min(start + block_size, len(k)))
                       if keep is None or keep[i][j]]
            if not indices:
                continue  # Includes a fully masked FIRST block.
            scores = [_score(qrow, k[j], scale) for j in indices]
            new_m = max(m, max(scores))
            alpha = 0.0 if l == 0.0 else math.exp(m - new_m)
            weights = [math.exp(s - new_m) for s in scores]
            l = alpha * l + math.fsum(weights)
            acc = [alpha * acc[c] + math.fsum(w * v[j][c]
                   for w, j in zip(weights, indices)) for c in range(dv)]
            m = new_m
        outputs.append([a / l for a in acc] if l else [0.0] * dv)
    return outputs
```

### 按代码逐句定位

- `_matrix` 检查 Python 容器、矩形形状、严格数值类型与范围；它不是 tensor 广播规则，也不将输入复制成另一份数组。
- `_validate` 负责 Q/K 的 D 和 K/V 的 S 对齐，以及 keep 的精确形状。它返回 d、dv，供分数缩放与输出循环使用。
- `_score` 中 `zip(qrow, krow)` 按特征维配对；宽度已经校验，否则 zip 的静默截断会掩盖错误。
- `masked_row` 只改变分数参与关系，不改变 V；被屏蔽项对应的概率为零。
- `indices` 只列出当前块内允许的 key；`if not indices: continue` 避免空块 max 和无效指数。
- `alpha = 0.0 if l == 0.0 ...` 明确空状态的单位转换；随后 l 与 acc 同时重缩放。
- `acc[c]` 对应一个 value 特征，不是某个 key 的概率；`for c in range(dv)` 解释了状态为何是 Dv 长度。

## 16 本轮验证摘要

19 项测试中 17 PASS、2 SKIP：11 项原测试通过，6 项独立审计通过；两项可选 PyTorch 对照因未安装依赖而跳过。原测试含 45 组固定种子随机输入 × 6 个 block_size，以及 70 位 Decimal 小规模独立参考。5 份 Python 源码语法检查、示例、小尺寸 CPU 基准、ZIP CRC/字节一致性和全新解压复测通过。

环境为 Linux x86_64、CPython 3.12.14。没有 PyTorch/Triton 执行、GPU 数值/JIT/性能验证、安装、模型下载、训练或部署。更完整的命令、边界及一次运行记录见 `VALIDATION.md`、`TEST_OUTPUT.txt`。
