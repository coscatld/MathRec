# Lesson 09：从 Online Softmax 到单头 Attention

离线、CPU、标准库优先。无需安装依赖，不下载模型，不训练，不访问网络。
本课桥接 online softmax 与 Attention；**不实现 Triton kernel，也不是 FlashAttention 性能复现**。

## 快速运行（Python 3.10+；本轮实际为 3.12.14）

```bash
python -m unittest -v test_attention audit_checks
python example.py
python benchmark.py --repeats 5
python -m py_compile attention.py test_attention.py example.py benchmark.py
```

`test_attention.py` 在 PyTorch 未安装时跳过两个可选对照。无需为完成本课安装 PyTorch。
有 PyTorch 时用 CPU float64；SDPA 不保证采用何种底层 kernel，不能据此宣称 GPU 性能。

## 文件与接口

- `attention.py`：标准库实现及输入检查。
- `attention_materialized(q, k, v, keep=None)` 返回 `Trace`：`scores` 为缩放后的未 mask 分数，`masked_scores`、`probabilities`、`output` 逐层可观察。
- `attention_streaming(q, k, v, keep=None, block_size=4)` 返回输出矩阵。每行分块扫 K/V，不建立 L×S 中间矩阵。
- `causal_keep(query_positions, key_positions)` 从**绝对位置**构建布尔 mask。
- `example.py`：L=2、S=5 的 causal 示例，打印完整数据流。
- `test_attention.py`：随机、边界、Decimal 独立高精度及可选 PyTorch 对照。
- `benchmark.py`：小尺寸 Python CPU 中位延迟，仅教学参考，包含验证及分配成本。

## 契约

Q=[L,D]、K=[S,D]、V=[S,Dv]；L/S/D/Dv 在 1..256，输入为 list/tuple 矩阵。
数值仅 int/float（拒绝 bool），必须有限且绝对值≤10^6；拒绝空矩阵、ragged、错误形状及 NaN/Inf。
`keep` 可省略（全部可见），否则严格为 [L,S] 布尔矩阵：**True=允许**。
`block_size` 是 1..256 的整数，末尾块可不足块长；输入不会被修改。
全 mask 行的 P 和 O 明确为零，不尝试对全 -inf 做 softmax。
位置是 0..10^9 的整数，长度 1..256。允许非排序位置，比较实际位置而非矩阵下标。

例如 query=[5]、keys=[0,1,2,3,4,5] 的 decode query 应看到全部六个 key。
不要用非方阵的隐式三角对齐猜测语义。PyTorch 对照显式传 `attn_mask`、
`is_causal=False`、`dropout_p=0.0`；只对有效行调用 softmax/SDPA，再把全 mask 行填零。
注意 PyTorch 的不同 mask API 可能用相反约定；本代码只采用上述 keep 语义。

## Streaming 不变量

对于已经扫过且允许的 key：

- m = 最大 score
- l = Σ exp(score-m)
- acc[c] = Σ exp(score-m) × V[key,c]
- O[c] = acc[c] / l

新块到来，m_new=max(m, max(block_scores))，alpha=exp(m-m_new)。
**旧 l 和旧 acc 都要乘 alpha**，然后加新块贡献。初始空状态不算 -inf-(-inf)。
全 mask 块跳过，包括第一块；整行没有有效元素则输出零。

两个实现均计算 O(L×S×(D+Dv)) 次量级运算。
Materialized trace 额外存储 O(L×S+L×Dv)；streaming 工作空间为 O(block_size+Dv)，
另有 O(L×Dv) 输出。已有输入以及调用者提供的 L×S mask 不计在工作空间内。
这不是 end-to-end 常量内存实现；Python 循环不保证更快，也没有 GPU I/O 评测。
浮点求和顺序不同，所以测试使用容差，不要求逐位相等。

## 可练习的改动

1. 在每块结束打印 m/l/acc，手推两块合并。
2. 故意只缩放 l、不缩放 acc，观察大分数跨块测试失败。
3. 用 causal_keep 与 padding keep 做逐元素 AND；测试 padding 全 mask 行。
4. 固定输入，尝试 block_size=1、2、3、256，比较误差与 Python 时间。

完整中文讲义由同目录 Markdown 课件提供。下一课计划穿插后端可观测性，再回到 tile GEMM / GPU 布局；不把后端性能结论提前套在本课上。

## 环境与边界

本轮执行 Linux x86_64 / CPython 3.12.14；Python 3.10、3.11 和其他平台未复测。
没有第三方必需依赖；PyTorch/Triton 未安装。可选测试依赖已有的 PyTorch SDPA API，
未建立或验证 PyTorch 版本兼容矩阵，也未生成经过安装验证的 lockfile。
源码仅使用普通 CPU float64 对照，不含 Triton/CUDA kernel、训练、模型下载、付费调用或部署。
API 语义按 2026-11-07 核读的 PyTorch 2.15 官方文档说明；文档版本不是实测版本。

输入会被扫描但不复制，且不会由这些函数修改。调用期间不要从其他线程修改输入；
不支持恶意对象子类或变动的迭代行为。公开接口扫描与分配计入 benchmark。
