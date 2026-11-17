# GPU 算子学习计划

> 版本 2026-11-15 · 起点 **2026-11-16（周一）** · 主线 9 周（至 2027-01-17），Phase 4 长期
> 来源：你的四阶段路线图（原文见[附录 A](#附录-a路线图原文)）+ 仓库现状（`kernel/L01–L09`）。
> 滚动计划：W1–W2 排到天；之后排到周，每个阶段结束时重排下一阶段。
> 假设：每天约 2 小时，周日兼作缓冲 / 复盘日。你的实际时间不同就整体平移，**别压缩退出标准**。

## 1. 总览

| 阶段 | 周期 | 日期 | 做什么 | 退出标准 |
|---|---|---|---|---|
| 0 环境与基线 | 3 天 | 11-16 ~ 11-18 | 拿到可用的 NVIDIA GPU；把已有 5 个 Triton 课在 GPU 上跑通并记数 | torch + triton + GPU 在线；L01 / L02 / L03 / L06 / L08 的 GPU 结果入账；GPU 峰值规格已记录 |
| 1 Triton：块并行直觉 | 11 天 | 11-19 ~ 11-29 | Vector Add、Fused Softmax、RMSNorm / LayerNorm（含反向）、MatMul（tiling + swizzle） | 4 个算子手敲完成、对拍 PyTorch 通过、带宽 / TFLOPs 对照表齐全 |
| 2 CUDA：访存瓶颈 | 3 周 | 11-30 ~ 12-20 | Reduce、Transpose、向量化 Element-wise；学会用 ncu | 每个算子有版本阶梯表 + ncu 证据；向量化 elementwise ≥ 峰值带宽 85% |
| 3 两个大算子 | 4 周 | 12-21 ~ 2027-01-17 | GEMM（Siboehm v1→v10、Tensor Core、CUTLASS 入门）；FlashAttention（Triton 版 → v2 → PagedAttention） | GEMM 逼近 cuBLAS 90%；Triton FA 前向正确且显存不随 N² 增长；paged attention toy 对拍通过 |
| 4 科研微创新 | 长期 | 2027-01-18 起 | 在你自己的 KV Cache 压缩 / 量化代码里，用 profiler 选瓶颈，写融合 kernel | 里程碑 M0–M5（见 §8） |

阶段周期取路线图区间的上限（Phase 1 为 1~2 周、Phase 2 为 2~3 周、Phase 3 为 3~4 周），因为这里要手敲并且每步记数。进度好就提前进入下一阶段。

## 2. 现状盘点（2026-11-15）

| 路线图要求 | 仓库里已有 | 缺口 |
|---|---|---|
| P1 · Vector Add | [L01](kernel/L01_residual_add/lesson.md) | GPU 实测；你自己手敲 |
| P1 · Fused Softmax / Online Softmax | [L06](kernel/L06_softmax/lesson.md)（整行稳定 softmax）、[L08](kernel/L08_online_softmax/lesson.md)（分块 online softmax + 可选 Triton kernel） | 同上 |
| P1 · RMSNorm / LayerNorm + 反向重计算 | [L03](kernel/L03_rmsnorm/lesson.md)（只有 RMSNorm 前向，`requires_grad` 直接拒绝） | LayerNorm；两者的反向 |
| P1 · Naive MatMul（2D tiling + swizzle） | 无 | 全部 |
| P2 · CUDA（Reduce / Transpose / 向量化 / ncu） | 无 | 全部 |
| P3 · GEMM | 无 | 全部 |
| P3 · FlashAttention | [L09](kernel/L09_attention/lesson.md)（Python 参考：QKᵀ、mask、在线 softmax、V；**没有 kernel**） | Triton FA、PagedAttention |
| P4 | 无 | 全部 |

三个会影响排期的事实：

1. **已有 Triton 代码全部没在 GPU 上跑过。** 生成讲义的环境里没有 torch / triton，GPU 用例要么 SKIP 要么没执行（见 [README 的验证状态](README.md#验证状态先读这一段)）。GPU 上跑出 bug 是预期之内，修 bug 本身就是练习。
2. **本机是 Apple M4。** 没有 NVIDIA GPU，Triton 也不支持 macOS，所以 Phase 0 必须先解决 GPU 的来源。Mac 上能做的：写 PyTorch 参考、测试和 benchmark 脚手架（conda 环境 `study` 里的 torch 是 CPU / MPS 版），读讲义和论文。
3. **每日讲义的顺序和本路线不同。** 讲义是朝 MTP 设计的（已经讲到 Attention，还没讲 MatMul）。本计划把 L01–L09 当作**阅读材料**，主线是你手敲的实现；L09 用作 FlashAttention 的 CPU 参考和前置预习。

## 3. 工作方式与验收规范

**每个算子四件套**（把路线图里的「手敲」和「do_bench 对比」落成动作）：

1. **参考**：先写 PyTorch 参考实现和手算小例子（CPU 即可，Mac 上就能做）。
2. **手敲 kernel**：不抄讲义代码，卡住了再对照 `kernel/L**`。
3. **正确性**：`torch.testing.assert_close`；固定覆盖空输入、1、BLOCK−1 / BLOCK / BLOCK+1、非对齐尺寸、极值或大偏移。
4. **性能**：Phase 1 用 `triton.testing.do_bench`（显式给 `quantiles`；`warmup` / `rep` 是毫秒预算，不是迭代次数），Phase 2 起加 ncu。同 shape、同 dtype 与 PyTorch / cuBLAS 对照，并用一句话解释差距。

**目录约定**（开工时创建）：

```text
practice/
├── RESULTS.md                     全局成绩单（列见下）
├── phase1_triton/{01_vector_add, 02_fused_softmax, 03_norm, 04_matmul}/
├── phase2_cuda/{reduce, transpose, elementwise_vec}/
├── phase3_gemm_fa/{gemm, flash_attn_triton, paged_attn_toy}/
└── phase4_kv_kernels/<project>/
每个算子目录：kernel.*   test_*.py   bench.py   NOTES.md（结果表 + 一句话结论）
```

**RESULTS.md 的列**：日期 · GPU · 算子@版本 · shape / dtype · 中位耗时 ms（p20–p80）· GB/s 或 TFLOPs · 占峰值 % · 相对 torch / cuBLAS · 备注。
文件顶部固定写：GPU 型号、compute capability、显存带宽与 FP32 / FP16 峰值（以官方规格为准，如 RTX 4060 约 272 GB/s）、驱动 / CUDA / torch / triton 版本。以后所有「占峰值 %」都以它为分母。

**计时纪律**（沿用讲义里的做法）：先编译并通过正确性，再计时；预热 + `synchronize`；对比双方都预分配输出；记录版本、GPU、shape、dtype；多轮取中位数，不挑最好的一次；JIT 时间、分配、参数校验有没有计入要写明。

**每日讲义怎么用**：讲义是阅读材料，不替代手敲。想让讲义和计划对齐，就把下面的主题队列交给生成讲义的任务：LayerNorm（前向 + 反向）→ MatMul tiling / swizzle → CUDA reduce 阶梯 → transpose 与 bank conflict → 向量化访存 → ncu 读法 → SGEMM 阶梯 → WMMA / Tensor Core → Triton FlashAttention → PagedAttention → 量化 KV attention。`serving/` 支线与本路线互不依赖，计划里不排课；时间紧时以主线为先。

## 4. Phase 0 — 环境与基线（11-16 ~ 11-18）

**要先做一个决定：GPU 从哪来。**（待你确认）

| 选项 | 适合 | 注意 |
|---|---|---|
| 云 GPU 实例（Linux + NVIDIA） | 全程，尤其 Phase 2 起 | 先确认能跑 `ncu`：不少容器环境不开放性能计数器，会报 `ERR_NVGPUCTRPERM`，选之前先试 |
| 自有或实验室的 NVIDIA 机器 | 全程 | 讲义基线是 Linux x86_64；Windows 需要 WSL2 |
| Colab / Kaggle 免费 GPU（T4 等） | Phase 1 的临时方案 | 通常用不了 ncu；部分特性（如 bf16、新架构 Tensor Core 路径）受限，需自行验证 |

**验收清单**

- [ ] `nvidia-smi` 正常；`python -c "import torch, triton; print(torch.__version__, triton.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"` 正常。讲义基线是 torch 2.5.1（cu121）+ triton 3.1.0、Python 3.10–3.12；用更新的版本也行，但要记录版本并留意 API 差异
- [ ] `nvcc --version` 正常（Phase 2 前）
- [ ] `ncu --version` 正常，并且**真的采集过一次**（Phase 2 前必须）
- [ ] 在 `practice/RESULTS.md` 顶部写入 GPU 规格
- [ ] 在 GPU 上跑 L01（`--gpu --bench`）、L02、L03、L06、L08 的 GPU 测试和 benchmark，结果（含失败）记进 RESULTS.md 的「基线」一节

**如果 11-18 仍没有 GPU**：不要空等。先在 Mac 上写 Phase 1 全部算子的 PyTorch 参考、测试用例和 bench 脚手架（不依赖 triton），拿到 GPU 后只剩 kernel 本身。Phase 1 整体顺延，退出标准不降。

## 5. Phase 1 — Triton：以 Block 为单位的并行直觉（11-19 ~ 11-29）

目标：跳过线程同步和 shared memory 管理，只聚焦分块（Tiling）和访存逻辑。下表的阈值是建议值，可按实测调整。

| # | 算子 | 要练的点 | 已有讲义 | 要补的 | 验收 |
|---|---|---|---|---|---|
| 1 | Vector Add | `tl.program_id` / `tl.arange` / mask 尾块 | L01 | 手敲；GPU 实测 | N ∈ {0, 1, B−1, B, B+1, 2²²} 与 torch 一致；带宽 ≥ `torch.add` 的 90% |
| 2 | Fused Softmax | Online Softmax；融合为什么省 HBM 读写 | L06、L08 | 手敲；写访存账 | 对拍 `torch.softmax`（含 ±1e4 偏移）；写出融合前后各读写几次 HBM（L08 的实现估算为约 3·M·N·dtype 字节）；与 torch 耗时对照 |
| 3 | RMSNorm / LayerNorm | 行级归约；反向传播重计算 | L03（仅 RMSNorm 前向） | LayerNorm 前向；两者反向（重算 mean / rstd；dW / dB 要跨行归约） | 前向对拍 `F.layer_norm` / `F.rms_norm`；反向对拍 autograd 的 x、weight、bias 梯度 |
| 4 | Naive MatMul | 2D tiling；L2 命中率优化（swizzle / grouped ordering） | 无 | 全部 | 对拍 `torch.matmul`（fp16 输入、fp32 累加，容差自己论证）；`GROUP_SIZE_M` 对比表；TFLOPs 占 cuBLAS 的百分比 |

**官方 Tutorials 01–06 打卡**（Vector Addition / Fused Softmax / Matrix Multiplication / Low-Memory Dropout / Layer Normalization / Fused Attention）：每个算子写完后对照同名教程，把差异写进 NOTES.md。04 Dropout 是选做（降级项），06 Fused Attention 留到 Phase 3。

**逐日安排**

| 日期 | 内容 |
|---|---|
| 11-16 ~ 11-18（周一至周三） | Phase 0 |
| 11-19 周四 | Vector Add：手敲、对拍、bench |
| 11-20 周五 | Fused Softmax（整行版）：手敲、对拍 |
| 11-21 周六 | Online Softmax（分块版），对照 L08；写访存账；bench |
| 11-22 周日 | 缓冲 / 周复盘，补齐 RESULTS.md |
| 11-23 周一 | LayerNorm 前向 |
| 11-24 周二 | LayerNorm 反向（含 dW / dB 归约） |
| 11-25 周三 | RMSNorm 反向、对拍；Norm 的 bench |
| 11-26 周四 | MatMul：基础 2D tiling |
| 11-27 周五 | MatMul：grouped ordering（swizzle）+ `GROUP_SIZE_M` 对比 |
| 11-28 周六 | MatMul bench（对 cuBLAS）；教程核对表 |
| 11-29 周日 | Phase 1 退出评审 |

**退出评审：不看资料，能讲清这 5 件事**

1. Triton 的 program 与 CUDA thread 的区别；`BLOCK` 是逻辑元素数而不是线程数。
2. mask 为什么 load 和 store 都要加。
3. Online Softmax 的状态更新（m、l），以及「分块各自 softmax 再拼起来」为什么是错的。
4. 为什么把 reduce 融进同一个 kernel 能省 HBM I/O（用你自己测的数字回答）。
5. swizzle 为什么能提升 L2 命中率。

## 6. Phase 2 — CUDA：死磕访存瓶颈（11-30 ~ 12-20）

核心只有两件事：让算力单元别闲着（compute-bound），或者别让数据在显存里来回搬运（memory-bound）。

**算子阶梯**（每个版本都记 ms、GB/s、占峰值 %，并用 ncu 说明「为什么变快」）：

| 算子 | 阶梯 | 要看的 ncu 证据 |
|---|---|---|
| Reduce（求和） | v0 全局内存直接累加 → v1 shared memory 树形归约 → v2 解决 bank conflict（顺序寻址）→ v3 每线程多元素加载、减少空闲线程 → v4 warp shuffle（`__shfl_down_sync`）+ 跨 warp 汇总 | shared memory bank conflict 次数、DRAM 吞吐 %、warp stall 原因 |
| Transpose | v0 朴素（读合并、写跨步）→ v1 shared memory tile（读写都合并）→ v2 padding（如 `tile[32][33]`）消除 bank conflict；以同尺寸 copy kernel 的带宽为上限 | 全局访存合并效率、bank conflict 次数 |
| Element-wise 向量化 | 标量 → `float4` / `half2`（一次读写 128-bit）；处理对齐与尾部 | DRAM 吞吐 % ≥ 峰值的 85% |

**ncu 用法**：不要盲猜。每个版本跑一次 `ncu --set full`，只看四处：SpeedOfLight（SM% 对 Memory%）→ Roofline → MemoryWorkloadAnalysis（L1 / L2 命中、DRAM 吞吐、bank conflict）→ WarpStateStats（stall 原因）。指标名随版本变化，用 `ncu --query-metrics` 查。对 Triton 写的 kernel 也能用，加 `-k regex:<kernel 名>` 过滤。

**周计划**

| 周 | 日期 | 内容 |
|---|---|---|
| W3 | 11-30 ~ 12-06 | nvcc / ncu 工具链走通；Reduce 全阶梯 |
| W4 | 12-07 ~ 12-13 | Transpose；向量化 Element-wise |
| W5 | 12-14 ~ 12-20 | 用 ncu 回看 Phase 1 的 Triton kernel（softmax / norm / matmul 的实际带宽与瓶颈）；整理《ncu 指标速查》；缓冲 |

资源（路线图指定）：LeetGPU、谭升博客、开源题库。补充：Mark Harris《Optimizing Parallel Reduction in CUDA》（NVIDIA）。

**退出标准**：3 个算子的阶梯表齐全，且每一步都有 ncu 证据；向量化 elementwise 达到峰值带宽的 85% 以上；能说清 coalescing 与 bank conflict 的区别，以及各自在 ncu 里怎么看出来。

## 7. Phase 3 — 攻克两个大算子（12-21 ~ 2027-01-17）

### 3A · GEMM（W6–W7：12-21 ~ 2027-01-03）

- 跟 Siboehm《How to Optimize a CUDA Matmul Kernel》，从 v1 Naive 手写到 v10（目标逼近 cuBLAS 的 90%）。每个版本记 GFLOPS、占 cuBLAS 的百分比，再加一句「改了什么、为什么快」。大致的演进是：合并访存 → shared memory 分块 → 寄存器分块 → 向量化 → warp 分块。
- W7 下半：双缓冲（Double Buffering / Software Pipelining）；Tensor Core 入门（WMMA，一个 16×16×16 的小例子，需要 sm_70+）；CUTLASS **只读不写**，能读懂一个配置即可（2.x：ThreadblockShape / WarpShape / InstructionShape / stages；3.x：CuTe 的 Layout / TiledMMA）。
- 周计划：W6 = v1–v5；W7 = v6–v10 + 双缓冲 + WMMA / CUTLASS 阅读（含元旦，按需顺延）。

### 3B · FlashAttention（W8–W9：2027-01-04 ~ 2027-01-17）

- **W8**：先在 CPU 上用 L09 的 `attention.py` 当 oracle，验证在线更新公式；再手敲 Triton 版 FlashAttention 前向（路线图说核心约 150 行；官方教程 06 的当前版本更长，含反向、TensorDescriptor 和硬件特化，先只抠前向核心）。对拍 `F.scaled_dot_product_attention`（causal 与非 causal，序列长度不是 BLOCK 的整数倍）；用 `torch.cuda.max_memory_allocated` 对比朴素 attention，验证 N×N 矩阵不再落到 HBM；TFLOPs 对 SDPA。反向传播选做，只读也行。
- **W9**：FlashAttention v1 与 v2 对照阅读（外层 / 内层循环顺序、减少非 matmul FLOPs、沿序列长度并行）；阅读 FlashInfer / vLLM 的 PagedAttention kernel；**手写一个 toy**：decode（q_len = 1）阶段的 paged attention。KV 存成物理块（布局自定，例如 `[num_blocks, block_size, H, D]`），`block_table[b, i]` 给出序列 b 第 i 个逻辑块的物理块号，kernel 按 block table gather，再用在线 softmax 累加，对拍连续 KV 的版本。这个 toy 同时是 Phase 4 的跳板。

能默写的在线更新（对每个 K/V 块）：

```text
m' = max(m, rowmax(s));   p = exp(s − m')
l' = l·exp(m − m') + rowsum(p)
acc' = acc·exp(m − m') + p @ V          # 最终 O = acc / l
```

**退出标准**：GEMM 阶梯表（v1–v10，含占 cuBLAS 的 %）；Triton FA 前向正确，显存与 TFLOPs 曲线齐全；paged attention toy 对拍通过；一页《block table 寻址》笔记（含图）。

## 8. Phase 4 — 结合科研场景做微创新（2027-01-18 起，长期）

原则：练算子最怕脱离实际需求。从你自己的 KV Cache 压缩 / 淘汰（Compaction / Eviction）或量化实验里，**用数据**选一个瓶颈。

**选题四问**

1. 用 `torch.profiler` 看，它占多少 GPU 时间？（不凭感觉）
2. 现在是几次 kernel launch、几次 HBM 往返？融合后的理论下限是多少？
3. 数值误差预算怎么度量？（与 fp16 参考实现的 attention 输出相对误差，加下游任务指标不降）
4. 有没有工业实现可以对照？（vLLM `csrc/`、SGLang `sgl-kernel/`、FlashInfer）

**两个候选方向**（来自路线图）：

- **A. Fused Quantized Attention Kernel**：如 K 4-bit / V 2-bit，把反量化与 QKᵀ、PV 融进一个 kernel。
- **B. Custom Gather-GEMV Kernel**：压缩 / 淘汰之后，按索引 gather 保留下来的 KV，再与 q 做 GEMV。

**里程碑**（每周一个）

| 里程碑 | 产出 |
|---|---|
| M0 基线 | profiler 报告：Top-N kernel、launch 次数、显存碎片与峰值；PyTorch 拼凑版的耗时 |
| M1 microbench 与判据 | 固定 shape 的 microbench；fp16 / fp32 参考实现；误差容限的定义 |
| M2 能跑 | Triton v1：功能正确，对拍 M1 |
| M3 变快 | tiling、`num_warps` / `num_stages` 调参、向量化访存；用 ncu 证明 |
| M4 对照工业实现 | 与 FlashInfer / vLLM / SGLang 的 kernel 比较：变长序列（Ragged）、GQA、动态 shape 各自怎么处理 |
| M5 集成 | 接回你的实验代码；端到端 tokens/s + 质量指标 |

**完成的定义**：microbench 和端到端两层数字都有；误差在预算之内；相对 M0 基线的提速有 profiler 证据。

## 9. 风险与降级

| 风险 | 对策 |
|---|---|
| 没有 GPU，或 ncu 用不了 | Phase 0 先解决；拿不到就先做 CPU 脚手架。Phase 2 起没有 ncu 就不开工，否则只能「盲猜性能」 |
| 进度落后 | 按顺序砍：Phase 1 的 Dropout → Phase 3 的 CUTLASS 实操（保留阅读）→ Phase 2 的 reduce 中间版本。**不砍**：手敲、对拍、记数 |
| 只读不写 | 手敲规则；每个算子都必须有 NOTES.md 和一行 RESULTS |
| 讲义代码有 bug | 它们没上过 GPU；以官方文档和你自己的实测为准 |
| 日期冲突（考试 / 假期） | 整体顺延；以退出标准为准，不以日期为准 |

## 10. 进度追踪

**Phase 0**
- [ ] GPU 来源已定
- [ ] torch / triton / nvcc / ncu 验证通过
- [ ] GPU 规格写入 RESULTS.md
- [ ] L01 / L02 / L03 / L06 / L08 的 GPU 基线入账

**Phase 1**
- [ ] 01 Vector Add
- [ ] 02 Fused Softmax（整行 + 分块）
- [ ] 03 Norm（LayerNorm 前向 / 反向，RMSNorm 前向 / 反向）
- [ ] 04 MatMul（tiling + swizzle）
- [ ] 官方 Tutorials 01–06 核对
- [ ] 退出评审 5 问通过

**Phase 2**
- [ ] Reduce 阶梯（v0–v4）
- [ ] Transpose（v0–v2）
- [ ] 向量化 Element-wise ≥ 峰值带宽 85%
- [ ] 《ncu 指标速查》

**Phase 3**
- [ ] GEMM v1–v10
- [ ] WMMA 小例子
- [ ] CUTLASS 阅读笔记
- [ ] Triton FlashAttention 前向
- [ ] paged attention toy
- [ ] 《block table 寻址》笔记

**Phase 4**
- [ ] M0 基线
- [ ] M1 microbench 与判据
- [ ] M2 能跑
- [ ] M3 变快
- [ ] M4 对照工业实现
- [ ] M5 集成

## 11. 资源

路线图指定：

- Triton 官方 Tutorials：<https://triton-lang.org/main/getting-started/tutorials/index.html>
- LeetGPU、谭升博客、开源题库（Phase 2）
- Siboehm《How to Optimize a CUDA Matmul Kernel》：<https://siboehm.com/articles/22/CUDA-MMM>
- FlashInfer：<https://github.com/flashinfer-ai/flashinfer>　vLLM：<https://github.com/vllm-project/vllm>　SGLang：<https://github.com/sgl-project/sglang>

补充：

- FlashAttention：<https://arxiv.org/abs/2205.14135>　FlashAttention-2：<https://arxiv.org/abs/2307.08691>
- Online softmax 原始论文：<https://arxiv.org/abs/1805.02867>
- PagedAttention（vLLM）：<https://arxiv.org/abs/2309.06180>
- Nsight Compute 文档：<https://docs.nvidia.com/nsight-compute/>
- Mark Harris，《Optimizing Parallel Reduction in CUDA》（NVIDIA）

## 附录 A：路线图原文

<details>
<summary>展开（你的原始版本，仅按阶段分段，措辞未改；<code>__shfl_down_sync</code> 里的下划线加了转义以便正确渲染）</summary>

> 第一阶段：用 Triton 建立“以 Block 为单位”的并行直觉（1~2 周）现在做大模型科研和工程，很多自定义算子（尤其是各种 KV Cache 压缩、Attention 变体）首选 OpenAI Triton 而非手写 CUDA，因为它能让你跳过繁琐的线程同步和共享内存管理，直接聚焦分块（Tiling）和访存逻辑。必练算子清单（按顺序手敲一遍）：Vector Add：熟悉 tl.program_id、tl.arange、Mask 边界处理。Fused Softmax：理解 Online Softmax 算法，体会为什么把 Reduce 操作融进一个 Kernel 能省掉大量的显存读写（HBM I/O）。RMSNorm / LayerNorm：掌握行级归约（Row-wise Reduction）和反向传播重计算。Naive MatMul（分块矩阵乘）：学习 2D Tiling 和 L2 Cache 命中率优化（Swizzling）。学习资源：直接刷 Triton 官方 Tutorials 的前 6 个案例，每写完一个，用 triton.testing.do_bench 跟 PyTorch 原生算子测一下带宽和耗时对比。
>
> 第二阶段：下沉到 CUDA，死磕“访存瓶颈”（2~3 周）算子优化的核心本质只有两件事：让算力单元别闲着（Compute-bound），或者别让数据在显存里来回搬运（Memory-bound）。必须用 CUDA 摸透 GPU 硬件架构。必练算子清单（刷经典 LeetGPU / 谭升博客 / 开源题库）：Reduce（归约求和）：从全局内存直接累加，一步步优化到共享内存（Shared Memory）、解决 Bank Conflict（存储体冲突）、再到 Warp Shuffle（\_\_shfl\_down\_sync） 无锁归约。这是 CUDA 面试和实战的绝对基本功。Transpose（矩阵转置）：练合并访存（Coalesced Memory Access）以及 Shared Memory Padding 消除 Bank Conflict。Element-wise 向量化访存：使用 float4 / half2 一次读写 128-bit 数据，把带宽打满到硬件理论峰值的 85% 以上。核心工具：学会用 Nsight Compute (ncu) 看指标。不要盲猜性能，看一眼 ncu 的 Roofline 模型、SM 活跃度、L1/L2 命中率和 Bank Conflict 次数，比看十篇博客都管用。
>
> 第三阶段：攻克大模型最核心的两个大算子（3~4 周）掌握基础后，直接啃大模型推理/训练中最重头的两个算子，吃透它们的源码等于打通任督二脉：GEMM（通用矩阵乘法）：推荐跟着 Siboehm 的《How to Optimize a CUDA Matmul Kernel》 博客，从版本 1（Naive）一路手写到版本 10（逼近 cuBLAS 90% 性能）。重点理解：双缓冲（Double Buffering / Software Pipelining）、Tensor Core（WMMA / MMA 指令）以及 CUTLASS 的抽象设计。FlashAttention（v1 / v2）：不要直接看官方几千行的 CUDA 模板代码，先看 Triton 版的 FlashAttention 实现（大约 150 行），弄懂 $Q, K, V$ 分块在 SRAM 中如何通过 Online Softmax 增量更新输出 $O$，彻底消除 $N \times N$ 注意力矩阵的 HBM 读写。之后再去对照看 FlashInfer 或 vLLM 里的 PagedAttention Kernel，理解非连续显存块（Block Table）下的访存寻址是怎么写的。
>
> 第四阶段：结合具体科研与工程场景做“微创新”（长期）练算子最怕脱离实际需求。当你有了一定基础后，最好的练手方式是改写现有推理框架中的瓶颈算子：针对长上下文与 KV Cache 优化：比如在做 KV Cache 压缩（Compaction / Eviction）或非对称量化（如 K 4-bit / V 2-bit）时，PyTorch 拼凑出来的 gather、matmul 和重排操作会有极高的 Kernel Launch 开销和显存碎片。你可以尝试写一个 Fused Quantized Attention Kernel 或 Custom Gather-GEMV Kernel，把反量化和矩阵乘融进一个算子里。对照工业级算子库抄作业：遇到不知道怎么写的算子，直接去翻 vLLM (csrc/)、SGLang (sgl-kernel/) 或 FlashInfer 的源码。看工业界是如何处理变长序列（Ragged Tensor）、GQA（Grouped-Query Attention）以及动态 Shape 的。

</details>
