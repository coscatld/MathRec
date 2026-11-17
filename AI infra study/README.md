# AI Infra Study

学习 GPU 算子与大模型推理工程的个人仓库。主线是 **Triton → CUDA → GEMM / FlashAttention → KV Cache 自定义算子**，另有一条 Node/TS 推理服务工程支线。

- 学习计划与排期：[PLAN.md](PLAN.md)
- 现状一句话：已有 10 课讲义，**所有 Triton kernel 都还没在 GPU 上跑过**（见下方「验证状态」），所以计划的第一步是补 GPU 实测。

## 目录结构

```text
AI infra study/
├── README.md     本文件：目录、课程总表、运行方式
├── PLAN.md       学习计划（四阶段路线 + 排期 + 验收标准）
├── kernel/       主线：Triton / CUDA 算子课程
├── serving/      支线：Node/TS 推理服务工程课程
├── practice/     （待创建）你自己手敲的实现，约定见 PLAN.md §3
└── _archive/     原始下载的 zip；已全部解压，确认无误后可整个删除
```

每课一个自包含目录 `L<课号>_<主题>/`：`lesson.md`（讲义）+ 代码 + 测试 + `README.md` + `VALIDATION.md`。

## 课程总表

| 课 | 日期 | 线 | 主题 | Triton kernel | 验证状态（生成讲义时的环境，非你的机器） |
|---|---|---|---|---|---|
| [L01](kernel/L01_residual_add/lesson.md) | 10-30 | kernel | 残差加法，写出第一个 Triton 算子 | ✅ | 仅标准库索引模拟（6 个长度）；torch / GPU 未执行 |
| [L02](kernel/L02_row_sum/lesson.md) | 10-31 | kernel | 从逐元素走向行求和规约 | ✅ | 14 个测试：8 通过 / 6 跳过 |
| [L03](kernel/L03_rmsnorm/lesson.md) | 11-01 | kernel | RMSNorm：把行规约变成模型组件 | ✅（仅前向） | 16 项：10 通过 / 6 跳过 |
| [L04](serving/L04_model_contract/lesson.md) | 11-02 | serving | 接口契约，搭起可替换的模型后端 | – | Node 24：13/13 通过 |
| [L05](serving/L05_timeout_cancellation/lesson.md) | 11-03 | serving | 超时不等于取消 | – | 13/13 通过 |
| [L06](kernel/L06_softmax/lesson.md) | 11-04 | kernel | 稳定的行 Softmax 与融合规约 | ✅ | 标准库 12 项通过；依赖测试 6 项跳过 |
| [L07](serving/L07_bounded_concurrency/lesson.md) | 11-05 | serving | 有界并发与背压 | – | 22 项通过 + 200 种子随机压力测试 |
| [L08](kernel/L08_online_softmax/lesson.md) | 11-06 | kernel | log-sum-exp 与分块 Online Softmax | ✅（可选） | 4 通过 / 3 跳过；独立审计 6 通过 |
| [L09](kernel/L09_attention/lesson.md) | 11-07 | kernel | 把 Attention 写完整：QKᵀ、mask、softmax、V | ❌ 纯 Python 参考 | 17 通过 / 2 跳过 |
| [L10](serving/L10_latency_observability/lesson.md) | 11-08 | serving | 调用超时 ≠ 模型执行结束：生命周期与指标 | – | 18 项通过 |

> 课号沿用原系列的顺序。kernel 与 serving 两条线交替出现，所以各自的课号不连续。

## 验证状态：先读这一段

- 所有 Triton kernel 的生成环境里都**没有 torch / triton**：GPU 路径要么 SKIP，要么根本没执行，**没有任何 GPU 正确性或性能数字**。表里的「通过」只指 CPU 侧的标准库模拟 / 参考实现。
- L09 不含 Triton/CUDA kernel，是 Python 参考实现（FlashAttention 风格的流式更新，CPU 版）。
- 结论：在你自己的 GPU 上跑通之前，把这些 kernel 当作「待验证的参考」，不要当作可信基线。

## 运行方式

在各课目录内运行（命令取自该课的 README / VALIDATION）：

```bash
# kernel/（Python；没有 torch / triton 时，相应用例会自动 SKIP）
cd kernel/L01_residual_add     && python lesson_01_residual_add.py     # 加 --gpu / --bench 需要 CUDA 环境
cd kernel/L02_row_sum          && python -m unittest test_stdlib test_torch -v
cd kernel/L03_rmsnorm          && python -m unittest test_stdlib test_torch -v
cd kernel/L06_softmax          && python -m unittest -v test_stdlib test_torch
cd kernel/L08_online_softmax   && python -m unittest discover -s tests -v && python audit_checks.py
cd kernel/L09_attention        && python -m unittest -v test_attention audit_checks

# serving/（Node，无 npm 依赖）
cd serving/L04_model_contract          && node --test model.test.ts
cd serving/L05_timeout_cancellation    && node --test timeout.test.ts
cd serving/L07_bounded_concurrency     && node --test executor.test.ts
cd serving/L10_latency_observability   && node --test metrics.test.js
```

环境说明：

- kernel 的 GPU 路径基线是 Linux x86_64 · Python 3.10–3.12 · torch 2.5.1（cu121）· triton 3.1.0。Triton 不支持 macOS，GPU 路径需要 Linux + NVIDIA GPU。
- 本机（Apple M4，2026-11-15 检测）：conda 环境 `study` 里有 torch（CPU / MPS，没有 CUDA 和 triton），可以跑 `test_torch` 里的 CPU 对照；GPU 用例会 SKIP。
- serving 各课的 `package.json` 声明 Node 24.x（生成时在 v24.22.1 实测）。本机 nvm 目前只装了 v22.22.2，建议先 `nvm install 24 && nvm use 24` 再跑。

## 新讲义怎么归档

1. 新下载的 zip 先解压到一个**新建的空目录**里检查（附件是外部文件，不要直接在下载目录里运行）。
2. 按主题放进 `kernel/` 或 `serving/`，目录名 `L<两位课号>_<主题英文短名>/`；讲义改名为 `lesson.md`，并同步改掉 README 里对讲义文件名的引用。
3. zip 里已经带了讲义，就**不要再单独留一份 `.md`**（这个目录之前的重复就是这么来的）。
4. 在上面的课程总表里补一行。

## 整理记录（2026-11-15）

- 4 篇 serving 课（11-02 / 11-03 / 11-05 / 11-08）的讲义被命名为 `Triton_CUDA_每日学习_*.md`，与内容不符；现按内容分到 `serving/`。
- 9 份讲义与对应 zip 内的同名文件逐字节相同（sha256 比对），只保留一份；第 1 课的 `.py` 与讲义中的代码块逐字节一致。
- 9 个 zip 解压为每课一个目录；原 zip 原封不动移到 `_archive/original-downloads/`。
- 5 个 README 里对旧讲义文件名的引用已改成 `lesson.md`（L02 / L03 / L04 / L07 / L08）。L02、L03 的 `lesson.md` 开头仍提到原 zip 名，那是讲义原文，未改。
- 没有修改任何代码文件的内容。
