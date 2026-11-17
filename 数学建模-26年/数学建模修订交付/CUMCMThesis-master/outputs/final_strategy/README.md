# 最终分问配置的论文与附件发布

本目录只发布已经完成的执行数组，不重新求解模型。第二问、第四问日前版保持母模型；第三问、第四问滚动版采用零时及日内均筛选。候选筛选的适用问题与启用时点通过开发期回测确定，报告范围为2025年2月1日至12月31日334天，不是未接触数据上的独立测试。

| 问题 | 执行数组 | 总费用/元 |
|---|---|---:|
| 第二问 | `data/q2_v2.npz` | 14,012,807.66 |
| 第三问 | `outputs/outer_candidate_test/q3_selected.npz` | 13,871,471.44 |
| 第四问日前版 | `data/q4_2_v2.npz` | 14,794,124.29 |
| 第四问滚动版 | `outputs/outer_candidate_test/q4_3_selected.npz` | 14,670,237.87 |

## 文件与统计口径

- `results_export.json`：上述四组完整最终执行数组与第一问结果，供附件填写；旧 `data/results_export.json` 保留为母模型证据。
- `summary.json`、`daily_summary.csv`：直接从执行数组重算的总量、日账单CVaR95、最高日费用、紧急天数、连续段数及逐日账单。
- `asset_manifest.json`：新生成正文数表与4幅图的哈希、对应源数组哈希；不改写旧实验溯源记录。
- `workbook_previews/workbook_verification.json`：五份附件逐数核对结果，涵盖最终两份滚动结果与三份保留附件。
- `workbook_previews/workbook_file_manifest.csv`：附件的真实路径、哈希、数值数目、数据源哈希。
- `before/`：本次覆盖前的主结果表、图与两份滚动工作簿。

功率按10分钟积分为kWh。计划費按零时原计划量全额结算，调整费相对零时原计划计算，紧急费另计；候选评分的终端偏离项不计入账单。CVaR95沿用母模型的精确经验公式（允许分数尾部质量），并非简单取最高若干日费用均值。

`emergency_intervals` 为发生紧急补购的10分钟时段数，`emergency_segments` 为按原附件阈值合并的连续时段数，两者不同。最终第三问分别为424与204，第四问滚动版分别为418与193。

最终执行数组没有规划参考SOC轨迹。因此总表不列参考SOC RMSE；`v2_control_difference` 的中图比较母模型与最终方案各自实际库存，不使用旧参考轨迹。旧信息来源消融、执行器对照和回放诊断保持母模型配置；价格—净负荷相关性图描述冻结预测误差，未因采购筛选而改变。

## 复现发布

从项目根目录运行已有科学绘图环境中的 `python3 code/publish_final_strategy.py`。该脚本备份将覆盖的资产，重新生成汇总、7份数表与4幅图的PDF/PNG；不修改任何旧执行数组、实验文件或旧数据导出。

工作簿使用系统提供的 `@oai/artifact-tool`。将 `code/export_final_workbooks.mjs` 放入本目录的运行子目录，在其旁将 `node_modules` 链接至工作环境依赖目录，然后通过环境变量 `CUMCM_PROJECT_ROOT` 指定项目根目录，运行该脚本。它只更新项目父目录 `附件/附件5/result3.xlsx` 与 `result4-3.xlsx`，沿用原附件模板、时间标签、工作表顺序和格式，重新计算并导出公式缓存。

本机已经执行的命令为：

```sh
python3 code/publish_final_strategy.py
CUMCM_PROJECT_ROOT="$PWD" /Users/mingze/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node outputs/final_strategy/runtime/export_final_workbooks.mjs
/Users/mingze/miniconda3/envs/deeptutor/bin/python code/verify_final_workbooks.py
```

正文由 `latexmk -xelatex main.tex` 编译。原图表生成器与原附件导出器仍服务于母模型历史复现；不要用它们覆盖最终混合配置的交付文件。
