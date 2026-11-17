# 建模框架、汇总框与页面密度修订

本轮仅整理已有模型的论文表达与排版。当前交付为项目根目录 `main.tex`、`main.pdf`，PDF 共 38 页；修订前为 49 页，副本保存在 `before/`。

## 已完成修改

- 问题重述后加入原生 TikZ 建模框架图，串联共同物理与结算基础、四问方法、固定执行、开发期对照及最终分问配置。
- 四问各增加一个紧凑模型汇总框，统一列出输入、决策、目标、约束与输出。
- 精简重复解释，共同评价范围、回放近似、比较口径及统计解释集中于 8.2；单位换算、供需不等式、计划量全额结算等必要模型条件保留在原处。
- 四组指定日期表改为日期并列，完整保留原有 392 个数值单元及全部紧急区间；额外列出四日紧急量总计。问题二费用分解表移入正文，与其指定日期结果同页。
- 调整浮动体间距、图幅、参考文献行距与附录文件表，合并重复时间映射说明，避免附录代码跨页。

## 复现

仅编译当前稿：

```sh
latexmk -xelatex -synctex=1 -interaction=nonstopmode -file-line-error -halt-on-error main.tex
```

若重新运行 `code/publish_final_strategy.py` 生成数表，应随后运行以下脚本恢复紧凑表格布局，再编译论文：

```sh
python code/make_compact_paper_tables.py
latexmk -xelatex -synctex=1 -interaction=nonstopmode -file-line-error -halt-on-error main.tex
```

该排版脚本读取已冻结执行数组，不运行年度模型；表格中的结果保留原精度计算后统一显示两位小数。脚本使用本目录 `before/tables/` 的原表进行数值保留核对。

## 核查证据

- `source_integrity.json`：四问 45 个独立数学环境、关键数值、四个汇总框、问题二费用分解表唯一引用，以及 106 份受保护代码、数据、原图和附件的哈希核查。
- `compact_tables.json`：四份紧凑表与输入数组的指纹，原 392 个数值单元逐值保留检查。
- `document_checks.json`：主文件输入链、标签、引用、插图、编译问题及当前源文件与 PDF 指纹。
- `verification.json`：本次交付汇总与视觉检查范围。
- `visual/`：最终 PDF 全文缩略图及重点页清晰预览；`compile.log` 保存本次编译输出。

本次未更换母模型、预测器、实时执行器、最终策略或工作簿。`outputs/final_strategy/` 中旧发布与审计记录保留为当次数值发布证据，其旧表格和旧 PDF 指纹对应历史版式；当前排版指纹以本目录的核查文件为准。
