"""Build a complete verbatim source appendix without running model programs.

Run from any directory: python outputs/code_appendix/build_code_appendix.py
Then compile main.tex with XeLaTeX. Existing source files are read verbatim.
"""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

# All current code files, followed by required frozen and delivery paths.
SOURCES = [
    ('code/requirements.txt', '原交付环境依赖清单'),
    ('code/data_forecast.py', '附件读取与因果预测'),
    ('code/time_index.py', '物理时段与附件索引映射'),
    ('code/optimize_dispatch.py', '确定性模型与原版场景调度'),
    ('code/dispatch_v2.py', '统一母模型、控制器与结算'),
    ('code/calibrate_v2.py', '一月配置校准与冻结'),
    ('code/run_main_v2.py', '四组母模型计算入口'),
    ('code/outer_candidate_selector.py', '候选生成与执行回放评分'),
    ('code/run_outer_candidate_experiment.py', '全时点筛选实验'),
    ('code/outer_timing_policy.py', '筛选启用时点策略'),
    ('code/run_outer_timing_ablation.py', '筛选时点消融与最终选型'),
    ('code/run_experiments.py', '原版模型对照实验'),
    ('code/experiments_v2.py', '母模型风险、备用与反馈对照'),
    ('code/diagnose_execution_gap.py', '冻结原版规划与执行诊断'),
    ('code/diagnose_inner_v2.py', '母模型执行器及续接诊断'),
    ('code/scenario_convergence.py', '历史场景压缩与收敛记录'),
    ('code/make_paper_assets.py', '原版结果图表生成'),
    ('code/make_paper_assets_v2.py', '母模型结果图表生成'),
    ('code/make_question_tables.py', '分问指定结果表生成'),
    ('code/make_visual_atlas.py', '调度与诊断图生成'),
    ('code/publish_final_strategy.py', '最终方案指标与图表发布'),
    ('code/make_compact_paper_tables.py', '紧凑指定日期表生成'),
    ('code/export_workbooks.mjs', '原版与母模型工作簿填写'),
    ('code/export_final_workbooks.mjs', '最终滚动工作簿填写'),
    ('code/verify_workbooks.py', '原版工作簿检查'),
    ('code/verify_final_workbooks.py', '最终工作簿检查'),
    ('code/check_dispatch_v2.py', '母模型数值与信息检查'),
    ('code/check_outer_candidate_selector.py', '候选回放单项检查'),
    ('code/check_outer_timing_policy.py', '筛选时点策略检查'),
    ('code/check_outer_timing_ablation.py', '筛选时点消融检查'),
    ('code/check_final_strategy_delivery.py', '最终执行与交付检查'),
    ('code/check_manuscript_sources.py', '原分问稿来源与结构检查'),
    ('code/run_all.ps1', '原版完整运行入口'),
    ('code/run_revision.ps1', '母模型修订复算入口'),
    ('code/package_revision.py', '原分问稿打包工具'),
    ('versions/v1_before_settlement_revision/code/optimize_dispatch.py',
     '原版诊断动态加载的冻结实现'),
    ('compile.sh', 'XeLaTeX 编译入口'),
    ('outputs/final_strategy/runtime/export_final_workbooks.mjs',
     '工作簿导出的执行目录副本'),
    ('outputs/logic_formula_review/verify_revision.py', '公式与版式专项审计'),
    ('outputs/code_appendix/build_code_appendix.py', '本代码附录生成器'),
]


def main():
    paths = [p for p, _ in SOURCES]
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate source path')
    actual = {p.relative_to(ROOT).as_posix() for p in (ROOT / 'code').iterdir()
              if p.is_file()}
    listed = {p for p in paths if p.startswith('code/')}
    if actual != listed:
        raise ValueError(f'Code coverage mismatch: {actual ^ listed}')
    records = []
    for i, (name, purpose) in enumerate(SOURCES, 1):
        blob = (ROOT / name).read_bytes()
        source = blob.decode('utf-8')
        records.append(dict(number=i, path=name, purpose=purpose,
                            lines=len(source.splitlines()), bytes=len(blob),
                            sha256=hashlib.sha256(blob).hexdigest(),
                            label=f'code:source:{i:02d}'))
    total = sum(x['lines'] for x in records)
    tex = [r'\clearpage', r'\section{完整程序代码}', r'\label{app:complete_code}',
           f'本附录逐文件收录{len(records)}份程序与依赖文件，共{total}行，覆盖数据读取、预测、优化、候选筛选、对照实验、图表、工作簿及检查。代码由源文件全文读入，保留注释与原始行号；长行折行时沿用同一行号。',
           r'\subsection*{运行环境与复现顺序}',
           r'Python 数值程序依赖 NumPy、SciPy、Matplotlib 和 OpenPyXL，LP 由 SciPy 的 HiGHS 接口求解。\path{code/requirements.txt}保留原交付环境清单；最终候选及消融运行记录使用 Python 3.11.15、NumPy 2.4.6、SciPy 1.17.1。Node.js 工作簿导出另依赖\path{@oai/artifact-tool} 2.8.59；运行执行目录副本时，设置\path{CUMCM_PROJECT_ROOT}为项目根目录，并确保依赖可被 Node.js 解析。PowerShell 入口用于原运行环境；论文使用 XeLaTeX 编译，代码采用 Menlo 与宋体字体。',
           r'复现按“附件与时间映射—一月配置冻结—母模型与外层筛选—结果发布及核对”推进。\path{run_all.ps1}及\path{run_revision.ps1}分别对应原版和母模型；最终滚动方案另由候选与时点实验入口生成，之后执行\path{publish_final_strategy.py}、工作簿导出与核对。\path{make_compact_paper_tables.py}恢复紧凑表排布。冻结配置、实测附件、空白模板及执行数组按\ref{tab:files}所述目录组织。',
           r'原版诊断按保存路径动态加载冻结实现，因此一并列出该文件；工作簿执行目录副本也完整列出。旧打包及诊断脚本保留当时版本的说明，其状态文字不作为当前最终方案的结果。各源码按下列相对路径保存，保持目录结构；第三方库按依赖安装，历史备份中的重复文件不再逐份列入。',
           r'\subsection*{文件与页码索引}',
           r'\begingroup\small\setlength{\tabcolsep}{4pt}',
           r'\begin{longtable}{r p{0.44\linewidth} p{0.30\linewidth} r r}',
           r'\toprule 编号 & 文件 & 用途 & 行数 & 页码\\\midrule\endfirsthead',
           r'\toprule 编号 & 文件 & 用途 & 行数 & 页码\\\midrule\endhead',
           r'\midrule\multicolumn{5}{r}{续下页}\\\endfoot',
           r'\bottomrule\endlastfoot']
    for x in records:
        tex.append(f"{x['number']} & \\path{{{x['path']}}} & {x['purpose']} & {x['lines']} & \\pageref{{{x['label']}}}\\\\")
    tex += [r'\end{longtable}\endgroup', r'\clearpage',
            r'\begingroup', r'\newfontfamily\appendixcodefont{Menlo}[NFSSFamily=appendixcode]',
            r'\setCJKfamilyfont{appendixcjk}{Songti SC}',
            r'\fvset{fontfamily=appendixcode,fontsize=\fontsize{8.5}{10.5}\selectfont,baselinestretch=1,formatcom=\CJKfamily{appendixcjk},numbers=left,numbersep=6pt,xleftmargin=2.8em,frame=lines,framesep=5pt,rulecolor=\color{black!25},breaklines=true,breakanywhere=true,breakautoindent=true,breakindent=1em,breaksymbolleft=\tiny\ensuremath{\hookrightarrow},tabsize=4,showspaces=false,showtabs=false}',
            r'\renewcommand{\theFancyVerbLine}{\scriptsize\color{black!50}\arabic{FancyVerbLine}}']
    for x in records:
        tex += [r'\Needspace{7\baselineskip}',
                f"\\subsection{{{x['purpose']}}}", f"\\label{{{x['label']}}}",
                f"{{\\small\\noindent\\path{{{x['path']}}}\\quad（{x['lines']}行）\\par}}",
                f"\\VerbatimInput{{{x['path']}}}",
                f"\\label{{{x['label']}:end}}"]
    tex += [r'\endgroup', '']
    target = ROOT / 'sections' / 'appendix_complete_code.tex'
    target.write_text('\n\n'.join(tex), encoding='utf-8')
    manifest = dict(files=len(records), total_lines=total,
                    code_directory_complete=True, code_directory_files=len(actual),
                    generated_tex=target.relative_to(ROOT).as_posix(), sources=records)
    (OUT / 'source_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: manifest[k] for k in ('files', 'total_lines', 'code_directory_files')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
