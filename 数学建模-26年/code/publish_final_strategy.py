"""Publish the frozen mixed final strategy without rerunning any model.

Inputs remain immutable: q2/q4_2 use the mother model; q3/q4_3 use the
previously executed all-update selector. Old experiments are not rewritten.
Run with the project's NumPy/Matplotlib environment, then run the companion
export_final_workbooks.mjs and the independent delivery/workbook audits.
"""
from __future__ import annotations
import csv
import datetime as dt
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import make_paper_assets as layout
import make_visual_atlas as atlas
from time_index import emergency_segments, load_time_mapping

# The original font list predates this macOS environment. Use an installed
# CJK family explicitly, so regenerated figures cannot silently show boxes.
for font in ('Arial Unicode MS', 'Songti SC', 'Noto Sans CJK SC', 'Microsoft YaHei'):
    if any(f.name == font for f in layout.font_manager.fontManager.ttflist):
        layout.plt.rcParams['font.family'] = font
        break
else:
    raise RuntimeError('A CJK font is required for the manuscript figures')

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/final_strategy'
TABLES = ROOT / 'tables'
FIG = ROOT / 'figures/results'
ATTACHMENTS = ROOT.parent / '附件/附件5'
CASES = ('q2', 'q3', 'q4_2', 'q4_3')
LABELS = {'q2': '问题2', 'q3': '问题3', 'q4_2': '问题4日前版', 'q4_3': '问题4滚动版'}
SOURCES = {c: (ROOT / f'outputs/outer_candidate_test/{c}_selected.npz'
                 if c in ('q3', 'q4_3') else ROOT / f'data/{c}_v2.npz') for c in CASES}
PRODUCTS = {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path):
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def backup(path):
    if path.exists():
        suffix = path.relative_to(ROOT) if path.is_relative_to(ROOT) else Path('attachments') / path.name
        target = OUT / 'before' / suffix
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def record(path, sources, **metadata):
    PRODUCTS[relative(path)] = {'sha256': sha(path),
        'sources_sha256': {relative(p): sha(p) for p in sources}, **metadata}


def native(value):
    if isinstance(value, dict): return {k: native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [native(v) for v in value]
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    return value


def write_json(path, value):
    path.write_text(json.dumps(native(value), ensure_ascii=False, indent=2), encoding='utf-8')


def summarize(r):
    cost = r['total_cost']
    # Exact empirical CVaR with fractional tail mass, as in the mother model.
    cvar = min(z + np.maximum(cost-z, 0).mean()/.05 for z in np.unique(cost))
    s = {key: float(r[key].sum()) for key in ('plan_cost', 'adjustment_cost', 'emergency_cost', 'total_cost')}
    s.update(days=len(cost), emergency_energy=float(r['emergency'].sum()),
        emergency_days=int((r['emergency'].sum(axis=1)>1e-6).sum()),
        emergency_intervals=int((r['emergency']>1e-6).sum()),
        emergency_segments=sum(len(emergency_segments(x)) for x in r['emergency']),
        initial_energy=float(r['energy'][0, 0]), final_energy=float(r['energy'][-1, -1]),
        daily_cost_cvar95=float(cvar), daily_cost_max=float(cost.max()),
        max_cost_date=str(r['dates'][int(cost.argmax())]),
        max_emergency_date=str(r['dates'][int(r['emergency'].sum(axis=1).argmax())]),
        max_emergency_day_kwh=float(r['emergency'].sum(axis=1).max()))
    s['monthly'] = [{ 'month': month, 'total_cost': float(cost[mask].sum()),
        'emergency_energy': float(r['emergency'][mask].sum()) }
        for month in sorted(set(str(d)[:7] for d in r['dates']))
        for mask in [np.array([str(d)[:7]==month for d in r['dates']])]]
    return s


def table(name, caption, headers, rows, sources, spec=None, note=''):
    path = TABLES / (name+'.tex')
    backup(path)
    spec = spec or 'L'+'R'*(len(headers)-1)
    lines = [r'\begin{table}[htbp]\centering\small', r'\caption{'+caption+'}',
        r'\label{tab:'+name+'}', r'\begin{tabularx}{\linewidth}{'+spec+r'}\toprule',
        ' & '.join(headers)+r'\\\midrule']
    lines += [' & '.join(map(str, row))+r'\\' for row in rows]
    lines += [r'\bottomrule\end{tabularx}']
    if note: lines += [r'\par\smallskip\parbox{\linewidth}{\footnotesize '+note+'}']
    lines += [r'\end{table}']
    path.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    record(path, sources)


def tables(results, summaries):
    for case in ('q3', 'q4_3'):
        for suffix in ('selected', 'decomposition'):
            backup(TABLES / f'{case}_{suffix}.tex')
        layout.selected_tables(case, results[case])
        for suffix in ('selected', 'decomposition'):
            record(TABLES / f'{case}_{suffix}.tex', [SOURCES[case]])
    rows = []
    for label, key, scale in [('计划费用/万元','plan_cost',1e4), ('调整费用/万元','adjustment_cost',1e4),
        ('紧急费用/万元','emergency_cost',1e4), ('实际总费用/万元','total_cost',1e4),
        ('紧急电量/kWh','emergency_energy',1), ('紧急购电天数','emergency_days',1),
        ('紧急购电连续时段数','emergency_segments',1), ('日总费用CVaR95/元','daily_cost_cvar95',1),
        ('最高日费用/元','daily_cost_max',1), ('2月1日期初库存/kWh','initial_energy',1),
        ('12月31日期末库存/kWh','final_energy',1)]:
        rows.append([label, *[str(summaries[c][key]) if key in ('emergency_days','emergency_segments')
                            else f'{summaries[c][key]/scale:.2f}' for c in CASES]])
    table('annual_summary', '最终分问配置的334天开发期闭环评价结果', ['指标', *LABELS.values()], rows,
        list(SOURCES.values()), note='问题2与问题4日前版沿用母模型；问题3与问题4滚动版在零时及日内均启用候选筛选。预测器与实时执行器沿用原冻结配置，筛选适用问题与时点依据开发期比较确定。实际费用不含候选评分的终端偏离项；CVaR95为实际日总账单的经验尾部均值。')
    timing_sources = [ROOT/'outputs/outer_timing_ablation/final_strategy.json']
    # Structure is deliberately read from executed arrays rather than typed totals.
    timing = {}
    for mode in ('main','midnight_only','intraday_only','all'):
        timing[mode] = {}
        for c in ('q3','q4_3'):
            p = (ROOT/f'data/{c}_v2.npz' if mode=='main' else SOURCES[c] if mode=='all'
                 else ROOT/f'outputs/outer_timing_ablation/{c}_{mode}.npz')
            timing_sources.append(p)
            timing[mode][c] = float(np.load(p)['total_cost'].sum())
    rows = [[label, *[f'{timing[m][c]:,.2f}' for c in ('q3','q4_3')]]
            for m,label in [('main','不启用筛选：母模型基线'), ('midnight_only','仅零时启用'),
                            ('intraday_only','仅日内启用'), ('all',r'\textbf{零时＋日内均启用：最终方案}')]]
    rows[-1][1:] = [r'\textbf{'+x+'}' for x in rows[-1][1:]]
    table('outer_timing_ablation','候选筛选启用时点对实际总费用的影响',
        ['候选筛选配置','问题3总费用/元','问题4滚动版总费用/元'], rows, timing_sources,
        note='各组均保留0、6、12、18时的合法预报更新与采购修订，仅改变是否启用候选筛选；关闭筛选时仍调用原情景优化模型。各方案从同一问题的共同初态出发，之后独立、连续推进状态，不每日重置。评价范围为2025年2月1日至12月31日，共334天。')
    rows=[]; sources=[]
    for c in ('q3','q4_3'):
        p=ROOT/f'data/{c}_v2.npz'; before=summarize(dict(np.load(p)))
        sources += [p,SOURCES[c]]
        for label,s in [('母模型',before),('最终方案',summaries[c])]:
            rows.append([LABELS[c],label,*[f'{s[key]/scale:.2f}' for key,scale in
                [('plan_cost',1e4),('adjustment_cost',1e4),('emergency_cost',1e4),('emergency_energy',1)]]])
    table('outer_screening_tradeoff','滚动采购候选筛选的费用与紧急补购权衡',
        ['问题','配置','计划费/万元','调整费/万元','紧急费/万元','紧急量/kWh'], rows,sources,'LLRRRR',
        '计划费按零时计划量全额结算，调整费始终以零时原计划为基准；紧急购电单独计费。终端偏离项不计入实际账单。两问题均以计划费用下降补偿新增调整费与紧急费。')
    return timing


def save_figure(fig, name, sources, caption='', **metadata):
    paths = [p if isinstance(p,Path) else ROOT/p for p in sources]
    for ext in ('pdf','png'):
        path = FIG/f'{name}.{ext}'; backup(path)
        fig.savefig(path,bbox_inches='tight',facecolor='white')
        record(path,paths,caption=caption,**metadata)
    layout.plt.close(fig)


def figures(results):
    plt=layout.plt; h=np.arange(144)/6; r=results['q3']
    base=dict(np.load(ROOT/'data/q3_v2.npz'))
    i=int(r['emergency'].sum(axis=1).argmax()); date=str(r['dates'][i])
    fig,axs=plt.subplots(3,1,figsize=(8.1,6.0),sharex=True,layout='constrained')
    axs[0].bar(h,r['adjusted'][i]-r['plan'][i],width=1/6,color=layout.COLORS[1]);axs[0].set_ylabel('常规上调量 / kWh')
    axs[1].plot(np.arange(145)/6,base['energy'][i],label='母模型实际库存',color=layout.COLORS[0],ls='--')
    axs[1].plot(np.arange(145)/6,r['energy'][i],label='最终方案实际库存',color=layout.COLORS[2]);axs[1].set_ylabel('储电量 / kWh');axs[1].legend()
    axs[2].bar(h,r['emergency'][i],width=1/6,color='#B45454');axs[2].set_ylabel('紧急购电量 / kWh');axs[2].set_xlabel('物理时刻 / h')
    for ax in axs:
        for t in (6,12,18):ax.axvline(t,color='#888888',ls=':',lw=.7)
        ax.grid(axis='y',alpha=.18);ax.set_xlim(0,24);ax.set_xticks(range(0,25,2))
    axs[0].set_title('问题3最终方案：'+date+'（紧急量最大日）',loc='left')
    save_figure(fig,'v2_control_difference',[SOURCES['q3'],ROOT/'data/q3_v2.npz'],
        '最终方案上调量和紧急购电量；中图两条均为各自连续运行的实际库存，无规划参考轨迹。',selected_date=date)
    fig,axs=plt.subplots(2,2,figsize=(8.1,4.8),layout='constrained')
    for ax,date in zip(axs.flat,layout.SELECTED):
        j=list(r['dates']).index(date);delta=r['adjusted'][j]-r['plan'][j]
        ax.bar(h,delta,width=1/6,color=layout.COLORS[1])
        if np.max(np.abs(delta)) < 1e-7:
            ax.set_ylim(0,1);ax.set_yticks([0,0.5,1])
            ax.text(.5,.5,'全天无常规上调',ha='center',va='center',transform=ax.transAxes,color=atlas.MUTED)
        ax.set_title(date,loc='left');ax.set_ylabel('上调量 / kWh');ax.set_xlabel('时刻 / h');ax.set_xlim(0,24);ax.grid(axis='y',alpha=.18)
    save_figure(fig,'v2_adjustments',[SOURCES['q3']], '最终方案四个指定日期的实际常规上调量。')
    fig,axs=plt.subplots(2,1,figsize=(8.1,5.2),sharex=True,layout='constrained')
    for j,(c,x) in enumerate(results.items()):
        month=np.array([int(d[5:7]) for d in x['dates']])
        axs[0].plot(range(2,13),[x['total_cost'][month==m].sum()/1e4 for m in range(2,13)],'o-',ms=3,label=LABELS[c],color=layout.COLORS[j])
        axs[1].plot(range(2,13),[x['emergency'][month==m].sum()/1e3 for m in range(2,13)],'o-',ms=3,color=layout.COLORS[j])
    axs[0].set_ylabel('月费用 / 万元');axs[0].legend(ncol=2)
    axs[1].set_ylabel('紧急量 / 千kWh');axs[1].set_xlabel('月份');axs[1].set_xticks(range(2,13))
    for ax in axs:ax.grid(axis='y',alpha=.18)
    save_figure(fig,'v2_monthly',list(SOURCES.values()), '四问最终分问配置的月费用和紧急购电量。')
    # Retain the established waterfall layout, with final arrays and new provenance.
    a,b=results['q4_2'],results['q4_3'];keys=('plan_cost','adjustment_cost','emergency_cost')
    delta=np.array([float(b[k].sum()-a[k].sum()) for k in keys]);first=float(a['total_cost'].sum())
    atlas.save=save_figure
    atlas.waterfall('q4_cost_waterfall',first,delta,
        ['日前方案：总费','零时计划费变化','计划调整费变化','紧急购电费变化','最终滚动方案：总费'],
        '问题4：日前至最终滚动方案的账单分解',[SOURCES['q4_2'],SOURCES['q4_3']],
        '最终分问配置的实际账单恒等分解，不将差值全归因于某个单独模块。',
        conservation_residual_yuan=float(first+delta.sum()-b['total_cost'].sum()))


def main():
    OUT.mkdir(parents=True,exist_ok=True);load_time_mapping()
    immutable=[*SOURCES.values(),*[ROOT/f'data/{c}_v2.npz' for c in CASES],
        ROOT/'data/time_mapping.json',ROOT/'outputs/outer_timing_ablation/final_strategy.json',
        ROOT/'code/make_paper_assets.py',ROOT/'code/make_visual_atlas.py',ROOT/'code/time_index.py']
    before={relative(p):sha(p) for p in set(immutable)}
    for name in ('result3.xlsx','result4-3.xlsx'):backup(ATTACHMENTS/name)
    results={c:dict(np.load(p,allow_pickle=False)) for c,p in SOURCES.items()}
    summaries={c:summarize(r) for c,r in results.items()}
    output={ 'version':'final_mixed_strategy_after_development_ablation',
        'q1':json.loads((ROOT/'data/q1.json').read_text(encoding='utf-8')), 'cases':results,
        'strategy':{c:('all_update_screening' if c in ('q3','q4_3') else 'mother_model') for c in CASES},
        'source_arrays':{c:relative(p) for c,p in SOURCES.items()} }
    path=OUT/'results_export.json'
    path.write_text(json.dumps(native(output),ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    record(path,list(SOURCES.values())+[ROOT/'data/q1.json'])
    write_json(OUT/'summary.json',summaries);record(OUT/'summary.json',list(SOURCES.values()))
    with (OUT/'daily_summary.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['case','date','plan_kwh','adjusted_kwh','emergency_kwh',
            'plan_cost_yuan','adjustment_cost_yuan','emergency_cost_yuan','total_cost_yuan','energy_start_kwh','energy_end_kwh'])
        for c,r in results.items():
            for i,date in enumerate(r['dates']):
                writer.writerow([c,date,*[r[k][i].sum() for k in ('plan','adjusted','emergency')],
                    *[r[k][i] for k in ('plan_cost','adjustment_cost','emergency_cost','total_cost')],r['energy'][i,0],r['energy'][i,-1]])
    record(OUT/'daily_summary.csv',list(SOURCES.values()))
    timing=tables(results,summaries);figures(results)
    assert before=={relative(p):sha(p) for p in set(immutable)}
    write_json(OUT/'asset_manifest.json',{'generated_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'generator_sha256':sha(Path(__file__)), 'inputs_sha256':before,
        'scope':'Publish existing executions only; original arrays, experiments and excluded mother-model diagnostics remain unchanged.',
        'products':PRODUCTS,'timing_costs_yuan':timing,
        'unchanged_diagnostic_figures':['q3_feedback_waterfall','q3_execution_band','q4_correlation_heatmap'],
        'reference_soc_note':'Final selected arrays have no reference SOC trace. The updated q3 trajectory compares actual SOC with mother-model actual SOC.'})
    print(json.dumps(summaries,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
