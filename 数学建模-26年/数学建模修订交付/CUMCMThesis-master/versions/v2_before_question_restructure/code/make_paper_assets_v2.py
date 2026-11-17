"""Generate revision tables/figures directly from frozen execution arrays. No LaTeX."""
import csv
import json
from pathlib import Path
import numpy as np
import make_paper_assets as old
from time_index import load_time_mapping, physical_slice, emergency_segments

ROOT=Path(__file__).resolve().parents[1]
DATA,TABLES,FIG=ROOT/'data',ROOT/'tables',ROOT/'figures/results'
CASES=old.LABELS
f=old.fmt


def table(name,caption,headers,rows,spec=None,note=''):
    spec=spec or 'L'+'R'*(len(headers)-1)
    out=[r'\begin{table}[htbp]\centering\small',r'\caption{'+caption+'}',
         r'\label{tab:'+name+r'}',r'\begin{tabularx}{\linewidth}{'+spec+r'}\toprule',
         ' & '.join(headers)+r'\\\midrule']
    out+=[' & '.join(str(c) for c in row)+r'\\' for row in rows]
    out += [r'\bottomrule\end{tabularx}']
    if note:out += [r'\par\smallskip\parbox{\linewidth}{\footnotesize '+note+'}']
    out += [r'\end{table}']
    (TABLES/(name+'.tex')).write_text('\n'.join(out),encoding='utf-8')


def selection_tables():
    sel=json.loads((DATA/'selection_v2.json').read_text(encoding='utf-8'))
    names={'weekday_2':'同星期2日','weekday_4':'同星期4日','mean_3':'最近3日',
           'mean_7':'最近7日','mean_14':'最近14日','last_week':'上周同日'}
    rows=[];control=[];metrics=[]
    for case,entry in sel['cases'].items():
        fc=entry['forecasts'];s=entry['validation_winner'];rolling=case in ('q3','q4_3')
        rows.append([CASES[case],names[fc['load_model']],
            '附件3发布值' if rolling else names[fc['pv_model']],
            names[fc['price_model']] if case.startswith('q4') else '已知电价',
            f(s['total_cost']),f(s['final_energy'])])
        for controller,cross,label in [('greedy',False,'规则执行'),('lp',False,'储能LP'),('lp',True,'储能LP+跨日')]:
            matches=[v for v in entry['validation_candidates'] if v['forecasts']==fc
                     and v['config']['controller']==controller and v['config']['cross_day']==cross]
            if matches:
                s=matches[0]['summary'];control.append([CASES[case],label,f(s['total_cost']),f(s['emergency_energy']),f(s['final_energy'])])
        d=dict(np.load(DATA/f'prepared_v2_{case}.npz'))
        variables=[('负荷/kW','load','forecast_load')]
        variables.append(('光伏/kW','pv','pv_issue_forecast' if rolling else 'forecast_pv_history'))
        if case.startswith('q4'):variables.append(('电价/(元/kWh)','price','forecast_price'))
        for label,actual,pred in variables:
            a=d[actual];b=d[pred][:,0] if pred=='pv_issue_forecast' else d[pred]
            vals=[]
            for sl in (slice(14,31),slice(31,365)):
                e=a[sl]-b[sl];fmt=(lambda x:f'{x:.5f}') if actual=='price' else f
                vals += [fmt(np.mean(np.abs(e))),fmt(np.sqrt(np.mean(e**2)))]
            metrics.append([CASES[case],label,*vals])
    table('v2_forecast_selection','一月验证冻结的预测配置与执行结果',
          ['问题','负荷','光伏','电价','验证费/元','末库存/kWh'],rows,'LLLLRR',
          '验证期为1月15日至31日；所有主案选择规则执行、28个历史场景，额外CVaR与备用均关闭。预测器先以统一储能LP进行逐坐标费用比较，再比较执行器，属于有限候选搜索。')
    table('v2_forecast_metrics','已选预测器的误差指标（光伏发布值取零时版本）',
          ['问题','变量','一月MAE','一月RMSE','评价期MAE','评价期RMSE'],metrics,'LLRRRR',
          '一月指1月15日至31日；评价期指2月至12月。评价期误差仅作结果描述，不参与选型；价格列与功率列单位不同。')
    table('v2_controller_validation','相同已选预测配置下的执行器一月验证',
          ['问题','执行器','实际费用/元','紧急量/kWh','末库存/kWh'],control,'LLRRR',
          '同一问题内使用相同1月15日期初库存，之后独立因果运行。费用不含终端罚项；末库存并不强制相等。')


def primary_tables():
    results={c:dict(np.load(DATA/f'{c}_v2.npz')) for c in CASES}
    q1=json.loads((DATA/'q1.json').read_text(encoding='utf-8'))
    old.write('q1_results.tex',r'\begin{table}[htbp]\centering\small\caption{问题1指定时段与全天购电结果（电量单位：kWh）}\label{tab:q1-purchase}'+
       old.purchase_block(q1['plan'],sum(q1['plan']),q1['cost'])+r'\end{table}'+'\n'+
       r'\begin{table}[htbp]\centering\small\caption{问题1储能充放电及首末电量（单位：kWh）}\label{tab:q1-battery}'+
       old.battery_block(np.array(q1['charge']),np.array(q1['discharge']),q1['energy'])+r'\end{table}')
    for c,r in results.items():old.selected_tables(c,r)
    summaries={c:json.loads((DATA/f'{c}_v2_summary.json').read_text(encoding='utf-8')) for c in CASES}
    rows=[]
    for label,key,scale in [('计划费用/万元','plan_cost',1e4),('调整费用/万元','adjustment_cost',1e4),
       ('紧急费用/万元','emergency_cost',1e4),('实际总费用/万元','total_cost',1e4),
       ('紧急电量/kWh','emergency_energy',1),('紧急购电天数','emergency_days',1),
       ('日总费用CVaR95/元','daily_cost_cvar95',1),('最高日费用/元','daily_cost_max',1),
       ('2月1日期初库存/kWh','initial_energy',1),('12月31日期末库存/kWh','final_energy',1),
       ('参考与实际SOC的RMSE/kWh','reference_soc_rmse',1)]:
        rows.append([label,*[f(summaries[c][key]/scale) for c in CASES]])
    table('annual_summary','冻结经济主案的2月至12月334天结果',['指标',*CASES.values()],rows,
          note='同一问题内各对照使用相同评价期初态；不同问题延续各自已冻结的一月运行状态。CVaR95描述实际日总账单，并非模型的场景紧急费用CVaR90。')
    with (DATA/'daily_summary.csv').open('w',encoding='utf-8-sig',newline='') as out:
        w=csv.writer(out);w.writerow(['case','date','plan_kwh','adjusted_kwh','emergency_kwh','plan_cost_yuan','adjustment_cost_yuan','emergency_cost_yuan','total_cost_yuan','energy_start_kwh','energy_end_kwh'])
        for c,r in results.items():
            for i,date in enumerate(r['dates']):
                w.writerow([c,date,*[r[k][i].sum() for k in ['plan','adjusted','emergency']],
                            *[r[k][i] for k in ['plan_cost','adjustment_cost','emergency_cost','total_cost']],r['energy'][i,0],r['energy'][i,-1]])
    return results,summaries


def gap_table():
    report=json.loads((DATA/'execution_gap_v1.json').read_text(encoding='utf-8'))
    rows=[]
    for c in CASES:
        s=report['summary'][c]['reporting_feb_to_dec']['all_updates'];g=s['delta_exec_yuan']
        rows.append([CASES[c],str(g['count']),f(g['mean']),f(100*g['negative_fraction'])+r'\%',
                     f(s['soc_weighted_rmse_kwh']['mean'])])
    table('v2_execution_gap','原版同计划、同初态、同场景的规划与执行偏差',
          ['问题','决策点数',r'平均$\Delta_{\rm exec}$/元','负偏差比例','SOC均方根偏差/kWh'],rows,
          note='每一决策点先按场景概率计算费用差与SOC均方根偏差，再对决策点取均值；滚动剩余时域相互重叠，不将这些差求和解释为年度费用。全部费用仅计实际结算项目。')
    path=DATA/'inner_gap_v2.json'
    if path.exists():
        a=json.loads(path.read_text(encoding='utf-8'))['aggregate']
        rows=[]
        for label,key in [('有符号期望费用偏差/元','signed_mean_gap_yuan'),
                          ('场景费用绝对偏差的期望/元','mean_absolute_scenario_gap_yuan'),
                          ('SOC均方根偏差/kWh','soc_rmse_kwh')]:
            rows.append([label,*[f(a[f'decision_mean_{c}_{key}']) for c in ['greedy','lp']]])
        table('v2_inner_gap','修订主计划下两种执行器的同场景回放诊断',
              ['指标','规则执行','内层LP'],rows,
              note='四个指定日、四个时点共16个决策点，每点28场景。先计算场景概率加权指标，再对决策点等权平均；两种执行器固定相同购电合同和初态，仅逐步揭示当前场景值。')


def experiment_tables():
    path=DATA/'experiments_v2.json'
    if not path.exists():return
    ex=json.loads(path.read_text(encoding='utf-8'))['experiments']
    specs={
       'v2_scenarios':('历史场景保留与缩减的经济对照',[
          ('q2_scenarios_28','问题2：28场景'),('q2_scenarios_12','问题2：12场景'),
          ('q3_scenarios_28','问题3：28场景'),('q3_scenarios_12','问题3：12场景')]),
       'v2_risk_modules':('问题2风险与备用的完整对照',[
          ('q2_expectation_only','期望费用'),('q2_risk_fixed_only','固定风险'),
          ('q2_risk_adaptive_only','自适应风险'),('q2_reserve_fixed_only','固定备用'),
          ('q2_reserve_adaptive_only','动态备用'),('q2_risk_reserve_fixed','固定风险+备用'),
          ('q2_risk_reserve_adaptive','自适应风险+备用'),('q2_risk_reserve_adaptive_12','自适应联合，12场景')]),
       'v2_feedback':('问题3的状态、光伏和负荷反馈分解',[
          ('q3_midnight_only','仅零时外层计划'),('q3_feedback_soc_only','仅更新SOC并重优化'),('q3_feedback_pv_only','再加入新光伏预报'),('q3_feedback_full','再加入负荷偏差反馈')]),
       'v2_controllers':('问题3执行控制与跨日续接对照',[
          ('q3_controller_greedy','规则执行（主案）'),('q3_cross_day_false','剩余日储能LP'),('q3_cross_day_true','储能LP+跨日价值')]),
       'v2_settlement':('问题3结算解释的敏感性（其余设置相同）',[
          ('q3_primary','主口径：不退款'),('q3_refund_interpretation','取消部分退款')]),
       'v2_baselines':('问题2在相同预测与初态下的基础对照',[
          ('q2_no_storage','无储能场景采购'),('q2_point_forecast','点预测采购'),('q2_expectation_only','场景期望费用（主案）')])}
    for name,(title,items) in specs.items():
        rows=[]
        for key,label in items:
            if key not in ex or ex[key].get('status')!='complete':continue
            s=ex[key]['summary']
            rows.append([label,f(s['total_cost']/1e4),f(s['emergency_energy']),f(s['daily_cost_cvar95']),f(s['final_energy'])])
        if rows:table(name,title,['策略','总费/万元','紧急量/kWh','日费CVaR95/元','末库存/kWh'],rows,
             note='所有对照覆盖相同334天；设置在运行前冻结，评价期结果不回写主案选择。各组状态连续演化。')
    rows=[]
    for key,label in [('q2_risk_fixed_only','固定风险'),('q2_risk_adaptive_only','自适应风险'),
                     ('q2_reserve_fixed_only','固定备用'),('q2_reserve_adaptive_only','动态备用'),
                     ('q2_risk_reserve_fixed','固定联合'),('q2_risk_reserve_adaptive','自适应联合')]:
        if key in ex and ex[key].get('status')=='complete':
            s=ex[key]['summary'];rows.append([label,f'{s["mean_realized_risk_weight"]:.5f}',f(s['mean_realized_reference_reserve_kwh'])])
    if rows:table('v2_conservatism','固定与自适应策略在评价期实现的平均保守程度',
                  ['策略','平均风险权重','平均参考备用/kWh'],rows,
                  note='固定参数由一月动态序列的均值匹配；由于季节与误差分布变化，评价期均值不保证相等，不用评价期重新匹配。')
    rows=[]
    for c in ['q2','q3']:
        for n in [12,28]:
            key=f'{c}_scenarios_{n}'
            if key in ex and ex[key].get('status')=='complete':
                s=ex[key]['summary'];rows.append([CASES[c],str(n),f'{s["mean_seconds_per_day"]:.4f}'])
    if rows:table('v2_solve_times','场景规模与每日平均运行时间',['问题','场景数','秒/日'],rows,
                  note='运行时间含预测、外层求解与执行等开销，并行运行下的本机计时仅供工程量级参考。')


def focused_figures(results):
    plt=old.plt;h=np.arange(144)/6
    r=results['q3']
    # Pick the largest emergency day by a fixed descriptive rule, not a tuned strategy.
    i=int(np.argmax(r['emergency'].sum(axis=1)));date=str(r['dates'][i])
    fig,axs=plt.subplots(3,1,figsize=(8.1,6.0),sharex=True,layout='constrained')
    axs[0].bar(h,r['adjusted'][i]-r['plan'][i],width=1/6,color=old.COLORS[1]);axs[0].set_ylabel('常规上调量 / kWh')
    axs[1].plot(np.arange(145)/6,r['reference_energy'][i],label='各次外层参考轨迹拼接',color=old.COLORS[0],ls='--')
    axs[1].plot(np.arange(145)/6,r['energy'][i],label='实际储电量',color=old.COLORS[2]);axs[1].set_ylabel('储电量 / kWh');axs[1].legend()
    axs[2].bar(h,r['emergency'][i],width=1/6,color='#B45454');axs[2].set_ylabel('紧急购电量 / kWh');axs[2].set_xlabel('物理时刻 / h')
    for ax in axs:
        for t in [6,12,18]:ax.axvline(t,color='#888888',ls=':',lw=.7)
        ax.grid(axis='y',alpha=.18);ax.set_xlim(0,24);ax.set_xticks(range(0,25,2))
    axs[0].set_title('问题3：'+date+'（评价期紧急量最大日）',loc='left')
    for ext in ['pdf','png']:fig.savefig(FIG/('v2_control_difference.'+ext))
    plt.close(fig)
    # Four prescribed dates: adjustments on a dedicated scale.
    fig,axs=plt.subplots(2,2,figsize=(8.1,4.8),layout='constrained')
    for ax,date in zip(axs.flat,old.SELECTED):
        j=list(r['dates']).index(date);ax.bar(h,r['adjusted'][j]-r['plan'][j],width=1/6,color=old.COLORS[1])
        ax.set_title(date,loc='left');ax.set_ylabel('上调量 / kWh');ax.set_xlabel('时刻 / h');ax.set_xlim(0,24);ax.grid(axis='y',alpha=.18)
    for ext in ['pdf','png']:fig.savefig(FIG/('v2_adjustments.'+ext))
    plt.close(fig)
    fig,axs=plt.subplots(2,1,figsize=(8.1,5.2),sharex=True,layout='constrained')
    for j,(c,x) in enumerate(results.items()):
        month=np.array([int(d[5:7]) for d in x['dates']])
        axs[0].plot(range(2,13),[x['total_cost'][month==m].sum()/1e4 for m in range(2,13)],'o-',ms=3,label=CASES[c],color=old.COLORS[j])
        axs[1].plot(range(2,13),[x['emergency'][month==m].sum()/1e3 for m in range(2,13)],'o-',ms=3,color=old.COLORS[j])
    axs[0].set_ylabel('月费用 / 万元');axs[0].legend(ncol=4)
    axs[1].set_ylabel('紧急量 / 千kWh');axs[1].set_xlabel('月份');axs[1].set_xticks(range(2,13))
    for ax in axs:ax.grid(axis='y',alpha=.18)
    for ext in ['pdf','png']:fig.savefig(FIG/('v2_monthly.'+ext))
    plt.close(fig)
    z=np.load(DATA/'execution_gap_v1.npz');pre='q3_largest_positive_gap_'
    grid=z[pre+'grid'];w=z[pre+'weights'];ref=z[pre+'reference_soc'];soc=z[pre+'replay_soc']
    fig,axs=plt.subplots(2,1,figsize=(8.1,4.8),sharex=True,layout='constrained')
    x=np.linspace(24-len(grid)/6,24,len(ref))
    axs[0].plot(x,ref,label='共享参考SOC',color=old.COLORS[0])
    axs[0].plot(x,w@soc,label='场景回放平均SOC',color=old.COLORS[1]);axs[0].fill_between(x,soc.min(axis=0),soc.max(axis=0),alpha=.15,color=old.COLORS[1]);axs[0].legend();axs[0].set_ylabel('储电量 / kWh')
    a=z[pre+'replay_emergency']-z[pre+'planning_emergency'];p=z[pre+'price']
    axs[1].bar(x[:-1],np.sum(w[:,None]*5*p*a,axis=0),width=1/6,color=old.COLORS[3]);axs[1].axhline(0,color='#555555',lw=.6);axs[1].set_ylabel('期望紧急费差 / 元');axs[1].set_xlabel('物理时刻 / h')
    for ax in axs:ax.grid(axis='y',alpha=.18);ax.set_xlim(0,24)
    for ext in ['pdf','png']:fig.savefig(FIG/('v2_execution_gap.'+ext))
    plt.close(fig)


if __name__=='__main__':
    load_time_mapping()
    selection_tables();results,summaries=primary_tables();gap_table();experiment_tables();focused_figures(results)
    print('Generated v2 tables, daily_summary.csv and focused figures; no LaTeX compilation.')
