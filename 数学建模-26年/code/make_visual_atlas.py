"""八幅真实数据图：热力矩阵、守恒桑基、场景带和费用瀑布。

输入数据只读；不改变选择、运行数组、旧代码或工作簿，不调用 LaTeX。
使用系统中已有 NumPy/Matplotlib：python code/make_visual_atlas.py
输出 figures/results/<name>.pdf/.png 和 data/visual_atlas.json。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors, font_manager
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from time_index import aggregate_four_hours, four_hour_blocks, load_time_mapping

ROOT = Path(__file__).resolve().parents[1]
DATA, FIG = ROOT / 'data', ROOT / 'figures' / 'results'
BLUE, GREEN, ORANGE, RED = '#205C79', '#328879', '#D58B40', '#B45652'
INK, MUTED = '#233B49', '#687982'
for family in ('Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'DejaVu Sans'):
    if any(f.name == family for f in font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = family
        break
plt.rcParams.update({'font.size': 9, 'axes.labelsize': 9, 'legend.fontsize': 8,
                     'axes.titlesize': 11, 'axes.titleweight': 'bold',
                     'axes.titlelocation': 'left', 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.unicode_minus': False,
                     'pdf.fonttype': 42, 'savefig.dpi': 240,
                     'text.color': INK, 'axes.labelcolor': INK,
                     'xtick.color': INK, 'ytick.color': INK})
REPORT = {'scope': 'Visual summaries of frozen actual inputs and execution outputs; no model or workbook mutation',
          'physical_time_mapping': 'data/time_mapping.json', 'figures': {}}
PREVIOUS_REPORT = json.loads((DATA/'visual_atlas.json').read_text(encoding='utf-8')) if (DATA/'visual_atlas.json').exists() else {}


def read_json(name):
    return json.loads((DATA / name).read_text(encoding='utf-8'))


def read_npz(name):
    with np.load(DATA / name, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig, name, sources, caption, **metadata):
    FIG.mkdir(parents=True, exist_ok=True)
    outputs = []
    for extension in ('pdf', 'png'):
        output = FIG / f'{name}.{extension}'
        fig.savefig(output, bbox_inches='tight', facecolor='white')
        outputs.append(str(output.relative_to(ROOT)).replace('\\', '/'))
    plt.close(fig)
    REPORT['figures'][name] = {
        'outputs': outputs, 'sources_sha256': {f'data/{name}': sha(DATA / name) for name in sources},
        'caption': caption, **metadata}
    png_hash = sha(FIG/f'{name}.png')
    REPORT['figures'][name]['png_sha256'] = png_hash
    prior_review = PREVIOUS_REPORT.get('figures',{}).get(name,{}).get('visual_review')
    if prior_review and prior_review.get('reviewed_png_sha256') == png_hash:
        REPORT['figures'][name]['visual_review'] = prior_review
    print(f'Saved {name}', flush=True)


def q1_heatmap(q1, prepared):
    flows = np.array([aggregate_four_hours(v) for v in (
        q1['plan'], prepared['q1_pv'] / 6, prepared['q1_load'] / 6,
        q1['charge'], q1['discharge'])])
    state = np.asarray(q1['energy'])
    interval_mean = (state[:-1] + state[1:]) / 2
    mean_state = np.array([interval_mean[b['begin']:b['end']].mean() for b in four_hour_blocks()])
    blocks = [b['physical_label'].replace('-', '\n—\n') for b in four_hour_blocks()]
    fig = plt.figure(figsize=(8.6, 4.8))
    grid = fig.add_gridspec(2, 2, width_ratios=[1, .025], height_ratios=[5, 1.1], hspace=.32, wspace=.06)
    ax, stock = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0])
    cmap = colors.LinearSegmentedColormap.from_list('atlas_energy', ['#F4F8FA', '#A1C6D3', BLUE])
    im = ax.imshow(flows, cmap=cmap, vmin=0, vmax=float(flows.max()), aspect='auto')
    for row in range(len(flows)):
        for col in range(6):
            ax.text(col, row, f'{flows[row, col]:,.0f}', ha='center', va='center',
                    color='white' if flows[row, col] > flows.max() * .63 else INK, fontsize=9)
    ax.set_yticks(range(5), ['常规购电', '光伏产能', '负荷用能', '母线侧充电', '母线侧放电'])
    ax.set_xticks([])
    ax.set_title('问题1：四小时能量矩阵', pad=12)
    cb = fig.colorbar(im, cax=fig.add_subplot(grid[0, 1])); cb.set_label('四小时累计电量 / kWh')
    im2 = stock.imshow(mean_state[None, :], cmap='YlGnBu', vmin=1200, vmax=10800, aspect='auto')
    for col, value in enumerate(mean_state):
        stock.text(col, 0, f'{value:,.0f}', ha='center', va='center',
                   color='white' if value > 7500 else INK, fontsize=9)
    stock.set_yticks([0], ['平均储电量'])
    stock.set_xticks(range(6), blocks, fontsize=8)
    cb2 = fig.colorbar(im2, cax=fig.add_subplot(grid[1, 1])); cb2.set_label('库存均值 / kWh')
    cb2.set_ticks([1200, 6000, 10800])
    for panel in (ax, stock):
        panel.set_xticks(np.arange(-.5, 6, 1), minor=True)
        panel.set_yticks(np.arange(-.5, len(flows) if panel is ax else 1, 1), minor=True)
        panel.grid(which='minor', color='white', linewidth=1.5)
        panel.tick_params(which='both', length=0)
        for spine in panel.spines.values(): spine.set_visible(False)
    fig.subplots_adjust(left=.17, right=.89, top=.90, bottom=.17)
    fig.text(.17, .015, '数字为原始 kWh；上部各流量共用同一色标，库存均值使用独立色标。', fontsize=8, color=MUTED)
    save(fig, 'q1_energy_heatmap', ['q1.json', 'prepared.npz', 'time_mapping.json'],
         '问题一六个物理四小时时段的母线侧能量及平均库存。各流量格为原始累计kWh并共用色标；库存以相邻SOC边界线性插值的时段均值计算，另设1200—10800kWh色标，不将库存与流量混作同一统计量。',
         aggregation={'flow_rows': ['grid', 'pv', 'load', 'charge', 'discharge'],
                      'flow_kwh': flows.tolist(), 'mean_stored_energy_kwh': mean_state.tolist(),
                      'mean_state_definition': 'Mean of (E_t+E_t+1)/2 over the 24 physical slots of each block'},
         color_encoding='Raw flow kWh on one sequential scale; inventory kWh on a separate fixed scale')


def q1_sankey(q1, prepared):
    g = float(np.sum(q1['plan'])); pv = float(np.sum(prepared['q1_pv']) / 6)
    load = float(np.sum(prepared['q1_load']) / 6)
    c, d = float(np.sum(q1['charge'])), float(np.sum(q1['discharge']))
    spill = float(np.sum(q1['spill'])); initial, final = q1['energy'][0], q1['energy'][-1]
    stored, withdrawn = .9 * c, d / .9
    closs, dloss = c - stored, withdrawn - d
    balances = {'common_bus': g + pv + d - load - c - spill,
                'charge_conversion': c - stored - closs,
                'storage_account': initial + stored - withdrawn - final,
                'discharge_conversion': withdrawn - d - dloss,
                'whole_day': g + pv + initial - load - spill - closs - dloss - final}
    assert max(abs(v) for v in balances.values()) < 1e-6, balances
    fig, ax = plt.subplots(figsize=(10, 6.8))
    ax.set_xlim(-.11, 1.12); ax.set_ylim(-.08, 1.22); ax.axis('off')
    scale = 4.2e-6  # one shared geometric scale for every flow and node
    width = .019
    amounts = {'grid': g, 'pv': pv, 'load': load, 'charge': c, 'stored_charge': stored,
               'withdrawn_storage': withdrawn, 'discharge': d, 'charge_loss': closs,
               'discharge_loss': dloss, 'initial_storage': initial, 'final_storage': final, 'spill': spill}
    def ribbon(x0, y0, x1, y1, value, color, control0=None, control1=None, alpha=.58):
        if value < 1e-7: return
        h = value * scale
        direction = 1 if x1 > x0 else -1
        a = x0 + direction * abs(x1-x0) * .43 if control0 is None else control0
        b = x1 - direction * abs(x1-x0) * .43 if control1 is None else control1
        vertices = [(x0,y0),(a,y0),(b,y1),(x1,y1),
                    (x1,y1+h),(b,y1+h),(a,y0+h),(x0,y0+h),(x0,y0)]
        codes = [MplPath.MOVETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4,
                 MplPath.LINETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4,MplPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MplPath(vertices,codes),facecolor=color,edgecolor=color,lw=.25,alpha=alpha,zorder=1))
    def node(x, y, amount, color=INK):
        ax.add_patch(Rectangle((x,y),width,amount*scale,facecolor=color,edgecolor='white',lw=.7,zorder=3))
    def turned_ribbon(source, target, amount, departure, arrival, color, reach=.17):
        """A rotated-port ribbon: scale is preserved perpendicular to each port.

        Horizontal converter nodes let the storage loop turn through 90 degrees
        without the self-overlap caused by forcing both ports to stay vertical.
        """
        p0,p3=np.array(source,float),np.array(target,float)
        ds,dt=np.array(departure,float),np.array(arrival,float)
        p1,p2=p0+reach*ds,p3-reach*dt
        ns,nt=np.array([-ds[1],ds[0]]),np.array([-dt[1],dt[0]])
        half=amount*scale/2
        upper=[p0+half*ns,p1+half*ns,p2+half*nt,p3+half*nt]
        lower=[p3-half*nt,p2-half*nt,p1-half*ns,p0-half*ns]
        vertices=[*upper,*lower,upper[0]]
        codes=[MplPath.MOVETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4,
               MplPath.LINETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4,MplPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MplPath(vertices,codes),facecolor=color,edgecolor=color,lw=.25,alpha=.58,zorder=1))
    def label(x, y, text, ha='center', **kwargs):
        ax.text(x,y,text,ha=ha,va='center',fontsize=9,zorder=5,**kwargs)
    busx, busy = .43, .49
    bus_h = (g+pv+d)*scale
    # Every external source joins one common electrical bus before distribution.
    ribbon(.015,busy+d*scale,busx,busy+d*scale,g,BLUE)
    ribbon(.015,busy+(d+g)*scale+.017,busx,busy+(d+g)*scale,pv,GREEN)
    node(-.004,busy+d*scale,g,BLUE);node(-.004,busy+(d+g)*scale+.017,pv,GREEN)
    node(busx,busy,g+pv+d)
    ribbon(busx+width,busy+c*scale,.985,.565,load,'#7C91A0',alpha=.43)
    node(.985,.565,load,'#748A99')
    # Storage is drawn as an explicit feedback loop; no grid/PV attribution is invented.
    chx, chy = .72, .245
    stx, sty = .45, .080
    dsx, dsy = .17, .245
    charge_left=chx-c*scale/2
    discharge_left=dsx-withdrawn*scale/2
    turned_ribbon((busx+width,busy+c*scale/2),(chx,chy+width),c,(1,0),(0,-1),ORANGE)
    ax.add_patch(Rectangle((charge_left,chy),c*scale,width,facecolor=ORANGE,edgecolor='white',lw=.7,zorder=3))
    turned_ribbon((charge_left+stored*scale/2,chy),(stx+width,sty+initial*scale+stored*scale/2),
                  stored,(0,-1),(-1,0),ORANGE,reach=.11)
    ribbon(.93,.022,stx+width,sty,initial,'#B7A284')
    node(.93,.022,initial,'#B7A284');node(stx,sty,initial+stored,'#5F7C78')
    turned_ribbon((stx,sty+final*scale+withdrawn*scale/2),(dsx,dsy),withdrawn,(-1,0),(0,1),GREEN,reach=.11)
    ribbon(stx,sty,.005,.022,final,'#B7A284')
    node(-.014,.022,final,'#B7A284')
    ax.add_patch(Rectangle((discharge_left,dsy),withdrawn*scale,width,facecolor=GREEN,edgecolor='white',lw=.7,zorder=3))
    turned_ribbon((discharge_left+dloss*scale+d*scale/2,dsy+width),(busx,busy+d*scale/2),
                  d,(0,1),(1,0),GREEN,reach=.16)
    # Converter losses are counted at each side of the 90%-efficient battery.
    turned_ribbon((charge_left+stored*scale+closs*scale/2,chy),(.985,.168+closs*scale/2),
                  closs,(0,-1),(1,0),RED,reach=.10)
    turned_ribbon((discharge_left+dloss*scale/2,dsy+width),(-.004,.305+dloss*scale/2),
                  dloss,(0,1),(-1,0),RED,reach=.07)
    node(.985,.168,closs,RED);node(-.023,.305,dloss,RED)
    label(-.035,busy+d*scale+g*scale/2,f'电网购电\n{g:,.1f}',ha='right',color=BLUE)
    label(-.035,busy+(d+g)*scale+.017+pv*scale/2,f'光伏产能\n{pv:,.1f}',ha='right',color=GREEN)
    label(1.02,.565+load*scale/2,f'负荷用能\n{load:,.1f}',ha='left')
    label(busx+width/2,busy+bus_h+.057,'共用母线',fontweight='bold')
    label(.69,.428,f'充电输入 {c:,.1f}',color='#9E642B')
    label(.275,.423,f'放电入母线 {d:,.1f}',color=GREEN)
    label(chx+.075,chy+.065,'充电转换 η=0.9',color='#9E642B')
    label(dsx-.015,dsy-.065,'放电转换 η=0.9',color=GREEN)
    label(.455,.233,'储能日内能量账本',fontweight='bold')
    label(.44,-.025,f'充入储能 {stored:,.1f}；从储能取出 {withdrawn:,.1f}',color=MUTED)
    label(.96,-.026,f'期初库存 {initial:,.1f}',color='#857158')
    label(.01,-.026,f'期末库存 {final:,.1f}',color='#857158')
    label(1.02,.172,f'充电损耗\n{closs:,.1f}',ha='left',color=RED)
    label(-.045,.309,f'放电损耗\n{dloss:,.1f}',ha='right',color=RED)
    label(.85,.535,f'弃余 {max(0.,spill):.1f}',color=MUTED)
    ax.set_title('问题1：电量守恒桑基图',pad=12)
    fig.text(.13,.045,'全部流带按同一 kWh 比例绘制；母线汇合后不追溯购电与光伏的充电份额。',fontsize=8,color=MUTED)
    fig.text(.13,.018,f'外部平衡：{g+pv+initial:,.1f} = 负荷 {load:,.1f} + 损耗 {closs+dloss:,.1f} + 期末库存 {final:,.1f} kWh',fontsize=8,color=MUTED)
    fig.subplots_adjust(left=.12,right=.88,top=.93,bottom=.11)
    save(fig,'q1_energy_sankey',['q1.json','prepared.npz'],
         '问题一全过程能量守恒桑基。电网与光伏在共用母线汇合；储能以母线—充电转换—库存账本—放电转换—母线的闭合回路表示，显式计入两侧90%效率损耗及期初、期末库存。流带使用同一kWh几何尺度，未分配无法识别的电源充电归属；零弃余仅标数值，不虚画流带。',
         flow_kwh=amounts,node_balance_residual_kwh=balances,
         geometric_scale='4.2e-6 axes coordinate perpendicular port width per kWh, common to every ribbon; charge/discharge conversion ports are rotated 90 degrees',
         storage_definition='Initial plus charged cell-side energy equals discharged cell-side energy plus final inventory; aggregate loop is not a chronological charge source attribution')


def q2_calendar(q2):
    dates = list(q2['dates']); first = [i for i,d in enumerate(dates) if d.endswith('-01')]
    labels = [str(int(dates[i][5:7]))+'月' for i in first]
    emergency = np.asarray(q2['emergency']); state = q2['energy'][:,1:]
    positive = emergency > 1e-7
    masked = np.ma.masked_where(~positive,emergency)
    cmap = plt.get_cmap('YlOrRd').copy();cmap.set_bad('#F6F8F9')
    fig,axs=plt.subplots(1,2,figsize=(9,6.2),sharey=True,layout='constrained')
    im=axs[0].imshow(masked,aspect='auto',interpolation='nearest',origin='upper',extent=[0,24,334,0],
        cmap=cmap,norm=colors.PowerNorm(.4,vmin=0,vmax=float(emergency.max())))
    axs[0].set_title('紧急购电：稀疏时段分布')
    cb=fig.colorbar(im,ax=axs[0],location='bottom',shrink=.92,pad=.065);cb.set_label('紧急电量 / kWh（幂次色标 γ=0.4）')
    im2=axs[1].imshow(state,aspect='auto',interpolation='nearest',origin='upper',extent=[0,24,334,0],
        cmap='YlGnBu',vmin=1200,vmax=10800)
    axs[1].set_title('储能：每日时段末库存')
    cb2=fig.colorbar(im2,ax=axs[1],location='bottom',shrink=.92,pad=.065);cb2.set_label('时段末储电量 / kWh')
    cb2.set_ticks([1200,3600,6000,8400,10800])
    for ax in axs:
        ax.set_xticks([0,6,12,18,24]);ax.set_xlabel('物理时刻 / h')
        ax.set_yticks(first,labels);ax.tick_params(length=3)
        for row in first:ax.axhline(row,color='white',lw=.5,alpha=.7)
    axs[0].set_ylabel('2025年评价日期（逐日排列）')
    fig.suptitle('问题2：334天 × 144时段的运行矩阵',x=.02,ha='left',fontsize=12,fontweight='bold')
    save(fig,'q2_calendar_heatmap',['q2_v2.npz','time_mapping.json'],
         '问题二2025年2—12月334天、每日144个物理十分钟时段的紧急电量和时段末储电量。左侧零紧急值留浅灰，正值使用γ=0.4的幂次色标以显示稀疏小额事件，色条仍标原始kWh；右侧为1200—10800kWh固定库存色标。每格一个真实执行时段，不平滑或补采样。',
         dimensions=[334,144],date_range=[dates[0],dates[-1]],positive_emergency_cells=int(positive.sum()),
         emergency_kwh=float(emergency.sum()),max_emergency_kwh=float(emergency.max()),
         soc_sampling='E_t+1 at each ten-minute interval end',zero_encoding='Light gray; emergency <= 1e-7 kWh',
         emergency_color_normalization={'type':'PowerNorm','gamma':.4,'vmin':0,'vmax':float(emergency.max())})


def q2_risk(ex):
    keys=['q2_expectation_only','q2_risk_fixed_only','q2_risk_adaptive_only','q2_reserve_fixed_only',
          'q2_reserve_adaptive_only','q2_risk_reserve_fixed','q2_risk_reserve_adaptive']
    labels=['期望费用\n主案','固定\n风险','自适应\n风险','固定\n备用','动态\n备用','固定\n风险+备用','自适应\n风险+备用']
    metrics=['total_cost','emergency_energy','daily_cost_cvar95','mean_realized_reference_reserve_kwh','final_energy']
    rowlabels=['总费用差 / %','紧急电量差 / %','日费CVaR95差 / %','平均参考备用差 / kWh','期末库存差 / kWh']
    raw=np.array([[ex[k]['summary'][metric] for k in keys] for metric in metrics])
    delta=raw-raw[:,0:1]
    delta[:3]=delta[:3]/raw[:3,0:1]*100
    denominators=np.max(np.abs(delta),axis=1);denominators[denominators==0]=1
    normalized=delta/denominators[:,None]
    fig,ax=plt.subplots(figsize=(9.5,4.7))
    cmap=colors.LinearSegmentedColormap.from_list('delta_atlas',[BLUE,'#FAFBFC',RED])
    im=ax.imshow(normalized,cmap=cmap,vmin=-1,vmax=1,aspect='auto')
    for row in range(len(metrics)):
        for col in range(len(keys)):
            text='0' if col==0 else (f'{delta[row,col]:+.2f}%' if row<3 else
                                   ('0.0' if abs(delta[row,col])<.05 else f'{delta[row,col]:+,.1f}'))
            ax.text(col,row,text,ha='center',va='center',color='white' if abs(normalized[row,col])>.65 else INK,fontsize=9)
    ax.set_xticks(range(len(keys)),labels,fontsize=8)
    ax.set_yticks(range(len(metrics)),rowlabels)
    ax.set_xticks(np.arange(-.5,len(keys),1),minor=True);ax.set_yticks(np.arange(-.5,len(metrics),1),minor=True)
    ax.grid(which='minor',color='white',lw=2);ax.tick_params(which='both',length=0)
    for spine in ax.spines.values():spine.set_visible(False)
    ax.set_title('问题2：风险与备用相对经济主案的指标变化',pad=14)
    cb=fig.colorbar(im,ax=ax,fraction=.024,pad=.03);cb.set_label('行内相对强度（负值下降，正值上升）')
    cb.set_ticks([-1,0,1],labels=['−1','0','+1'])
    fig.subplots_adjust(left=.24,right=.94,top=.86,bottom=.21)
    fig.text(.24,.05,'格内保留真实差值；颜色按各指标最大绝对差缩放，跨行颜色不能比较变化量大小。',fontsize=8,color=MUTED)
    fig.text(.24,.018,'备用主案为零，使用 kWh 绝对差；期末库存另列，不将库存差隐藏在节费结论中。',fontsize=8,color=MUTED)
    save(fig,'q2_risk_heatmap',['experiments_v2.json'],
         '七种风险与备用设置对经济主案的指标差异。前三行为相对主案百分比，平均参考备用因主案为零使用kWh绝对差，末库存使用kWh差。颜色在每一指标行内除以最大绝对变化，仅展示变化方向和行内强弱，不支持跨行色深的量级比较；数字保留真实单位，红色不自动等同于不利。',
         policy_keys=keys,metric_keys=metrics,raw_metric_values=raw.tolist(),display_deltas=delta.tolist(),
         displayed_units=['percent','percent','percent','kWh','kWh'],
         color_row_denominators=denominators.tolist(),color_formula='row_delta/max(abs(row_delta))',
         reserve_note='Reference reserve is an optimizer setting, not an enforced actual-SOC buffer')


def waterfall(name,first,deltas,rowlabels,title,sources,caption,**metadata):
    first=first/1e4;changes=np.asarray(deltas)/1e4
    levels=np.r_[first,first+np.cumsum(changes)]
    last=float(levels[-1]);n=len(changes)+2
    low=float(levels.min()-max(4,np.ptp(levels)*.23));high=float(levels.max()+max(8,np.ptp(levels)*.32))
    fig,(left,right)=plt.subplots(1,2,figsize=(9.2,4.2),sharey=True,
        gridspec_kw={'width_ratios':[.6,6.3],'wspace':.055})
    totals={0:first,n-1:last}
    for ax in (left,right):
        for row,value in totals.items():ax.barh(row,value,color=BLUE,height=.58,zorder=3)
        for j,delta in enumerate(changes,start=1):
            previous=levels[j-1];current=levels[j]
            ax.barh(j,abs(delta),left=min(previous,current),color=RED if delta>0 else GREEN,height=.58,zorder=3)
        for j in range(n-1):
            level=levels[min(j,len(levels)-1)]
            ax.plot([level,level],[j+.30,j+1-.30],color='#7D8C94',lw=.8,ls='--',zorder=2)
        ax.set_ylim(n-.45,-.65)
        ax.grid(axis='x',color='#DDE4E8',lw=.6,zorder=0)
        ax.tick_params(axis='y',length=0)
    left.set_xlim(0,25);left.set_xticks([0]);left.set_yticks(range(n),rowlabels)
    right.set_xlim(low,high);right.tick_params(axis='y',labelleft=False)
    left.spines['right'].set_visible(False);right.spines['left'].set_visible(False)
    mark=.013
    for ax,x in ((left,1),(right,0)):
        kw={'transform':ax.transAxes,'color':MUTED,'clip_on':False,'lw':1}
        ax.plot((x-mark,x+mark),(-mark,+mark),**kw)
        ax.plot((x-mark,x+mark),(1-mark,1+mark),**kw)
    for row,value in totals.items():
        right.text(value+(high-low)*.017,row,f'{value:,.2f}',ha='left',va='center',color=BLUE,fontweight='bold')
    for j,delta in enumerate(changes,start=1):
        current=levels[j]
        right.text(current+(high-low)*.017 if delta>0 else current-(high-low)*.017,j,
                   f'{delta:+.2f}',ha='left' if delta>0 else 'right',va='center',color=RED if delta>0 else GREEN,fontweight='bold')
    fig.suptitle(title,x=.02,y=.99,ha='left',fontsize=12,fontweight='bold')
    fig.subplots_adjust(left=.24,right=.94,top=.87,bottom=.22)
    fig.text(.59,.08,'实际结算费用 / 万元（横轴断开；蓝色为总额，绿/红为减少/增加）',ha='center',fontsize=8,color=MUTED)
    save(fig,name,sources,caption,first_total_yuan=first*1e4,component_changes_yuan=(changes*1e4).tolist(),
         successive_totals_yuan=(levels*1e4).tolist(),final_total_yuan=last*1e4,
         axis='Horizontal broken axis; total bars begin at zero in the left panel, right panel magnifies the invoice region',
         axis_unit='10000 CNY',**metadata)


def q3_feedback(ex):
    keys=['q3_midnight_only','q3_feedback_soc_only','q3_feedback_pv_only','q3_feedback_full']
    bills=np.array([ex[key]['summary']['total_cost'] for key in keys])
    waterfall('q3_feedback_waterfall',bills[0],np.diff(bills),
              ['仅零时计划：总费','仅更新SOC并重优化','再加入新光伏预报','再加入负荷偏差反馈','完整反馈：总费'],
              '问题3：滚动反馈的费用变化分解',['experiments_v2.json'],
              '从仅零时计划依次增加SOC更新、光伏预报和负荷偏差反馈的实际总费用瀑布。正增量完整保留：加入光伏预报这一条件差使费用上升，不将其隐藏为单调收益。横轴为万元并明确断开，起终蓝条均从零起，其他条为相邻方案差；这些条件比较不等于独立因果效应。',
              policy_sequence=keys,conditional_comparison='All policies use the same frozen forecasts/base inputs and January prefix within Q3; state and realized decisions evolve separately')


def weighted_quantile(values,weights,probability):
    order=np.argsort(values,axis=0)
    ordered=np.take_along_axis(values,order,axis=0)
    cdf=np.cumsum(weights[order],axis=0)
    indices=np.argmax(cdf>=probability,axis=0)
    return ordered[indices,np.arange(values.shape[1])]


def q3_band():
    diagnostic=read_json('inner_gap_v2.json')
    # The v2 file has expected SOC only. A quantile band requires the actual
    # scenario-by-time trajectories, retained in this frozen v1 diagnostic.
    assert 'greedy_expected_soc_kwh' in diagnostic['decisions'][0]
    arrays=read_npz('execution_gap_v1.npz');info=read_json('execution_gap_v1.json')
    example='q3_largest_positive_gap';prefix=example+'_'
    states=arrays[prefix+'replay_soc'];reference=arrays[prefix+'reference_soc'];weights=arrays[prefix+'weights']
    assert abs(weights.sum()-1)<1e-10 and np.all(weights>=0)
    lo=weighted_quantile(states,weights,.1);hi=weighted_quantile(states,weights,.9);mean=weights@states
    meta=info['examples'][example]
    x=np.arange(states.shape[1])/6+meta['physical_start_index']/6
    fig,ax=plt.subplots(figsize=(8.8,4.1))
    ax.fill_between(x,lo,hi,color='#8CB8CE',alpha=.46,label='加权 P10—P90 场景带（非置信区间）',zorder=1)
    for line in states:ax.plot(x,line,color='#9AAEB8',lw=.45,alpha=.32,zorder=2)
    ax.plot(x,mean,color=BLUE,lw=2,label='场景概率加权平均SOC',zorder=3)
    ax.plot(x,reference,color=RED,ls='--',lw=1.8,label='规划共享参考SOC',zorder=4)
    ax.set_xticks(range(0,25,3));ax.set_xlim(0,24);ax.set_ylim(800,11200)
    ax.set_yticks([1200,3600,6000,8400,10800]);ax.grid(axis='y',alpha=.18)
    ax.set_xlabel('物理时刻 / h');ax.set_ylabel('储电量 / kWh')
    ax.set_title(f'问题3原版固定计划回放：{meta["date"]} {meta["update_hour"]:02d}:00')
    ax.legend(loc='upper left',bbox_to_anchor=(0,-.21),ncol=1,frameon=False)
    fig.subplots_adjust(left=.10,right=.98,top=.86,bottom=.29)
    save(fig,'q3_execution_band',['execution_gap_v1.npz','execution_gap_v1.json','inner_gap_v2.json'],
         f'原版问题三{meta["date"]}零时同一合同的12场景回放。阴影为每个时刻按原场景概率计算的P10—P90加权经验分位带，细灰线为个体场景，实线为概率均值，虚线为共享参考SOC。该带是场景分布范围，不是置信区间或全年覆盖保证。新版诊断只保存期望轨迹，故使用保留个体轨迹的原版案例，未伪造新版分位带。',
         diagnostic_version='v1_before_settlement_revision',example=example,date=meta['date'],
         update_hour=meta['update_hour'],scenario_count=len(weights),weights=weights.tolist(),
         quantile_definition='Weighted empirical inverse CDF, min x with cumulative scenario probability >= p, independently at each boundary',
         quantiles=[.1,.9],lower_soc_kwh=lo.tolist(),upper_soc_kwh=hi.tolist(),mean_soc_kwh=mean.tolist(),
         reference_soc_kwh=reference.tolist(),uncertainty_interpretation='Scenario band only, not a confidence interval; the selected diagnostic day is not representative by random sampling')


def q4_correlation(q4,prepared):
    actual_load,actual_pv,actual_price=[prepared[k][31:] for k in ('load','pv','price')]
    def issued(key):
        v=prepared[key]
        return np.concatenate([v[31:,j,36*j:36*(j+1)] for j in range(4)],axis=1)
    predicted_load=issued('forecast_load_issue');predicted_pv=issued('pv_issue_forecast')
    predicted_price=issued('forecast_price_issue')
    assert np.max(np.abs((predicted_load-predicted_pv)/6-q4['forecast_net']))<1e-8
    assert np.max(np.abs(predicted_price-q4['forecast_price']))<1e-8
    actual=np.array([x.ravel() for x in [actual_load,actual_pv,actual_load-actual_pv,actual_price]])
    residual=np.array([x.ravel() for x in [actual_load-predicted_load,actual_pv-predicted_pv,
                       actual_load-actual_pv-predicted_load+predicted_pv,actual_price-predicted_price]])
    correlation=[np.corrcoef(values) for values in (actual,residual)]
    assert actual.shape==(4,48096) and np.isfinite(correlation).all()
    fig,axs=plt.subplots(1,2,figsize=(8.8,4.0))
    names=['负荷','光伏','净负荷','电价'];cmap=colors.LinearSegmentedColormap.from_list('corr_atlas',[BLUE,'#FAFBFC',RED])
    for ax,matrix,title in zip(axs,correlation,['实际值','预测误差：实际 − 最新合法预测']):
        im=ax.imshow(matrix,vmin=-1,vmax=1,cmap=cmap)
        for row in range(4):
            for col in range(4):
                ax.text(col,row,f'{matrix[row,col]:+.2f}',ha='center',va='center',
                        color='white' if abs(matrix[row,col])>.68 else INK,fontsize=11)
        ax.set_xticks(range(4),names);ax.set_yticks(range(4),names)
        ax.set_xticks(np.arange(-.5,4,1),minor=True);ax.set_yticks(np.arange(-.5,4,1),minor=True)
        ax.grid(which='minor',color='white',lw=1.5);ax.tick_params(which='both',length=0)
        for spine in ax.spines.values():spine.set_visible(False)
        ax.set_title(title,pad=10,fontsize=10)
    fig.subplots_adjust(left=.09,right=.88,top=.80,bottom=.17,wspace=.29)
    cb=fig.colorbar(im,cax=fig.add_axes([.92,.22,.02,.51]));cb.set_label('Pearson 相关系数')
    fig.suptitle('问题4：价荷实际值与预测误差的相关结构',x=.05,y=.98,ha='left',fontsize=12,fontweight='bold')
    fig.text(.09,.025,'334天 × 144时段；相关不表示因果，未校正日内/季节共同变化或时间依赖。',fontsize=8,color=MUTED)
    save(fig,'q4_correlation_heatmap',['q4_3_v2.npz','prepared_v2_q4_3.npz','selection_v2.json'],
         '问题四-3冻结配置下48,096个评价时段的实际值与预测误差Pearson相关矩阵。误差为实际减去各时段最近一次合法发布的预测，按0、6、12、18时的可执行36段拼接，并与执行结果内保存的净负荷和价格预测逐项核对。色标固定为−1至1；相关不等于因果，未作时间依赖校正或显著性推断。',
         n_observations=48096,variable_order=names,actual_correlation=correlation[0].tolist(),
         residual_correlation=correlation[1].tolist(),residual_units=['kW','kW','kW','CNY/kWh'],
         forecast_join='Within each day concatenate issue j forecast at slots [36*j,36*(j+1)) for j=0..3',
         statistical_limit='Descriptive pooled Pearson correlations; temporal dependence and common seasonal/diurnal structure remain')


def q4_waterfall(a,b):
    keys=['plan_cost','adjustment_cost','emergency_cost']
    delta=np.array([np.sum(b[k])-np.sum(a[k]) for k in keys])
    first,last=float(np.sum(a['total_cost'])),float(np.sum(b['total_cost']))
    assert abs(first+delta.sum()-last)<1e-6
    waterfall('q4_cost_waterfall',first,delta,
              ['静态方案：总费','日前计划费变化','计划调整费变化','紧急购电费变化','滚动方案：总费'],
              '问题4：静态至滚动方案的账单分解',['q4_2_v2.npz','q4_3_v2.npz'],
              '问题四-2到问题四-3总费用的账单分项瀑布。计划费与调整费的正增量、紧急购电费的负增量逐项显示，三者之和严格等于总费差。横轴万元并明确断开，起终条表示从零累计的总额。两方案按各自一月冻结配置运行，因此这是账单恒等分解，不将差值全归因为单一滚动模块。',
              component_keys=keys,conservation_residual_yuan=first+float(delta.sum())-last,
              initial_storage_kwh=[float(a['energy'][0,0]),float(b['energy'][0,0])],
              final_storage_kwh=[float(a['energy'][-1,-1]),float(b['energy'][-1,-1])])


def main():
    load_time_mapping()
    inputs=['q1.json','prepared.npz','q2_v2.npz','q4_2_v2.npz','q4_3_v2.npz',
            'prepared_v2_q4_3.npz','selection_v2.json','experiments_v2.json',
            'execution_gap_v1.json','execution_gap_v1.npz','inner_gap_v2.json','time_mapping.json']
    fingerprints={name:sha(DATA/name) for name in inputs}
    q1=read_json('q1.json');prepared=read_npz('prepared.npz');q2=read_npz('q2_v2.npz')
    q4a,q4b=read_npz('q4_2_v2.npz'),read_npz('q4_3_v2.npz')
    experiments=read_json('experiments_v2.json')
    assert experiments['completed']
    q1_heatmap(q1,prepared);q1_sankey(q1,prepared);q2_calendar(q2)
    q2_risk(experiments['experiments']);q3_feedback(experiments['experiments']);q3_band()
    q4_correlation(q4b,read_npz('prepared_v2_q4_3.npz'));q4_waterfall(q4a,q4b)
    assert all(sha(DATA/name)==value for name,value in fingerprints.items()), 'A source file changed during plotting'
    REPORT['checks']={'source_files_unchanged':True,'figure_count':len(REPORT['figures']),
                      'q1_all_node_flow_balance_max_abs_kwh':max(abs(x) for x in REPORT['figures']['q1_energy_sankey']['node_balance_residual_kwh'].values())}
    (DATA/'visual_atlas.json').write_text(json.dumps(REPORT,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Eight figures and provenance report complete. Inputs unchanged; no LaTeX run.',flush=True)


if __name__=='__main__':main()
