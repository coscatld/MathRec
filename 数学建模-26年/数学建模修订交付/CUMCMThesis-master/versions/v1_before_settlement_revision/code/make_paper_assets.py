"""从真实求解数组生成论文数表、插图和可追溯CSV；不调用LaTeX。"""
from pathlib import Path
import sys
import json
import csv
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".python_deps"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

DATA, TABLES, FIG = ROOT/"data", ROOT/"tables", ROOT/"figures/results"
TABLES.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
for font in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"):
    if any(f.name == font for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = font
        break
plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,
                     "axes.unicode_minus":False,"pdf.fonttype":42,"savefig.dpi":220,
                     "axes.labelsize":9,"legend.fontsize":8})
COLORS=["#1F5B7B","#E28F41","#328879","#8A607E"]
SELECTED=["2025-03-20","2025-06-21","2025-09-23","2025-12-21"]
LABELS={"q2":"问题2","q3":"问题3","q4_2":"问题4-2","q4_3":"问题4-3"}


def fmt(x): return f"{float(x):.2f}"
def write(name,text): (TABLES/name).write_text(text,encoding="utf-8")
def clock(i): return f"{i//6:02d}:{(i%6)*10:02d}"
def intervals(v):
    ids=np.flatnonzero(v>1e-6)
    groups=np.split(ids,np.flatnonzero(np.diff(ids)>1)+1) if len(ids) else []
    return [(f"{clock(g[0])}--{clock(g[-1]+1)}",float(v[g].sum())) for g in groups]


def purchase_block(v,energy_total,cost):
    s=[r"\begin{tabularx}{\linewidth}{lRlRlR}",r"\toprule",
       r"时间段 & 购电量 & 时间段 & 购电量 & 时间段 & 购电量\\",r"\midrule"]
    for row in [[60,72,84],[96,108,120]]:
        s.append(" & ".join(f"{clock(t)}--{clock(t+1)} & {fmt(v[t])}" for t in row)+r"\\")
    s.extend([r"\midrule",f"全天购电量 & {fmt(energy_total)} & "+r"\multicolumn{2}{c}{全天购电费（元）} & \multicolumn{2}{r}{"+fmt(cost)+r"}\\",
              r"\bottomrule",r"\end{tabularx}"])
    return "\n".join(s)


def battery_block(ch,dis,e):
    s=[r"\begin{tabularx}{\linewidth}{lRRlRR}",r"\toprule",
       r"时间段 & 充电量 & 放电量 & 时间段 & 充电量 & 放电量\\",r"\midrule"]
    for a,b in [(0,24),(48,72),(96,120)]:
        cells=[]
        for t in [a,b]:
            cells.extend([f"{clock(t)}--{clock(t+24)}",fmt(ch[t:t+24].sum()),fmt(dis[t:t+24].sum())])
        s.append(" & ".join(cells)+r"\\")
    s.extend([r"\midrule",r"0:00储电量 & \multicolumn{2}{r}{"+fmt(e[0])+r"} & 24:00储电量 & \multicolumn{2}{r}{"+fmt(e[-1])+r"}\\",r"\bottomrule",r"\end{tabularx}"])
    return "\n".join(s)


def selected_tables(case,r):
    ids=[list(r["dates"]).index(d) for d in SELECTED]
    arr=r["adjusted"] if case in ("q3","q4_3") else r["plan"]
    p=[r"\begin{table}[htbp]\centering\small",
       r"\caption{"+LABELS[case]+r"指定日期购电结果（沿用题目表1布局，电量单位：kWh）}",
       r"\label{tab:"+case+r"-purchase}"]
    b=[r"\begin{table}[htbp]\centering\small",
       r"\caption{"+LABELS[case]+r"指定日期储能结果（沿用题目表2布局，单位：kWh）}",
       r"\label{tab:"+case+r"-battery}"]
    for date,i in zip(SELECTED,ids):
        p += [r"\par\smallskip\textbf{"+date+r"}\par\smallskip",
              purchase_block(arr[i],arr[i].sum(),r["total_cost"][i])]
        b += [r"\par\smallskip\textbf{"+date+r"}\par\smallskip",
              battery_block(r["charge"][i],r["discharge"][i],r["energy"][i])]
    p += [r"\par\smallskip\parbox{\linewidth}{\footnotesize 注：问题2与4-2列示计划购电量；问题3与4-3列示最终生效的调整后常规购电量。全天购电量不含紧急购电，全天购电费含计划、调整与紧急费用。}",r"\end{table}"]
    b += [r"\end{table}"]
    em=[intervals(r["emergency"][i]) for i in ids]
    e=[r"\begin{table}[htbp]\centering\small",
       r"\caption{"+LABELS[case]+r"指定日期紧急购电（题目表3布局，单位：kWh）}",
       r"\label{tab:"+case+r"-emergency}",r"\setlength{\tabcolsep}{3pt}",
       r"\begin{tabularx}{\linewidth}{lRlRlRlR}\toprule",
       " & ".join(r"\multicolumn{2}{c}{"+d+r"}" for d in SELECTED)+r"\\",
       " & ".join(["时间段 & 购电量"]*4)+r"\\\midrule"]
    for j in range(max(1,max(map(len,em)))):
        cells=[]
        for group in em:
            if j<len(group): cells += [group[j][0],fmt(group[j][1])]
            elif j==0 and not group: cells += ["无","0.00"]
            else: cells += [""," "]
        e.append(" & ".join(cells)+r"\\")
    e += [r"\midrule", " & ".join("合计 & "+fmt(r["emergency"][i].sum()) for i in ids)+r"\\",
          r"\bottomrule\end{tabularx}\end{table}"]
    write(f"{case}_selected.tex","\n\n".join(["\n".join(p),"\n".join(b),"\n".join(e)]))
    # Transparent cost reconciliation and before/after day-ahead volumes.
    c=[r"\begin{table}[htbp]\centering\small",
       r"\caption{"+LABELS[case]+r"指定日期计划、调整与费用分解}",
       r"\begin{tabularx}{\linewidth}{lRRRRR}\toprule",
       r"日期 & 计划量/kWh & 生效量/kWh & 计划费/元 & 调整费/元 & 紧急费/元\\\midrule"]
    for date,i in zip(SELECTED,ids):
        c.append(" & ".join([date,fmt(r["plan"][i].sum()),fmt(r["adjusted"][i].sum()),
                              fmt(r["plan_cost"][i]),fmt(r["adjustment_cost"][i]),fmt(r["emergency_cost"][i])])+r"\\")
    c += [r"\bottomrule\end{tabularx}\end{table}"]
    write(f"{case}_decomposition.tex","\n".join(c))


def figures(data,q1,results):
    h=np.arange(144)/6
    fig,axs=plt.subplots(3,1,figsize=(8.1,6.4),sharex=True,layout="constrained")
    axs[0].plot(h,data["q1_load"],label="小区负荷",color=COLORS[0])
    axs[0].plot(h,data["q1_pv"],label="光伏出力",color=COLORS[1])
    axs[0].step(h,np.asarray(q1["plan"])*6,where="post",label="外网购电",color=COLORS[2])
    axs[0].set_ylabel("功率 / kW");axs[0].legend(ncol=3,loc="upper left")
    axs[1].bar(h,np.asarray(q1["charge"])*6,width=1/6,label="充电",color=COLORS[2])
    axs[1].bar(h,-np.asarray(q1["discharge"])*6,width=1/6,label="放电",color=COLORS[1])
    axs[1].set_ylabel("储能功率 / kW");axs[1].legend(ncol=2)
    axs[2].plot(np.arange(145)/6,q1["energy"],color=COLORS[0],label="储电量")
    axs[2].axhline(1200,color="#9CA3AF",ls=":");axs[2].axhline(10800,color="#9CA3AF",ls=":")
    axs[2].set_ylabel("储电量 / kWh");axs[2].set_xlabel("时刻 / h")
    ax=axs[2].twinx();ax.plot(h,data["fixed_price"],color=COLORS[3],alpha=.8,label="电价")
    ax.set_ylabel("电价 / (元/kWh)");ax.spines["right"].set_visible(True)
    axs[2].set_xticks(np.arange(0,25,2));axs[2].set_xlim(0,24)
    for a in axs:a.grid(axis="y",alpha=.18)
    fig.savefig(FIG/"q1_dispatch.pdf");fig.savefig(FIG/"q1_dispatch.png");plt.close(fig)
    fig,axs=plt.subplots(3,1,figsize=(8.1,6),sharex=True,layout="constrained")
    for ax,key,title in zip(axs,["load","pv","price"],["小区负荷 / kW","光伏出力 / kW","电价 / (元/kWh)"]):
        im=ax.imshow(data[key],aspect="auto",origin="lower",extent=(0,24,1,365),cmap="cividis")
        ax.set_ylabel("年内日序");ax.set_title(title,loc="left",fontsize=10)
        fig.colorbar(im,ax=ax,pad=.02)
    axs[-1].set_xlabel("时刻 / h");axs[-1].set_xticks(np.arange(0,25,3))
    fig.savefig(FIG/"annual_inputs.pdf");fig.savefig(FIG/"annual_inputs.png");plt.close(fig)
    r=results["q3"]
    fig,axs=plt.subplots(2,2,figsize=(8.5,6.1),layout="constrained",sharex=True)
    for ax,date in zip(axs.flat,SELECTED):
        i=list(r["dates"]).index(date)
        ax.plot(h,r["plan"][i]*6,color=COLORS[0],lw=1,label="零时计划")
        ax.plot(h,r["adjusted"][i]*6,color=COLORS[1],lw=1,label="最终常规购电")
        ax.fill_between(h,0,r["emergency"][i]*6,color="#B45454",alpha=.6,label="紧急购电")
        for x in (6,12,18):ax.axvline(x,color="#999999",lw=.7,ls=":")
        ax.set_title(date,loc="left");ax.set_xlabel("时刻 / h");ax.set_ylabel("功率 / kW")
        ax.set_xticks(np.arange(0,25,6));ax.set_xlim(0,24);ax.grid(axis="y",alpha=.18)
    handles,labels=axs[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside upper center",ncol=3)
    fig.savefig(FIG/"q3_seasons.pdf");fig.savefig(FIG/"q3_seasons.png");plt.close(fig)
    month=np.array([int(d[5:7]) for d in data["dates"]])
    fig,axs=plt.subplots(2,1,figsize=(8.1,5.5),sharex=True,layout="constrained")
    for j,(case,r) in enumerate(results.items()):
        costs=[r["total_cost"][month==m].sum()/1e4 for m in range(2,13)]
        emer=[r["emergency"][month==m].sum()/1e3 for m in range(2,13)]
        axs[0].plot(range(2,13),costs,"o-",ms=3,label=LABELS[case],color=COLORS[j])
        axs[1].plot(range(2,13),emer,"o-",ms=3,label=LABELS[case],color=COLORS[j])
    axs[0].set_ylabel("月度费用 / 万元");axs[0].legend(ncol=4)
    axs[1].set_ylabel("紧急电量 / 千kWh");axs[1].set_xlabel("月份");axs[1].set_xticks(range(2,13))
    for a in axs:a.grid(axis="y",alpha=.18)
    fig.savefig(FIG/"monthly_results.pdf");fig.savefig(FIG/"monthly_results.png");plt.close(fig)


def experiment_tables():
    path=DATA/"experiments.json"
    if not path.exists():return
    ex=json.loads(path.read_text(encoding="utf-8"))["experiments"]
    groups={
      "ablation.tex":("问题2的逐层对照（评价期334天）",[
        ("q2_no_storage","无储能+场景采购"),("q2_point_forecast","点预测LP"),
        ("q2_sp_expectation","场景期望成本"),("q2_cvar_no_reserve","场景+CVaR"),
        ("q2_proposed","场景+CVaR+参考备用")]),
      "updates.tex":("问题3预报更新与反馈频率的对照",[
        ("q3_midnight_only","仅0时"),("q3_00_06","0、6时"),
        ("q3_00_06_12","0、6、12时"),("q3_proposed","0、6、12、18时")]),
      "sensitivity.tex":("问题3风险与终端参数敏感性",[
        ("q3_proposed",r"主策略：$\kappa=1,\lambda_0=0.1$"),
        ("q3_reserve_0p5",r"备用系数$\kappa=0.5$"),
        ("q3_reserve_1p5",r"备用系数$\kappa=1.5$"),
        ("q3_risk_0",r"风险基数$\lambda_0=0$"),
        ("q3_risk_0p2",r"风险基数$\lambda_0=0.2$"),
        ("q3_terminal_0","取消终端偏离惩罚"),
        ("q3_refund_interpretation","下调退款解释")])}
    for filename,(title,items) in groups.items():
        lines=[r"\begin{table}[htbp]\centering\small\caption{"+title+r"}",
          r"\label{tab:"+filename[:-4]+r"}",r"\begin{tabularx}{\linewidth}{LRRRR}\toprule",
          r"策略 & 总费用/万元 & 紧急量/kWh & 日费CVaR95/元 & 年末电量/kWh\\\midrule"]
        for key,label in items:
            v=ex[key]["summary"]
            lines.append(" & ".join([label,fmt(v["total_cost"]/1e4),fmt(v["emergency_energy"]),
                                     fmt(v["daily_cost_cvar95"]),fmt(v["final_energy"])])+r"\\")
        lines += [r"\bottomrule\end{tabularx}\end{table}"]
        write(filename,"\n".join(lines))
    # Decomposition contrasts same forecast source and settlement within Q3.
    keys=["q3_midnight_only","q3_00_06","q3_00_06_12","q3_proposed"]
    fig,axs=plt.subplots(1,2,figsize=(8.1,3.3),layout="constrained")
    labels=["0时","0/6时","0/6/12时","0/6/12/18时"]
    vals=[ex[k]["summary"]["total_cost"]/1e4 for k in keys]
    axs[0].bar(labels,vals,color=COLORS)
    axs[0].set_ylabel("总费用 / 万元");axs[0].set_ylim(0,1550)
    for i,v in enumerate(vals):axs[0].text(i,v+12,f"{v:.2f}",ha="center",fontsize=8)
    vals=[ex[k]["summary"]["emergency_energy"]/1e3 for k in keys]
    axs[1].bar(labels,vals,color=COLORS);axs[1].set_ylabel("紧急电量 / 千kWh")
    for i,v in enumerate(vals):axs[1].text(i,v+1,f"{v:.2f}",ha="center",fontsize=8)
    for ax in axs:ax.tick_params(axis="x",labelsize=8);ax.grid(axis="y",alpha=.18)
    fig.savefig(FIG/"update_value.pdf");fig.savefig(FIG/"update_value.png");plt.close(fig)


def main():
    data=dict(np.load(DATA/"prepared.npz",allow_pickle=False))
    q1=json.loads((DATA/"q1.json").read_text(encoding="utf-8"))
    results={k:dict(np.load(DATA/f"{k}_full.npz",allow_pickle=False)) for k in LABELS}
    q1table=r"\begin{table}[htbp]\centering\small\caption{问题1指定时段与全天购电结果（电量单位：kWh）}\label{tab:q1-purchase}"+"\n"+purchase_block(q1["plan"],sum(q1["plan"]),q1["cost"])+r"\end{table}"+"\n"
    q1table+=r"\begin{table}[htbp]\centering\small\caption{问题1储能充放电及首末电量（单位：kWh）}\label{tab:q1-battery}"+"\n"+battery_block(np.array(q1["charge"]),np.array(q1["discharge"]),q1["energy"])+r"\end{table}"
    write("q1_results.tex",q1table)
    for case,r in results.items():selected_tables(case,r)
    summaries={case:json.loads((DATA/f"{case}_summary.json").read_text()) for case in LABELS}
    s=[r"\begin{table}[htbp]\centering\small\caption{2025年2月1日至12月31日费用与风险结果}\label{tab:annual}",
       r"\begin{tabularx}{\linewidth}{LRRRR}\toprule",
       "指标 & "+" & ".join(LABELS.values())+r"\\\midrule"]
    metrics=[("计划费用/万元","plan_cost",1e4),("调整费用/万元","adjustment_cost",1e4),
             ("紧急费用/万元","emergency_cost",1e4),("总费用/万元","total_cost",1e4),
             ("紧急电量/万kWh","emergency_energy",1e4),("发生紧急购电天数","emergency_days",1),
             ("日费用CVaR95/万元","daily_cost_cvar95",1e4),("最高日费用/万元","daily_cost_max",1e4),
             ("2月1日期初电量/kWh","initial_energy",1),("12月31日期末电量/kWh","final_energy",1)]
    for label,key,scale in metrics:
        s.append(label+" & "+" & ".join(fmt(summaries[c][key]/scale) for c in LABELS)+r"\\")
    s += [r"\bottomrule\end{tabularx}\end{table}"]
    write("annual_summary.tex","\n".join(s))
    with (DATA/"daily_summary.csv").open("w",encoding="utf-8-sig",newline="") as f:
        fields=["case","date","plan_kwh","adjusted_kwh","emergency_kwh","plan_cost_yuan","adjustment_cost_yuan","emergency_cost_yuan","total_cost_yuan","energy_start_kwh","energy_end_kwh"]
        w=csv.writer(f);w.writerow(fields)
        for case,r in results.items():
            for i in range(31,365):
                w.writerow([case,r["dates"][i],r["plan"][i].sum(),r["adjusted"][i].sum(),r["emergency"][i].sum(),r["plan_cost"][i],r["adjustment_cost"][i],r["emergency_cost"][i],r["total_cost"][i],r["energy"][i,0],r["energy"][i,-1]])
    figures(data,q1,results)
    experiment_tables()
    print("已生成 q1、四个case指定日期表、年度汇总、daily_summary.csv及4组图（PDF+PNG）。")


if __name__=="__main__":main()
