"""Record genuine k-medoids reduction iterations without changing dispatch results.

The main policy keeps 28 scenarios. This figure diagnoses its 12-scenario
comparison. Shading is across historical windows, not a confidence interval
and not a trace of the LP solver.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from optimize_dispatch import reduce_scenarios
from make_paper_assets import plt, COLORS

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'
FIG=ROOT/'figures/results'


def trace_reduction(errors,count=12):
    x=errors/max(float(np.std(errors)),1e-6)
    dist=np.sqrt(np.maximum(0,np.sum(x*x,axis=1)[:,None]+np.sum(x*x,axis=1)[None,:]-2*x@x.T))
    centers=[int(np.argmin(dist.sum(axis=1)))]
    while len(centers)<count:
        dd=dist[:,centers].min(axis=1);dd[centers]=-1;centers.append(int(dd.argmax()))
    def objective(cs):return float(np.mean(np.min(dist[:,cs],axis=1)))
    objectives=[objective(centers)]
    stable=False
    for iteration in range(1,9):
        labels=dist[:,centers].argmin(axis=1);new=[]
        for k,center in enumerate(centers):
            ids=np.flatnonzero(labels==k)
            new.append(int(ids[np.argmin(dist[np.ix_(ids,ids)].sum(axis=1))]) if len(ids) else center)
        objectives.append(objective(new))
        stable=new==centers;centers=new
        if stable:break
    labels=dist[:,centers].argmin(axis=1)
    weights=np.bincount(labels,minlength=len(centers))/len(errors);keep=weights>0
    expected,w=reduce_scenarios(errors,None,count)
    assert np.array_equal(np.asarray(centers)[keep],expected)
    assert np.max(np.abs(weights[keep]-w))<1e-12
    assert np.max(np.diff(objectives))<1e-10
    return {'objectives':objectives,'relative_objectives':(np.asarray(objectives)/objectives[0]).tolist(),
            'iterations':iteration,'stable':stable,'medoids':expected.tolist(),'weights':w.tolist()}


def main():
    source=DATA/'prepared_v2_q2.npz';data=dict(np.load(source,allow_pickle=False));records=[]
    for day in range(31,365):
        history=np.arange(max(1,day-28),day)
        errors=((data['load'][history]-data['pv'][history])-
                (data['forecast_load'][history]-data['forecast_pv_history'][history]))/6
        rec=trace_reduction(errors);rec['date']=str(data['dates'][day]);records.append(rec)
    maxstep=max(r['iterations'] for r in records)
    curves=np.stack([np.pad(r['relative_objectives'],(0,maxstep+1-len(r['relative_objectives'])),mode='edge') for r in records])
    q=np.quantile(curves,[.1,.5,.9],axis=0)
    report={'diagnostic':'12 representative scenarios from 28 previous days; actual medoid-update iterations',
       'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
       'objective':'mean standardized Euclidean distance from each historical trajectory to its nearest representative',
       'normalization':'each historical window divided by its own initialized objective',
       'band':'pointwise 10th to 90th percentiles across 334 historical windows; not a confidence interval',
       'padding':'after a window terminates, repeat its terminal value solely to align the plot x-axis',
       'not_claimed':'LP solver convergence, global clustering optimality, or economic policy convergence',
       'all_stable':all(r['stable'] for r in records),'max_iterations':maxstep,
       'median_iterations':float(np.median([r['iterations'] for r in records])),
       'median_final_relative_objective':float(np.median(curves[:,-1])),
       'quantiles':{'p10':q[0].tolist(),'median':q[1].tolist(),'p90':q[2].tolist()},'records':records}
    (DATA/'scenario_convergence.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    np.savez_compressed(DATA/'scenario_convergence.npz',relative_objectives=curves,dates=data['dates'][31:],quantiles=q)
    fig,axs=plt.subplots(1,2,figsize=(8.2,3.5),gridspec_kw={'width_ratios':[1.8,1]},layout='constrained')
    steps=np.arange(maxstep+1)
    axs[0].fill_between(steps,q[0],q[2],color=COLORS[0],alpha=.2,label='历史窗口P10–P90')
    axs[0].plot(steps,q[1],'o-',color=COLORS[0],ms=4,lw=1.8,label='历史窗口中位数')
    axs[0].set_xlabel('代表点更新迭代次数');axs[0].set_ylabel('距离目标 / 各窗口初始值')
    axs[0].set_xticks(steps);axs[0].set_title('(a) 场景缩减的真实迭代记录',loc='left',fontsize=10)
    axs[0].legend(fontsize=8,loc='upper right');axs[0].grid(axis='y',alpha=.18)
    counts=np.bincount([r['iterations'] for r in records],minlength=maxstep+1)
    axs[1].bar(steps[1:],counts[1:],color=COLORS[2],width=.65)
    for k in steps[1:]:axs[1].text(k,counts[k]+2,str(counts[k]),ha='center',fontsize=8)
    axs[1].set_xticks(steps[1:]);axs[1].set_xlabel('代表点稳定时的迭代次数');axs[1].set_ylabel('历史窗口个数')
    axs[1].set_ylim(0,counts.max()*1.16);axs[1].grid(axis='y',alpha=.18);axs[1].set_title('(b) 334个窗口的停止分布',loc='left',fontsize=10)
    for ext in ['pdf','png']:fig.savefig(FIG/f'q2_scenario_convergence.{ext}',dpi=300)
    plt.close(fig)
    print(json.dumps({k:v for k,v in report.items() if k not in ['records','quantiles']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
