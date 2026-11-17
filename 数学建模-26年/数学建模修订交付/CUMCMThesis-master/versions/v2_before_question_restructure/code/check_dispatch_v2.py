"""Numerical output checks and two independent behavioral probes; no paper review."""
import json
from dataclasses import replace
import numpy as np
from dispatch_v2 import DATA, Config, EMIN, ETA, inner_lp, simulate, make_forecast_data


def main():
    report={'checks':{},'scope':'numerical execution and information checks; no LaTeX or paper review'}
    z=inner_lp(np.zeros(2),np.ones(2),np.array([1.,2.]),EMIN+1/ETA,
               terminal_scale=0,return_full=True)
    assert z['discharge']<1e-7 and abs(z['emergency']-1)<1e-7
    assert abs(z['monetary_forecast']-5)<1e-7
    report['checks']['two_slot_energy_value']={'greedy_emergency_bill':10.,
       'lp_emergency_bill':z['monetary_forecast'],'lp_first_emergency':z['emergency'],
       'interpretation':'LP buys one unit now at emergency price5 and preserves one deliverable unit for price10; terminal penalty zero'}
    # Alter only as-yet-unobserved truth and future published forecasts.
    # Rebuild causal historical/bias forecasts, then compare the already executed prefix.
    raw=dict(np.load(DATA/'prepared.npz'));day,cut=200,63
    selection=json.loads((DATA/'selection_v2.json').read_text(encoding='utf-8'))
    altered={k:v.copy() for k,v in raw.items()}
    altered['load'][day,cut:]*=1.3;altered['pv'][day,cut:]*=.3;altered['price'][day,cut:]*=1.7
    altered['pv_issue_forecast'][day,2:]*=.4
    altered['pv_hourly_raw'][day,2:]*=.4
    fc=selection['cases']['q3']['forecasts']
    inputs=[make_forecast_data(x,**fc) for x in [raw,altered]]
    for control,cross in [('greedy',False),('lp',True)]:
        cfg=Config(controller=control,cross_day=cross)
        a,b=[simulate(x,'q3',cfg,start_day=day,end_day=day+1) for x in inputs]
        differences={k:float(np.max(np.abs(a[k][0,:cut]-b[k][0,:cut])))
                     for k in ['plan','adjusted','charge','discharge','emergency']}
        differences['energy']=float(np.max(np.abs(a['energy'][0,:cut+1]-b['energy'][0,:cut+1])))
        assert max(differences.values())<1e-6,differences
        report['checks']['future_truth_perturbation_'+control]={'day_index':day,'first_altered_slot':cut,
            'prefix_max_differences':differences,'cross_day':cross}
    for case in selection['cases']:
        r=dict(np.load(DATA/f'{case}_v2.npz'))
        d=dict(np.load(DATA/f'prepared_v2_{case}.npz'))
        n=(d['load'][31:]-d['pv'][31:])/6
        balance=r['adjusted']+r['discharge']+r['emergency']-r['charge']-r['spill']-n
        state=np.diff(r['energy'],axis=1)-ETA*r['charge']+r['discharge']/ETA
        checks={'balance_max_abs_kwh':float(np.max(np.abs(balance))),
                'state_max_abs_kwh':float(np.max(np.abs(state))),
                'day_continuity_max_abs_kwh':float(np.max(np.abs(r['energy'][1:,0]-r['energy'][:-1,-1]))),
                'cost_component_max_abs_yuan':float(np.max(np.abs(r['total_cost']-r['plan_cost']-r['adjustment_cost']-r['emergency_cost']))),
                'simultaneous_charge_discharge_count':int(np.sum((r['charge']>1e-7)&(r['discharge']>1e-7))),
                'min_energy':float(r['energy'].min()),'max_energy':float(r['energy'].max())}
        assert max(checks[k] for k in list(checks)[:4])<1e-6
        assert checks['simultaneous_charge_discharge_count']==0
        assert checks['min_energy']>=1200-1e-6 and checks['max_energy']<=10800+1e-6
        assert np.max(r['charge'])<=5000/6+1e-6 and np.max(r['discharge'])<=5000/6+1e-6
        report['checks'][case]=checks
    (DATA/'numerical_checks_v2.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
