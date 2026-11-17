"""Settlement-driven procurement with a causal ten-minute storage controller.

Outer purchases change only at authorized publication times. The inner LP
holds all current-day contracts fixed, observes only the current realization,
and executes one action. It is not a scenario-tree optimal policy.
"""
from __future__ import annotations
import json
import time
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
import optimize_dispatch as old

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"
DT,ETA,EMIN,EMAX,U=1/6,.9,1200.,10800.,5000/6


@dataclass
class Config:
    scenarios: int = 28
    history: int = 28
    risk_mode: str = "none"
    risk_scale: float = .1
    fixed_risk: float = .1
    reserve_mode: str = "none"
    reserve_scale: float = 1.
    fixed_reserve: float = 0.
    controller: str = "lp"
    feedback: str = "full"
    cross_day: bool = False
    terminal_scale: float = 1.
    cancellation_refund: bool = False
    updates: tuple = (0,36,72,108)
    no_storage: bool = False


def historical_forecast(actual, day, name, prior, known_before=None):
    """Forecast a target day with explicitly bounded available history."""
    known=day if known_before is None else min(day,known_before)
    if known<=0:return prior.copy()
    if name=="last_week":ids=[day-7] if 0<=day-7<known else []
    elif name.startswith("weekday_"):
        count=int(name.split("_")[-1]);ids=[day-7*i for i in range(1,count+1) if 0<=day-7*i<known]
    elif name.startswith("mean_"):
        count=int(name.split("_")[-1]);ids=list(range(max(0,known-count),known))
    else:raise ValueError(name)
    if not ids:ids=list(range(max(0,known-7),known))
    return np.mean(actual[ids],axis=0)


def issue_updates(actual, forecast, floor=0.):
    out=np.full((len(actual),4,144),np.nan)
    for day in range(len(actual)):
        out[day,0]=forecast[day]
        for issue in (1,2,3):
            start=36*issue
            bias=np.mean(actual[day,start-12:start]-forecast[day,start-12:start])
            adjust=bias*np.exp(-np.arange(1,145-start)/36)
            out[day,issue,start:]=np.maximum(floor,forecast[day,start:]+
                np.clip(adjust,-.3*forecast[day,start:],.3*forecast[day,start:]))
    return out


def make_forecast_data(raw, load_model="weekday_4", pv_model="mean_7", price_model="weekday_4"):
    d=dict(raw)
    for variable,model,prior,key in [("load",load_model,"q1_load","forecast_load"),
              ("pv",pv_model,"q1_pv","forecast_pv_history"),
              ("price",price_model,"fixed_price","forecast_price")]:
        d[key]=np.stack([historical_forecast(raw[variable],i,model,raw[prior]) for i in range(len(raw[variable]))])
    d["forecast_load_issue"]=issue_updates(raw["load"],d["forecast_load"])
    d["forecast_price_issue"]=issue_updates(raw["price"],d["forecast_price"],1e-6)
    d["chosen_load_model"]=np.array(load_model)
    d["chosen_pv_model"]=np.array(pv_model)
    d["chosen_price_model"]=np.array(price_model)
    return d


def feedback_data(data, mode):
    if mode=="full":return data
    d=dict(data)
    for key in ("forecast_load","forecast_price"):
        d[key+"_issue"]=np.repeat(data[key][:,None,:],4,axis=1)
    if mode=="soc_only":
        d["pv_issue_forecast"]=np.repeat(data["pv_issue_forecast"][:,0:1,:],4,axis=1)
    elif mode!="pv_only":raise ValueError(mode)
    return d


def point_forecast(data,day,issue,case):
    rolling=case in ("q3","q4_3");dynamic=case in ("q4_2","q4_3")
    load=data["forecast_load_issue"][day,issue] if rolling else data["forecast_load"][day]
    pv=data["pv_issue_forecast"][day,issue] if rolling else data["forecast_pv_history"][day]
    price=(data["forecast_price_issue"][day,issue] if rolling else data["forecast_price"][day]) if dynamic else data["fixed_price"]
    return (load-pv)*DT,price.copy()


def outer_inputs(data,day,start,case,cfg):
    base=old.Settings(scenarios=cfg.scenarios,history=cfg.history,risk_scale=cfg.risk_scale,
      reserve_scale=cfg.reserve_scale,terminal_scale=cfg.terminal_scale,
      no_storage=cfg.no_storage,cancellation_refund=cfg.cancellation_refund)
    ns,ps,w,rho,lam=old.scenario_inputs(data,day,start,case,base)
    if cfg.risk_mode=="none":lam=0.
    elif cfg.risk_mode=="fixed":lam=cfg.fixed_risk
    elif cfg.risk_mode!="adaptive":raise ValueError(cfg.risk_mode)
    if cfg.reserve_mode=="none":rho[:]=0.
    elif cfg.reserve_mode=="fixed":
        rho[:]=cfg.fixed_reserve;rho[-1]=0.
    elif cfg.reserve_mode!="adaptive":raise ValueError(cfg.reserve_mode)
    return ns,ps,w,rho,lam,base


@lru_cache(maxsize=300)
def inner_matrices(n,m):
    """Cache fixed sparse coefficients: n committed slots, m advisory slots."""
    H=n+m;N=3*H+m+2
    ch=np.arange(H);dis=H+ch;en=2*H+ch
    grid=3*H+np.arange(m);tp=N-2;tm=N-1
    rr=[];cc=[];vv=[]
    for t in range(H):
        rr.extend([t]*3);cc.extend([int(en[t]),int(ch[t]),int(dis[t])]);vv.extend([1,-ETA,1/ETA])
        if t:rr.append(t);cc.append(int(en[t-1]));vv.append(-1)
    rr.extend([H]*3);cc.extend([int(en[-1]),tp,tm]);vv.extend([1,-1,1])
    eq=coo_matrix((vv,(rr,cc)),shape=(H+1,N)).tocsr()
    rr=[];cc=[];vv=[]
    for j in range(m):
        rr.extend([j]*3);cc.extend([n+j,H+n+j,int(grid[j])]);vv.extend([1,-1,-1])
    ub=coo_matrix((vv,(rr,cc)),shape=(m,N)).tocsr() if m else None
    return H,N,ch,dis,en,grid,tp,tm,eq,ub


def inner_lp(contract, net, prices, energy, terminal_scale=1.,
             next_net=None, next_prices=None, return_full=False):
    """Fixed-contract deterministic remaining-horizon LP; only action 0 executes.

    Emergency purchases serve load, not deliberate battery charging: in
    committed slots charge <= positive surplus, discharge <= positive deficit.
    Future-day grid variables are advisory, discarded after action 0.
    """
    contract=np.asarray(contract);net=np.asarray(net);prices=np.asarray(prices)
    n=len(contract);m=0 if next_net is None else len(next_net)
    H,N,ch,dis,en,grid,tp,tm,Aeq,Aub=inner_matrices(n,m)
    surplus=np.maximum(contract-net,0);deficit=np.maximum(net-contract,0)
    obj=np.zeros(N);obj[ch]=obj[dis]=1e-7
    obj[dis[:n]]-=5*prices
    if m:obj[grid]=next_prices
    obj[tp]=obj[tm]=terminal_scale*float(np.median(prices))
    lower=np.zeros(N);upper=np.full(N,np.inf)
    upper[ch[:n]]=np.minimum(U,surplus)
    upper[dis[:n]]=np.minimum(U,deficit)
    if m:upper[ch[n:]]=U;upper[dis[n:]]=U
    lower[en]=EMIN;upper[en]=EMAX
    rhs=np.zeros(H+1);rhs[0]=energy;rhs[-1]=6000.
    res=linprog(obj,A_ub=Aub,b_ub=-np.asarray(next_net) if m else None,
                A_eq=Aeq,b_eq=rhs,bounds=np.column_stack([lower,upper]),method="highs",
                options={"presolve":True})
    if not res.success:raise RuntimeError(f"Inner LP: {res.message}")
    x=res.x;c=max(0.,float(x[ch[0]]));d=max(0.,float(x[dis[0]]))
    h=max(0.,float(deficit[0]-d));spill=max(0.,float(surplus[0]-c))
    e=energy+ETA*c-d/ETA
    out={"charge":c,"discharge":d,"energy":e,"emergency":h,"spill":spill}
    if return_full:
        out.update(reference_energy=np.r_[energy,x[en]],reference_charge=x[ch],
          reference_discharge=x[dis],reference_emergency=deficit-x[dis[:n]],
          monetary_forecast=float(5*prices@(deficit-x[dis[:n]])+
                    (np.asarray(next_prices)@x[grid] if m else 0.)))
    return out


def next_day_forecast(data,day,issue,case):
    """Use only the issue's already available 24h forecast beyond midnight."""
    m=issue*36
    if m==0:return None,None
    lh=historical_forecast(data["load"],day+1,str(data["chosen_load_model"]),data["q1_load"],known_before=day)[:m]
    # The current issue predicts hours issue*6+1 ... issue*6+24.
    xp=issue*6+np.arange(1,25)
    pv=np.interp(24+np.arange(1,m+1)/6,xp,data["pv_hourly_raw"][day,issue])
    if case=="q4_3":
        ph=historical_forecast(data["price"],day+1,str(data["chosen_price_model"]),data["fixed_price"],known_before=day)[:m]
    else:ph=data["fixed_price"][:m].copy()
    return (lh-pv)*DT,np.maximum(ph,1e-6)


def initial_state(case,day):
    if day==0:return 6000.
    path=ROOT/"versions/v1_before_settlement_revision/data"/f"{case}_full.npz"
    return float(np.load(path)["energy"][day,0])


def simulate(data,case,cfg,start_day=31,end_day=365,initial_energy=None,progress=False):
    data=feedback_data(data,cfg.feedback)
    nd=end_day-start_day
    arrays={k:np.zeros((nd,144)) for k in ("plan","adjusted","charge","discharge",
       "emergency","spill","reserve","forecast_net","forecast_price","risk_trace")}
    arrays["energy"]=np.zeros((nd,145));arrays["reference_energy"]=np.zeros((nd,145))
    for k in ("plan_cost","adjustment_cost","emergency_cost","total_cost","risk","solve_seconds","inner_seconds"):
        arrays[k]=np.zeros(nd)
    arrays["dates"]=data["dates"][start_day:end_day].copy()
    energy=initial_state(case,start_day) if initial_energy is None else initial_energy
    rolling=case in ("q3","q4_3");dynamic=case in ("q4_2","q4_3")
    updates=tuple(cfg.updates) if rolling else (0,)
    for ix,day in enumerate(range(start_day,end_day)):
        tick=time.perf_counter();arrays["energy"][ix,0]=energy
        arrays["reference_energy"][ix,0]=energy
        effective=np.zeros(144);issue=0;nextnet=nextprice=None
        for t in range(144):
            if t in updates:
                issue=t//36
                ns,ps,w,rho,lam,base=outer_inputs(data,day,t,case,cfg)
                reachable=np.maximum(0.,energy-EMIN+ETA*U*np.arange(1,145-t))
                rho=np.minimum(rho,reachable)
                sol=old.solve_lp(ns,ps,w,energy,base,
                        reference=None if t==0 else arrays["plan"][ix,t:],reserve=rho,risk=lam)
                effective[t:]=sol["grid"]
                if t==0:arrays["plan"][ix]=effective;arrays["risk"][ix]=lam
                arrays["reserve"][ix,t:]=rho;arrays["risk_trace"][ix,t:]=lam
                arrays["reference_energy"][ix,t:]=sol["energy"]
                fnet,fprice=point_forecast(data,day,issue,case)
                arrays["forecast_net"][ix,t:]=fnet[t:]
                arrays["forecast_price"][ix,t:]=fprice[t:]
                if cfg.cross_day and rolling:nextnet,nextprice=next_day_forecast(data,day,issue,case)
            actual_net=(data["load"][day,t]-data["pv"][day,t])*DT
            actual_price=data["price"][day,t] if dynamic else data["fixed_price"][t]
            arrays["adjusted"][ix,t]=effective[t]
            if cfg.controller=="greedy" or cfg.no_storage:
                c,d,energy,h,waste=old.execute_step(effective[t],actual_net,energy,cfg.no_storage)
            elif cfg.controller=="lp":
                gap=actual_net-effective[t]
                if (gap>=0 and energy<=EMIN+1e-7) or (gap<=0 and energy>=EMAX-1e-7) or abs(gap)<1e-10:
                    # Both battery actions are forced to zero by current bounds.
                    c=d=0.;h=max(0.,gap);waste=max(0.,-gap)
                else:
                    future_net=fnet[t:].copy();future_net[0]=actual_net
                    future_price=fprice[t:].copy();future_price[0]=actual_price
                    inner_tick=time.perf_counter()
                    action=inner_lp(effective[t:],future_net,future_price,energy,cfg.terminal_scale,
                                    nextnet,nextprice)
                    arrays["inner_seconds"][ix]+=time.perf_counter()-inner_tick
                    c,d,energy,h,waste=(action[k] for k in ("charge","discharge","energy","emergency","spill"))
            else:raise ValueError(cfg.controller)
            for name,value in (("charge",c),("discharge",d),("emergency",h),("spill",waste)):
                arrays[name][ix,t]=value
            arrays["energy"][ix,t+1]=energy
        price=data["price"][day] if dynamic else data["fixed_price"]
        plan=arrays["plan"][ix];delta=arrays["adjusted"][ix]-plan
        arrays["plan_cost"][ix]=price@plan
        arrays["adjustment_cost"][ix]=price@(1.5*np.maximum(delta,0)+
                  (-.5 if cfg.cancellation_refund else .5)*np.maximum(-delta,0))
        arrays["emergency_cost"][ix]=5*price@arrays["emergency"][ix]
        arrays["total_cost"][ix]=sum(arrays[k][ix] for k in ("plan_cost","adjustment_cost","emergency_cost"))
        arrays["solve_seconds"][ix]=time.perf_counter()-tick
        if progress and ((ix+1)%30==0 or ix==nd-1):
            print(f"{case}/{cfg.controller}: {ix+1}/{nd} days, cost {arrays['total_cost'][:ix+1].sum():.2f}, "
                  f"sec/day {arrays['solve_seconds'][:ix+1].mean():.3f}",flush=True)
    validate(data,case,arrays,cfg,start_day)
    return arrays


def validate(data,case,r,cfg,start_day):
    n=len(r["dates"]);net=(data["load"][start_day:start_day+n]-data["pv"][start_day:start_day+n])*DT
    error=r["adjusted"]+r["emergency"]+r["discharge"]-r["charge"]-net-r["spill"]
    assert np.max(np.abs(error))<1e-5
    assert np.max(np.abs(np.diff(r["energy"],axis=1)-ETA*r["charge"]+r["discharge"]/ETA))<1e-5
    assert EMIN-1e-5<=r["energy"].min() and r["energy"].max()<=EMAX+1e-5
    assert max(r["charge"].max(),r["discharge"].max())<=U+1e-5
    assert np.minimum(r["charge"],r["discharge"]).max()<1e-5
    assert np.max(np.abs(r["energy"][1:,0]-r["energy"][:-1,-1]),initial=0)<1e-5
    assert np.allclose(r["adjusted"][:,:36],r["plan"][:,:36])
    if case in ("q2","q4_2"):assert np.allclose(r["adjusted"],r["plan"])
    if not cfg.cancellation_refund:assert np.min(r["adjusted"]-r["plan"])>=-1e-5


def summary_v2(r):
    out=old.summary(r,start=0)
    out.update(mean_realized_risk=float(np.mean(r["risk_trace"])),
        mean_reference_reserve=float(np.mean(r["reserve"])),
        reference_soc_rmse=float(np.sqrt(np.mean((r["reference_energy"]-r["energy"])**2))),
        mean_inner_seconds_per_day=float(np.mean(r["inner_seconds"])))
    return out


def january_matching(data,case):
    cfg=Config(risk_mode="adaptive",reserve_mode="adaptive")
    lam=[];rho=[]
    updates=(0,36,72,108) if case in ("q3","q4_3") else (0,)
    for day in range(14,31):
        for start in updates:
            *_,r,l,b=outer_inputs(data,day,start,case,cfg)
            take=min(36,len(r)) if len(updates)>1 else len(r)
            rho.extend(r[:take]);lam.extend([l]*take)
    return {"fixed_risk":float(np.mean(lam)),"fixed_reserve":float(np.mean(rho)),
            "period":["2025-01-15","2025-01-31"],"matching":"mean of actually covered slots at each outer update"}


def save_result(case,r,cfg):
    np.savez_compressed(DATA/f"{case}_v2.npz",**r)
    (DATA/f"{case}_v2_summary.json").write_text(json.dumps(summary_v2(r),indent=2),encoding="utf-8")


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--benchmark",action="store_true");args=p.parse_args()
    raw=dict(np.load(DATA/"prepared.npz",allow_pickle=False));data=make_forecast_data(raw)
    r=simulate(data,"q3",Config(),start_day=31,end_day=33,progress=True)
    print(json.dumps(summary_v2(r),indent=2))
