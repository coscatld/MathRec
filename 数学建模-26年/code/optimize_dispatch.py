"""C题统一调度模型：确定性LP、残差场景两阶段LP和6小时滚动优化。

所有优化变量采用kWh；输入功率仅在入口乘1/6小时。原始附件只读。
运行: python code/optimize_dispatch.py --case all
程序按日推进，未来真实负荷、光伏和价格仅进入执行器/结算器。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".python_deps"))
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

DT, ETA, EMIN, EMAX, PMAX = 1 / 6, 0.9, 1200., 10800., 5000.


@dataclass
class Settings:
    scenarios: int = 12
    history: int = 28
    alpha: float = 0.90
    risk_scale: float = 0.10
    reserve_scale: float = 1.0
    reserve_cap: float = 2400.
    terminal_scale: float = 1.0
    terminal_target: float = 6000.
    updates: tuple = (0, 36, 72, 108)
    # Literal settlement: paid day-ahead quantity is not refunded on reduction.
    cancellation_refund: bool = False
    deterministic: bool = False
    no_storage: bool = False


def sparse(rows, cols, vals, shape):
    return coo_matrix((vals, (rows, cols)), shape=shape).tocsr()


def reduce_scenarios(net_errors, price_errors, count):
    """确定性k-medoids缩减；保留真实日轨迹及簇频率，不拆散价荷关联。"""
    m = len(net_errors)
    if m <= count:
        return np.arange(m), np.full(m, 1 / m)
    # Block scaling avoids price being overwhelmed by the kWh coordinate.
    blocks = [net_errors / max(float(np.std(net_errors)), 1e-6)]
    if price_errors is not None:
        blocks.append(price_errors / max(float(np.std(price_errors)), 1e-6))
    x = np.concatenate(blocks, axis=1)
    dist = np.sqrt(np.maximum(0, np.sum(x*x, axis=1)[:, None]
                             + np.sum(x*x, axis=1)[None, :] - 2*x@x.T))
    centers = [int(np.argmin(dist.sum(axis=1)))]
    while len(centers) < count:
        dd = dist[:, centers].min(axis=1)
        dd[centers] = -1
        centers.append(int(dd.argmax()))
    for _ in range(8):
        labels = dist[:, centers].argmin(axis=1)
        new = []
        for j, center in enumerate(centers):
            ids = np.flatnonzero(labels == j)
            new.append(int(ids[np.argmin(dist[np.ix_(ids, ids)].sum(axis=1))])
                       if len(ids) else center)
        if new == centers:
            break
        centers = new
    labels = dist[:, centers].argmin(axis=1)
    weights = np.bincount(labels, minlength=len(centers)) / m
    keep = weights > 0
    return np.asarray(centers)[keep], weights[keep]


def solve_lp(net, prices, weights, e0, cfg, *, reference=None,
             reserve=None, risk=0., q1=False):
    """两阶段限制策略类：g/c/d/E为场景共享第一阶段，h_s为第二阶段。

    真实执行时重新按净负荷和实际电量投影，绝不执行某条已知未来轨迹。
    紧急费用CVaR使用整条时序场景的总损失。
    """
    S, n = net.shape
    # g, charge, discharge, end-of-interval E, emergency(S*n), z(S), eta,
    # terminal absolute deviations, adjustment up/down if applicable.
    g = np.arange(n)
    ch = n + np.arange(n)
    dis = 2*n + np.arange(n)
    en = 3*n + np.arange(n)
    emer = 4*n + np.arange(S*n).reshape(S, n)
    z = 4*n + S*n + np.arange(S)
    va = 4*n + S*n + S
    tp, tm = va + 1, va + 2
    total = tm + 1
    if reference is not None:
        up = total + np.arange(n)
        down = total + n + np.arange(n)
        total += 2*n
    c = np.zeros(total)
    mean_price = weights @ prices
    if reference is None:
        c[g] = mean_price
    else:
        c[up] = 1.5 * mean_price
        c[down] = (-0.5 if cfg.cancellation_refund else 0.5) * mean_price
    c[ch] = c[dis] = 1e-7  # Numerical tie-break only, not a billed battery cost.
    c[emer] = weights[:, None] * 5 * prices
    c[z] = risk * weights / (1 - cfg.alpha)
    c[va] = risk
    c[tp] = c[tm] = cfg.terminal_scale * float(np.median(mean_price))
    bounds = [(0, None)] * total
    power_cap = 0 if cfg.no_storage else PMAX * DT
    for i in np.r_[ch, dis]:
        bounds[i] = (0, power_cap)
    if reserve is None:
        reserve = np.zeros(n)
    for t, i in enumerate(en):
        bounds[i] = (EMIN + float(reserve[t]), EMAX)
    if cfg.no_storage:
        for i in en:
            bounds[i] = (e0, e0)
    if q1:
        for i in emer.ravel():
            bounds[i] = (0, 0)
        bounds[en[-1]] = (e0, e0)
    # E[t] - E[t-1] - eta*charge[t] + discharge[t]/eta = 0.
    er, ec, ev, rhs = [], [], [], []
    for t in range(n):
        r = len(rhs)
        er.extend([r]*3); ec.extend([en[t], ch[t], dis[t]])
        ev.extend([1, -ETA, 1/ETA])
        if t:
            er.append(r); ec.append(en[t-1]); ev.append(-1)
        rhs.append(e0 if t == 0 else 0.)
    r = len(rhs)
    er.extend([r]*3); ec.extend([en[-1], tp, tm]); ev.extend([1, -1, 1])
    rhs.append(cfg.terminal_target)
    if reference is not None:
        for t in range(n):
            r = len(rhs)
            er.extend([r]*3); ec.extend([g[t], up[t], down[t]])
            ev.extend([1, -1, 1]); rhs.append(float(reference[t]))
    ur, uc, uv, ub = [], [], [], []
    for s in range(S):
        for t in range(n):
            r = len(ub)
            ur.extend([r]*4); uc.extend([ch[t], dis[t], g[t], emer[s, t]])
            uv.extend([1, -1, -1, -1]); ub.append(-float(net[s, t]))
        r = len(ub)
        ur.extend([r]*(n+2)); uc.extend(emer[s].tolist()+[va,z[s]])
        uv.extend((5*prices[s]).tolist()+[-1,-1]); ub.append(0.)
    result = linprog(c, A_ub=sparse(ur,uc,uv,(len(ub),total)), b_ub=ub,
                     A_eq=sparse(er,ec,ev,(len(rhs),total)), b_eq=rhs,
                     bounds=bounds, method="highs", options={"presolve":True})
    if not result.success:
        raise RuntimeError(f"LP failed: {result.message}; e0={e0}; n={n}")
    x = result.x
    # Any simultaneous charge/discharge can be removed at unchanged E while
    # decreasing demand; positive tie-break makes such a solution dominated.
    simultaneous = np.minimum(x[ch], x[dis]).max(initial=0)
    if simultaneous > 1e-4:
        raise AssertionError(f"LP simultaneous charging: {simultaneous}")
    return {"grid":np.maximum(x[g],0), "charge":x[ch], "discharge":x[dis],
            "energy":np.r_[e0,x[en]], "objective":float(result.fun),
            "dual_energy":result.eqlin.marginals[:n],
            "scenario_emergency":x[emer], "iterations":int(result.nit)}


def execute_step(grid, net, energy, no_storage=False):
    """因果执行器：当前计划电量与实际净负荷之差，经物理边界投影。"""
    cap = 0. if no_storage else PMAX*DT
    if grid >= net:
        charge = min(grid-net, cap, max(0., (EMAX-energy)/ETA))
        discharge = emergency = 0.
        spill = max(0., grid-net-charge)
    else:
        discharge = min(net-grid, cap, max(0., (energy-EMIN)*ETA))
        charge = spill = 0.
        emergency = max(0., net-grid-discharge)
    nxt = energy + ETA*charge - discharge/ETA
    return charge, discharge, nxt, emergency, spill


def forecast_at(data, key, day, issue):
    if key + "_issue" in data:
        return data[key + "_issue"][day,issue]
    return data[key][day]


def scenario_inputs(data, day, start, case, cfg):
    issue = start // 36
    rolling = case in ("q3", "q4_3")
    dynamic = case in ("q4_2", "q4_3")
    load_fc = (forecast_at(data,"forecast_load",day,issue) if rolling
               else data["forecast_load"][day])
    pv_fc = (data["pv_issue_forecast"][day,issue] if rolling
             else data["forecast_pv_history"][day])
    price_fc = (forecast_at(data,"forecast_price",day,issue) if rolling
                else data["forecast_price"][day]) if dynamic else data["fixed_price"]
    base = (load_fc-pv_fc)[start:]*DT
    price = price_fc[start:]
    if day == 0 or cfg.deterministic:
        return base[None,:],price[None,:],np.ones(1),np.zeros(len(base)),0.
    history = np.arange(max(1,day-cfg.history),day)
    if not len(history):
        history = np.array([0])
    errs, perrs = [], []
    for j in history:
        lf = (forecast_at(data,"forecast_load",j,issue) if rolling
              else data["forecast_load"][j])
        pf = data["pv_issue_forecast"][j,issue] if rolling else data["forecast_pv_history"][j]
        errs.append(((data["load"][j]-data["pv"][j])-(lf-pf))[start:]*DT)
        if dynamic:
            cf = forecast_at(data,"forecast_price",j,issue) if rolling else data["forecast_price"][j]
            perrs.append((data["price"][j]-cf)[start:])
    errs = np.asarray(errs)
    pe = np.asarray(perrs) if dynamic else None
    selected, weights = reduce_scenarios(errs,pe,cfg.scenarios)
    ns = base[None,:] + errs[selected]
    ps = (np.maximum(0.01, price[None,:]+pe[selected]) if dynamic
          else np.broadcast_to(price,ns.shape).copy())
    # Uncertainty-to-reserve map: empirical upper 80% one-hour cumulative error.
    # No independence/normality assumption; covariance is retained in trajectories.
    reserve = np.zeros(len(base))
    for t in range(len(base)):
        end = min(t+7,len(base))
        err_sum = errs[:,t+1:end].sum(axis=1)
        reserve[t] = min(cfg.reserve_cap, max(0.,float(np.quantile(err_sum,.8)))/ETA)
    reserve *= cfg.reserve_scale
    # Increase uncertainty sensitivity relative to forecast net-load scale.
    rel = float(np.std(errs,axis=0).mean()/max(np.abs(base).mean(),50.))
    risk = cfg.risk_scale * min(3., rel/.20)
    return ns,ps,weights,reserve,risk


def simulate(data, case, cfg, days=365, *, progress=True):
    rolling = case in ("q3","q4_3")
    dynamic = case in ("q4_2","q4_3")
    arrays = {k:np.zeros((days,144)) for k in
              ("plan","adjusted","charge","discharge","emergency","spill","reserve")}
    arrays["energy"] = np.zeros((days,145))
    for k in ("plan_cost","adjustment_cost","emergency_cost","total_cost","risk","solve_seconds"):
        arrays[k] = np.zeros(days)
    energy = 6000.
    updates = cfg.updates if rolling else (0,)
    for day in range(days):
        tic = time.perf_counter()
        arrays["energy"][day,0] = energy
        effective = None
        for t in range(144):
            if t in updates:
                ns,ps,w,reserve,risk = scenario_inputs(data,day,t,case,cfg)
                # Reserve reaches its target at at most 5000 kW charging from
                # today's observed state. It is never a hard impossible jump.
                reachable = np.maximum(0.,energy-EMIN+ETA*PMAX*DT*np.arange(1,145-t))
                reserve = np.minimum(reserve,reachable)
                solution = solve_lp(ns,ps,w,energy,cfg,
                                    reference=None if t==0 else arrays["plan"][day,t:],
                                    reserve=reserve,risk=risk)
                if t == 0:
                    arrays["plan"][day] = solution["grid"]
                    arrays["risk"][day] = risk
                effective = solution["grid"]
                arrays["reserve"][day,t:] = reserve
                decision_start = t
            grid = float(effective[t-decision_start])
            arrays["adjusted"][day,t] = grid
            net = (data["load"][day,t]-data["pv"][day,t])*DT
            ch,dis,energy,em,spill = execute_step(grid,net,energy,cfg.no_storage)
            arrays["charge"][day,t] = ch
            arrays["discharge"][day,t] = dis
            arrays["energy"][day,t+1] = energy
            arrays["emergency"][day,t] = em
            arrays["spill"][day,t] = spill
        actual_price = data["price"][day] if dynamic else data["fixed_price"]
        plan,adj = arrays["plan"][day],arrays["adjusted"][day]
        arrays["plan_cost"][day] = actual_price@plan
        down_rate = -.5 if cfg.cancellation_refund else .5
        arrays["adjustment_cost"][day] = actual_price@(1.5*np.maximum(adj-plan,0)
                                                          +down_rate*np.maximum(plan-adj,0))
        arrays["emergency_cost"][day] = 5*actual_price@arrays["emergency"][day]
        arrays["total_cost"][day] = sum(arrays[k][day] for k in
                                ("plan_cost","adjustment_cost","emergency_cost"))
        arrays["solve_seconds"][day] = time.perf_counter()-tic
        if progress and ((day+1)%30==0 or day==days-1):
            print(f"{case}: day {day+1}/{days}; cost={arrays['total_cost'][:day+1].sum():.2f}; "
                  f"E={energy:.2f}; seconds/day={arrays['solve_seconds'][:day+1].mean():.3f}",flush=True)
    arrays["dates"] = data["dates"][:days]
    validate(data,case,arrays,cfg)
    return arrays


def validate(data,case,r,cfg):
    n = len(r["plan"])
    net = (data["load"][:n]-data["pv"][:n])*DT
    balance = r["adjusted"]+r["emergency"]+r["discharge"]-r["charge"]-net-r["spill"]
    dynamics = np.diff(r["energy"],axis=1)-ETA*r["charge"]+r["discharge"]/ETA
    assert np.max(np.abs(balance)) < 1e-6
    assert np.max(np.abs(dynamics)) < 1e-6
    assert r["energy"].min() >= EMIN-1e-6 and r["energy"].max() <= EMAX+1e-6
    assert np.max(np.minimum(r["charge"],r["discharge"])) < 1e-6
    assert np.max(r["charge"]) <= PMAX*DT+1e-6
    assert np.max(r["discharge"]) <= PMAX*DT+1e-6
    assert np.max(np.abs(r["energy"][1:,0]-r["energy"][:-1,-1]),initial=0) < 1e-6
    assert np.allclose(r["adjusted"][:,:36],r["plan"][:,:36])
    if case in ("q2","q4_2"):
        assert np.allclose(r["adjusted"],r["plan"])
    if not cfg.cancellation_refund:
        assert np.min(r["adjusted"]-r["plan"]) >= -1e-5


def summary(result, start=31):
    r = {k:v[start:] for k,v in result.items()}
    cost = r["total_cost"]
    # Empirical daily CVaR via exact minimization, allowing fractional tail mass.
    candidates = np.unique(cost)
    cvar = min(z + np.maximum(cost-z,0).mean()/.05 for z in candidates)
    return {"days":len(cost),"total_cost":float(cost.sum()),
            "plan_cost":float(r["plan_cost"].sum()),
            "adjustment_cost":float(r["adjustment_cost"].sum()),
            "emergency_cost":float(r["emergency_cost"].sum()),
            "plan_energy":float(r["plan"].sum()),
            "adjusted_energy":float(r["adjusted"].sum()),
            "emergency_energy":float(r["emergency"].sum()),
            "emergency_intervals":int((r["emergency"]>1e-6).sum()),
            "emergency_days":int((r["emergency"].sum(axis=1)>1e-6).sum()),
            "spill_energy":float(r["spill"].sum()),
            "charge_energy":float(r["charge"].sum()),
            "discharge_energy":float(r["discharge"].sum()),
            "initial_energy":float(r["energy"][0,0]),
            "final_energy":float(r["energy"][-1,-1]),
            "max_actual_grid_power":float((r["adjusted"]+r["emergency"]).max()/DT),
            "daily_cost_cvar95":float(cvar),"daily_cost_max":float(cost.max()),
            "mean_seconds_per_day":float(r["solve_seconds"].mean())}


def run_q1(data):
    cfg = Settings(risk_scale=0,reserve_scale=0,terminal_scale=0)
    net = (data["q1_load"]-data["q1_pv"])*DT
    s = solve_lp(net[None,:],data["fixed_price"][None,:],np.ones(1),6000,cfg,q1=True)
    spill = s["grid"]+s["discharge"]-s["charge"]-net
    cost = float(data["fixed_price"]@s["grid"])
    baseline = float(data["fixed_price"]@np.maximum(net,0))
    return {"plan":s["grid"],"charge":s["charge"],"discharge":s["discharge"],
            "energy":s["energy"],"cost":cost,"spill":spill,
            "no_storage_cost":baseline,"saving_rate":1-cost/baseline,
            "dual_energy":s["dual_energy"]}


def jsonable(obj):
    if isinstance(obj,np.ndarray):return obj.tolist()
    if isinstance(obj,(np.floating,np.integer)):return obj.item()
    if isinstance(obj,dict):return {k:jsonable(v) for k,v in obj.items()}
    if isinstance(obj,(tuple,list)):return [jsonable(v) for v in obj]
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case",default="all",choices=["all","q1","q2","q3","q4_2","q4_3"])
    parser.add_argument("--days",type=int,default=365)
    args=parser.parse_args()
    data=dict(np.load(ROOT/"data/prepared.npz",allow_pickle=False))
    out=ROOT/"data"; out.mkdir(exist_ok=True)
    cfg=Settings()
    q1=run_q1(data)
    (out/"q1.json").write_text(json.dumps(jsonable(q1),ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Q1 cost={q1['cost']:.4f}, no storage={q1['no_storage_cost']:.4f}",flush=True)
    cases=["q2","q3","q4_2","q4_3"] if args.case=="all" else ([] if args.case=="q1" else [args.case])
    for case in cases:
        r=simulate(data,case,cfg,days=args.days)
        np.savez_compressed(out/f"{case}_full.npz",**r)
        if args.days>31:
            (out/f"{case}_summary.json").write_text(json.dumps(summary(r),indent=2),encoding="utf-8")
    if all((out/f"{case}_full.npz").exists() for case in ["q2","q3","q4_2","q4_3"]):
        export={"q1":jsonable(q1),"settings":asdict(cfg),"cases":{}}
        for case in ["q2","q3","q4_2","q4_3"]:
            r=dict(np.load(out/f"{case}_full.npz",allow_pickle=False))
            export["cases"][case]=jsonable({k:v[31:] for k,v in r.items()})
        (out/"results_export.json").write_text(json.dumps(export,ensure_ascii=False,separators=(",",":")),encoding="utf-8")


if __name__=="__main__":
    main()
