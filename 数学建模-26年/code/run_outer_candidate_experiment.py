"""Isolated outer-procurement experiment; preserves frozen manuscript results.

Run with a Python environment containing NumPy and SciPy:
    python code/run_outer_candidate_experiment.py --workers 4

The candidate menu is fixed before outcomes are examined. Historical replay
holds each candidate contract fixed through midnight. Actual simulation
reselects only at legal updates and carries its own storage state across days.
This development backtest is not an untouched independent test set.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import time

for thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                   "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[thread_var] = "1"

import numpy as np
import scipy
import dispatch_v2 as model
import optimize_dispatch as legacy
from outer_candidate_selector import select_candidate

ROOT = Path(__file__).resolve().parents[1]
CASES = ("q2", "q3", "q4_2", "q4_3")
MARGINS = (0.25, 0.5, 1.0)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [native(v) for v in value]
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(native(value), ensure_ascii=False,
                                    indent=2, allow_nan=False), encoding="utf-8")


def compare_costs(baseline, selected):
    before = float(baseline["total_cost"].sum())
    after = float(selected["total_cost"].sum())
    return {"baseline_bill_yuan": before, "selected_bill_yuan": after,
            "bill_change_yuan": after-before,
            "bill_change_percent": 100*(after/before-1),
            "baseline_emergency_kwh": float(baseline["emergency"].sum()),
            "selected_emergency_kwh": float(selected["emergency"].sum()),
            "baseline_final_energy_kwh": float(baseline["energy"][-1,-1]),
            "selected_final_energy_kwh": float(selected["energy"][-1,-1])}


def simulate_selected(data, case, cfg, start_day, end_day, initial_energy):
    """Change only the outer procurement decision; use original step executor."""
    if cfg.controller != "greedy" or cfg.cross_day or cfg.no_storage:
        raise ValueError("This experiment freezes the existing greedy executor with storage.")
    if cfg.risk_mode != "none" or cfg.reserve_mode != "none":
        raise ValueError("This experiment uses the existing zero-risk, zero-reserve main case.")
    data = model.feedback_data(data, cfg.feedback)
    nd = end_day-start_day
    result = {key: np.zeros((nd,144)) for key in
              ("plan", "adjusted", "charge", "discharge", "emergency", "spill")}
    result["energy"] = np.zeros((nd,145))
    for key in ("plan_cost", "adjustment_cost", "emergency_cost", "total_cost"):
        result[key] = np.zeros(nd)
    result["dates"] = data["dates"][start_day:end_day].copy()
    rolling = case in ("q3", "q4_3")
    dynamic = case in ("q4_2", "q4_3")
    updates = tuple(cfg.updates) if rolling else (0,)
    energy = float(initial_energy)
    decisions = []
    tick = time.perf_counter()
    for ix, day in enumerate(range(start_day, end_day)):
        result["energy"][ix,0] = energy
        effective = np.zeros(144)
        for slot in range(144):
            if slot in updates:
                ns, ps, weights, reserve, risk, base = model.outer_inputs(
                    data, day, slot, case, cfg)
                reach = np.maximum(0., energy-model.EMIN
                                   +model.ETA*model.U*np.arange(1,145-slot))
                reserve = np.minimum(reserve, reach)
                point_net, point_price = model.point_forecast(data, day, slot//36, case)
                choice = select_candidate(
                    ns, ps, weights, energy, base, point_net[slot:], point_price[slot:],
                    reference=None if slot == 0 else result["plan"][ix,slot:].copy(),
                    current_contract=None if slot == 0 else effective[slot:].copy(),
                    reserve=reserve, risk=risk, margin_scales=MARGINS)
                # Current effective contract includes the most recent legal
                # revision. It must not be reconstructed from future executions.
                effective[slot:] = choice["grid"]
                if slot == 0:
                    result["plan"][ix] = effective.copy()
                record = {"date": str(data["dates"][day]), "slot": slot,
                          "hour": slot//6, "initial_energy_kwh": energy,
                          "selected_name": choice["selected_name"],
                          "baseline_score": float(choice["baseline_score"]),
                          "estimated_score_reduction_yuan": float(choice["score_improvement"]),
                          "terminal_beta": float(choice["terminal_beta"]),
                          "generation_seconds": float(choice["generation_seconds"]),
                          "replay_seconds": float(choice["replay_seconds"]),
                          "candidates": [{k: native(v) for k,v in candidate.items()
                                          if k != "grid"}
                                         for candidate in choice["candidates"]]}
                assert record["estimated_score_reduction_yuan"] >= -1e-5
                decisions.append(record)
            # Actual future values are first read at this execution step, after
            # any procurement decision at this slot has already been selected.
            actual_net = (data["load"][day,slot]-data["pv"][day,slot])*model.DT
            grid = float(effective[slot])
            c, d, energy, h, spill = legacy.execute_step(grid, actual_net, energy)
            for key, value in (("adjusted",grid), ("charge",c), ("discharge",d),
                               ("emergency",h), ("spill",spill)):
                result[key][ix,slot] = value
            result["energy"][ix,slot+1] = energy
        prices = data["price"][day] if dynamic else data["fixed_price"]
        original = result["plan"][ix]
        delta = result["adjusted"][ix]-original
        down_rate = -.5 if cfg.cancellation_refund else .5
        result["plan_cost"][ix] = prices@original
        result["adjustment_cost"][ix] = prices@(
            1.5*np.maximum(delta,0)+down_rate*np.maximum(-delta,0))
        result["emergency_cost"][ix] = 5*prices@result["emergency"][ix]
        result["total_cost"][ix] = sum(result[key][ix] for key in
            ("plan_cost", "adjustment_cost", "emergency_cost"))
        if (ix+1) % 30 == 0 or ix == nd-1:
            print(f"{case}/outer-selector: {ix+1}/{nd} days, "
                  f"bill={result['total_cost'][:ix+1].sum():.2f}, "
                  f"elapsed={time.perf_counter()-tick:.1f}s", flush=True)
    model.validate(data, case, result, cfg, start_day)
    assert abs(result["energy"][0,0]-initial_energy) < 1e-7
    assert np.allclose(result["total_cost"], result["plan_cost"]
                       +result["adjustment_cost"]+result["emergency_cost"])
    return result, decisions


def run_case(case, start_date, end_date, output_dir):
    began = time.perf_counter()
    output_dir = Path(output_dir)
    selection = json.loads((ROOT/"data/selection_v2.json").read_text(encoding="utf-8"))
    assert selection.get("frozen")
    cfg = model.Config(**selection["cases"][case]["config"])
    prepared = ROOT/f"data/prepared_v2_{case}.npz"
    primary_path = ROOT/f"data/{case}_v2.npz"
    source_paths = [prepared, primary_path, ROOT/"data/selection_v2.json",
                    ROOT/"code/dispatch_v2.py", ROOT/"code/optimize_dispatch.py",
                    ROOT/"code/outer_candidate_selector.py", Path(__file__).resolve()]
    source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in source_paths}
    data = dict(np.load(prepared, allow_pickle=False))
    primary = dict(np.load(primary_path, allow_pickle=False))
    start_day = int(np.flatnonzero(data["dates"] == start_date)[0])
    end_day = int(np.flatnonzero(data["dates"] == end_date)[0])+1
    if start_day >= end_day:
        raise ValueError("Invalid date interval")
    stored_start = int(np.flatnonzero(primary["dates"] == start_date)[0])
    nd = end_day-start_day
    initial = float(primary["energy"][stored_start,0])
    baseline = model.simulate(data,case,cfg,start_day,end_day,initial,progress=True)
    stored_grid = primary["adjusted"][stored_start:stored_start+nd]
    baseline_validation = {
        "max_grid_difference_kwh": float(np.max(np.abs(baseline["adjusted"]-stored_grid))),
        "bill_difference_yuan": float(baseline["total_cost"].sum()
                                       -primary["total_cost"][stored_start:stored_start+nd].sum()),
        "same_opening_energy": bool(abs(baseline["energy"][0,0]-initial)<1e-7)}
    # Compare against an actual rerun on the installed solver version. Preserve
    # any difference from the delivered arrays instead of silently ignoring it.
    selected, decisions = simulate_selected(data,case,cfg,start_day,end_day,initial)
    comparison = compare_costs(baseline, selected)
    ns, ps, weights, _, _, base = model.outer_inputs(data,end_day-1,0,case,cfg)
    beta = float(base.terminal_scale*np.median(weights@ps))
    comparison["common_final_inventory_beta"] = beta
    for name, result in (("baseline",baseline),("selected",selected)):
        for field in ("plan_cost","adjustment_cost","emergency_cost"):
            comparison[f"{name}_{field}_yuan"] = float(result[field].sum())
        penalty = beta*abs(float(result["energy"][-1,-1])-base.terminal_target)
        comparison[f"{name}_final_inventory_penalty_yuan"] = penalty
        comparison[f"{name}_bill_plus_final_inventory_yuan"] = float(result["total_cost"].sum())+penalty
    comparison["inventory_adjusted_change_yuan"] = (
        comparison["selected_bill_plus_final_inventory_yuan"]
        -comparison["baseline_bill_plus_final_inventory_yuan"])
    months = np.array([str(x)[:7] for x in selected["dates"]])
    monthly = []
    for month in sorted(set(months)):
        mask = months == month
        before = float(baseline["total_cost"][mask].sum())
        after = float(selected["total_cost"][mask].sum())
        monthly.append({"month":month,"baseline_bill_yuan":before,
                        "selected_bill_yuan":after,"bill_change_yuan":after-before,
                        "baseline_emergency_kwh":float(baseline["emergency"][mask].sum()),
                        "selected_emergency_kwh":float(selected["emergency"][mask].sum())})
    reductions = np.array([d["estimated_score_reduction_yuan"] for d in decisions])
    counts = dict(Counter(d["selected_name"] for d in decisions))
    assert source_hashes == {str(p.relative_to(ROOT)): digest(p) for p in source_paths}
    summary = {"case":case,"completed":True,"period":[start_date,end_date],"days":nd,
               "initial_energy_kwh":initial,"frozen_config":asdict(cfg),
               "source_sha256":source_hashes,"baseline_reproduction":baseline_validation,
               "comparison":comparison,"monthly":monthly,"candidate_selection_counts":counts,
               "decision_count":len(decisions),
               "decisions_with_lower_estimated_score":int(np.sum(reductions>1e-6)),
               "mean_estimated_score_reduction_yuan_per_decision":float(reductions.mean()),
               "max_estimated_score_reduction_yuan_per_decision":float(reductions.max()),
               "actual_bill_improved_days":int(np.sum(selected["total_cost"]<baseline["total_cost"]-1e-6)),
               "actual_bill_worsened_days":int(np.sum(selected["total_cost"]>baseline["total_cost"]+1e-6)),
               "elapsed_seconds":time.perf_counter()-began,
               "validation":{"physical_constraints":True,"common_opening_soc":True,
                             "bill_components":True,"score_never_above_baseline":True,
                             "source_inputs_unchanged":True}}
    np.savez_compressed(output_dir/f"{case}_baseline.npz",**baseline)
    np.savez_compressed(output_dir/f"{case}_selected.npz",**selected)
    write_json(output_dir/f"{case}_decisions.json", decisions)
    write_json(output_dir/f"{case}_summary.json", summary)
    print(f"DONE {case}: bill change {comparison['bill_change_yuan']:.2f} yuan "
          f"({comparison['bill_change_percent']:.3f}%), choices={counts}",flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases",nargs="+",choices=CASES,default=list(CASES))
    parser.add_argument("--start-date",default="2025-02-01")
    parser.add_argument("--end-date",default="2025-12-31")
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/outer_candidate_test")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    definition = {
        "experiment":"outer_candidate_replay_selection_v1",
        "period":[args.start_date,args.end_date],"cases":args.cases,
        "candidate_menu":["scenario_lp","point_lp",*[f"margin_{x:g}sigma" for x in MARGINS],
                          "no_adjust at intraday updates only"],
        "margin_definition":"point-forecast net demand plus fixed multiplier of weighted scenario standard deviation per slot; point prices unchanged",
        "candidate_menu_tuning":"predeclared without looking at new experiment outcomes",
        "controller":"existing greedy execute_step; no inner-controller changes",
        "scenario_policy":"same original up-to-28 historical paths for every candidate; Q4 keeps paired price/net-load paths",
        "replay_policy":"fixed remaining-day contract, causal slot-by-slot storage, no later contract/forecast updates in replay",
        "score":"scenario-weighted procurement + adjustment relative original zero-hour plan + fivefold scenario-priced emergency + common beta*abs(replay terminal SOC-6000)",
        "terminal_beta":"terminal_scale times median over remaining slots of scenario-mean price; identical across candidates",
        "no_adjust":"retain current outstanding effective contract, not reset to zero-hour original plan",
        "actual_execution":"only legal outer updates; each strategy carries its own SOC and contracts forward; same frozen forecasts and common opening SOC within each case",
        "actual_bill":"plan + adjustment + emergency only; terminal score excluded from bill and shown separately at evaluation endpoint",
        "endpoint_inventory_comparison":"common beta from lawful final-day midnight scenarios; apply terminal penalty once at evaluation endpoint, not as sum of overlapping replay penalties",
        "in_sample_guarantee":"adding candidates cannot raise minimum score at the same state under the exact same score; no future actual-bill guarantee",
        "interpretation":"development backtest on previously examined evaluation period, not untouched independent validation; different cases have different initial states",
        "python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,
        "completed":False,"results":{}}
    write_json(output/"experiment.json",definition)
    if args.workers == 1:
        for case in args.cases:
            definition["results"][case] = run_case(case,args.start_date,args.end_date,str(output))
            write_json(output/"experiment.json",definition)
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers,len(args.cases))) as pool:
            pending = {pool.submit(run_case,case,args.start_date,args.end_date,str(output)):case
                       for case in args.cases}
            for future in as_completed(pending):
                case = pending[future]
                definition["results"][case] = future.result()
                write_json(output/"experiment.json",definition)
    definition["completed"] = True
    write_json(output/"experiment.json",definition)


if __name__ == "__main__":
    main()
