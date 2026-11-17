"""Frozen-choice, reporting-period ablations for dispatch_v2.

Run only AFTER January-based selection and primary results are frozen:
    py code/experiments_v2.py
    py code/experiments_v2.py --workers 2 --resume

The only outputs are data/experiments_v2.json and data/experiments_v2.npz.
The driver never changes the selected policy or primary schedules. Variants
use identical prepared forecasts and the primary policy's February initial
SOC, then evolve their own causal closed-loop states. No reporting-period
outcome is used for parameter choice or to match fixed/adaptive conservatism.

Expected selection contract (additional metadata is retained via file hash):
    cases[case].config: keyword fields for dispatch_v2.Config
    cases.q2.january_calibration.fixed_risk / fixed_reserve: January-only
        average adaptive values for matched fixed-policy comparisons.
Matching values can alternatively be in cases.q2.matched_parameters or in
top-level january_calibration.q2 / matched_parameters.q2.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, fields
import hashlib
import json
import os
from pathlib import Path
import time

# Parallel experiments use processes; avoid nested numerical thread pools.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "data" / "experiments_v2.json"
OUT_NPZ = ROOT / "data" / "experiments_v2.npz"
CASES = ("q2", "q3")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, (np.integer, np.floating)): return value.item()
    if isinstance(value, dict): return {k:native(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [native(v) for v in value]
    return value


def selected_case(selection, case):
    if case in selection.get("cases", {}):
        record = selection["cases"][case]
    elif case in selection:
        record = selection[case]
    elif case in selection.get("configs", {}):
        record = {"config":selection["configs"][case]}
    else:
        raise KeyError(f"selection_v2.json has no frozen configuration for {case}")
    if "config" not in record:
        if "chosen_config" in record: record = {**record, "config":record["chosen_config"]}
        elif "settings" in record: record = {**record, "config":record["settings"]}
        else: raise KeyError(f"{case} selection record is missing config")
    return record


def matched_values(selection, case):
    record = selected_case(selection, case)
    candidates = [record.get("january_calibration", {}), record.get("matched_parameters", {}),
                  selection.get("january_calibration", {}).get(case, {}),
                  selection.get("matched_parameters", {}).get(case, {}), record]
    result = {}
    for field, aliases in {
        "fixed_risk":("fixed_risk", "january_matched_fixed_risk", "mean_adaptive_risk"),
        "fixed_reserve":("fixed_reserve", "january_matched_fixed_reserve", "mean_adaptive_reserve_kwh")}.items():
        found = next((container[key] for container in candidates for key in aliases if key in container), None)
        if found is None:
            raise KeyError(f"Missing January-only {field} for {case}; reporting-period averages must not be used as a fallback")
        result[field] = float(found)
        if not np.isfinite(result[field]) or result[field] < 0:
            raise ValueError(f"Invalid January matched parameter {case}.{field}={found}")
    return result


def experiment_specs(selection):
    """All comparisons are declared before reading any variant's outcomes."""
    matched = matched_values(selection, "q2")
    specs = []
    for case in CASES:
        for count in (12,28):
            specs.append((f"{case}_scenarios_{count}", case, {"scenarios":count},
                          "historical scenario compression: identical other parameters"))
    clean = {"risk_mode":"none", "reserve_mode":"none"}
    specs.extend([
        ("q2_expectation_only", "q2", clean,
         "economic expectation objective with both optional conservatism modules disabled"),
        ("q2_point_forecast", "q2", {**clean, "__point_forecast_only":True},
         "single deterministic point-forecast trajectory with identical forecast model, February initial SOC and executor"),
        ("q2_no_storage", "q2", {**clean, "no_storage":True},
         "economic scenario procurement with storage actions disabled; same forecasts and initial stored energy, which remains unavailable"),
        ("q2_risk_fixed_only", "q2", {**clean, "risk_mode":"fixed", "fixed_risk":matched["fixed_risk"]},
         "fixed CVaR weight matched to January mean adaptive weight; no reserve"),
        ("q2_risk_adaptive_only", "q2", {**clean, "risk_mode":"adaptive", "risk_scale":.1},
         "adaptive CVaR only; same predetermined .1 risk base used for January matching"),
        ("q2_reserve_fixed_only", "q2", {**clean, "reserve_mode":"fixed", "fixed_reserve":matched["fixed_reserve"]},
         "fixed reserve matched to January mean adaptive reference reserve; no CVaR"),
        ("q2_reserve_adaptive_only", "q2", {**clean, "reserve_mode":"adaptive", "reserve_scale":1.},
         "dynamic reserve only; standalone reserve ablation requested by review"),
        ("q2_risk_reserve_fixed", "q2", {"risk_mode":"fixed", "fixed_risk":matched["fixed_risk"],
                                         "reserve_mode":"fixed", "fixed_reserve":matched["fixed_reserve"]},
         "both January-matched fixed modules; tests combined conservatism"),
        ("q2_risk_reserve_adaptive", "q2", {"risk_mode":"adaptive", "risk_scale":.1,
                                            "reserve_mode":"adaptive", "reserve_scale":1.},
         "both optional adaptive modules; complements expectation, risk-only and reserve-only groups"),
        ("q2_risk_reserve_adaptive_12", "q2", {"scenarios":12, "risk_mode":"adaptive", "risk_scale":.1,
                                               "reserve_mode":"adaptive", "reserve_scale":1.},
         "scenario compression under active CVaR and reserve; compare directly with the otherwise identical 28-scenario adaptive joint group"),
        ("q3_midnight_only", "q3", {"updates":(0,), "cross_day":False},
         "midnight forecast and purchase only; no later outer reoptimization, with the same physical executor"),
        ("q3_refund_interpretation", "q3", {"cancellation_refund":True},
         "alternative settlement sensitivity: a reduction earns a net half-price refund; changed market rule, never a policy-selection candidate"),
        ("q3_feedback_soc_only", "q3", {"feedback":"soc_only", "cross_day":False},
         "update actual SOC and reoptimize while retaining midnight load, PV and price forecasts"),
        ("q3_feedback_pv_only", "q3", {"feedback":"pv_only", "cross_day":False},
         "update actual SOC and newly published PV forecasts; retain midnight load and price forecasts"),
        ("q3_feedback_full", "q3", {"feedback":"full", "cross_day":False},
         "SOC, newly published PV and causal load/price corrections under the full legal information set"),
        ("q3_controller_greedy", "q3", {"controller":"greedy"},
         "same outer-policy settings; greedy storage controller changes subsequent observed SOC and therefore later outer decisions"),
        ("q3_controller_lp", "q3", {"controller":"lp"},
         "same outer-policy settings with the fixed-contract remaining-horizon LP storage controller"),
        ("q3_cross_day_false", "q3", {"controller":"lp", "cross_day":False},
         "truncate remaining horizon at midnight with the same terminal-penalty coefficient"),
        ("q3_cross_day_true", "q3", {"controller":"lp", "cross_day":True},
         "use legally available next-day forecasts only for continuation value, not advance purchase execution"),
    ])
    return specs, matched


def extended_summary(result, model):
    out = dict(model.summary_v2(result))
    delta = result["adjusted"]-result["plan"]
    out.update({
        "evaluation_initial_energy":float(result["energy"][0,0]),
        "evaluation_final_energy":float(result["energy"][-1,-1]),
        "stored_energy_change":float(result["energy"][-1,-1]-result["energy"][0,0]),
        "adjustment_up_energy":float(np.maximum(delta,0).sum()),
        "adjustment_down_energy":float(np.maximum(-delta,0).sum()),
        "mean_realized_risk_weight":float(np.mean(result.get("risk_trace",result.get("risk")))) if "risk" in result or "risk_trace" in result else None,
        "mean_realized_reference_reserve_kwh":float(np.mean(result["reserve"])) if "reserve" in result else None,
        "recorded_solve_seconds":float(np.sum(result["solve_seconds"])) if "solve_seconds" in result else None,
    })
    return native(out)


def daily_arrays(result):
    out = {"dates":result["dates"], "initial_energy":result["energy"][:,0],
           "final_energy":result["energy"][:,-1]}
    for field in ("total_cost", "plan_cost", "adjustment_cost", "emergency_cost", "risk", "solve_seconds"):
        if field in result:
            arr = result[field]
            out[field] = arr.mean(axis=tuple(range(1,arr.ndim))) if arr.ndim>1 else arr
    for field in ("plan", "adjusted", "emergency", "spill", "charge", "discharge"):
        if field in result: out[f"{field}_energy"] = result[field].sum(axis=1)
    if "reserve" in result: out["reference_reserve_mean"] = result["reserve"].mean(axis=1)
    if "risk_trace" in result: out["realized_risk_weight_mean"] = result["risk_trace"].mean(axis=1)
    return out


def run_one(job):
    import dispatch_v2 as model
    name, case, settings, initial_energy, start, end, point_only = job
    inputs = dict(np.load(ROOT / "data" / f"prepared_v2_{case}.npz",allow_pickle=False))
    begin = time.perf_counter()
    original_outer = model.outer_inputs
    if point_only:
        # This isolated worker-local adapter changes only scenario generation.
        # It never overwrites observed load/PV, so execution still bills against
        # the original actual data; restore it before the worker's next job.
        def point_outer(data,day,update,case,cfg):
            ns,ps,w,rho,lam,base = original_outer(data,day,update,case,cfg)
            net,price = model.point_forecast(data,day,update//36,case)
            return net[None,update:],price[None,update:],np.ones(1),np.zeros(144-update),0.,base
        model.outer_inputs = point_outer
    try:
        result = model.simulate(inputs, case, model.Config(**settings), start_day=start,
                                end_day=end, initial_energy=initial_energy, progress=False)
    finally:
        model.outer_inputs = original_outer
    if len(result["plan"]) != end-start:
        raise AssertionError(f"{name}: wrong evaluation length")
    if abs(float(result["energy"][0,0])-initial_energy)>1e-6:
        raise AssertionError(f"{name}: unequal opening SOC")
    return name, extended_summary(result, model), daily_arrays(result), time.perf_counter()-begin


def update_differences(report):
    differences = {}
    for name, entry in report["experiments"].items():
        reference = f"{entry['case']}_primary"
        if name==reference or entry.get("status")!="complete": continue
        current = entry["summary"]
        base = report["experiments"][reference]["summary"]
        differences[name] = {"reference":reference,
            "cost_minus_reference_yuan":current["total_cost"]-base["total_cost"],
            "cost_percent_vs_reference":100*(current["total_cost"]/base["total_cost"]-1),
            "emergency_kwh_minus_reference":current["emergency_energy"]-base["emergency_energy"],
            "daily_total_cost_cvar95_minus_reference":current["daily_cost_cvar95"]-base["daily_cost_cvar95"],
            "final_energy_minus_reference_kwh":current["evaluation_final_energy"]-base["evaluation_final_energy"]}
    report["differences_vs_frozen_primary"] = differences
    paired = {}
    pairs = [
        ("q2_scenario_compression", "q2_scenarios_28", "q2_scenarios_12"),
        ("q3_scenario_compression", "q3_scenarios_28", "q3_scenarios_12"),
        ("q2_scenario_compression_with_cvar", "q2_risk_reserve_adaptive", "q2_risk_reserve_adaptive_12"),
        ("q2_risk_adaptation", "q2_risk_fixed_only", "q2_risk_adaptive_only"),
        ("q2_reserve_adaptation", "q2_reserve_fixed_only", "q2_reserve_adaptive_only"),
        ("q2_joint_adaptation", "q2_risk_reserve_fixed", "q2_risk_reserve_adaptive"),
        ("q3_soc_reoptimization", "q3_midnight_only", "q3_feedback_soc_only"),
        ("q3_settlement_interpretation", "q3_primary", "q3_refund_interpretation"),
        ("q3_new_pv_information", "q3_feedback_soc_only", "q3_feedback_pv_only"),
        ("q3_load_bias_feedback", "q3_feedback_pv_only", "q3_feedback_full"),
        ("q3_inner_controller", "q3_controller_greedy", "q3_controller_lp"),
        ("q3_cross_midnight_information", "q3_cross_day_false", "q3_cross_day_true"),
    ]
    for label,reference,current in pairs:
        if any(report["experiments"].get(name,{}).get("status")!="complete" for name in (reference,current)):
            continue
        base = report["experiments"][reference]["summary"]
        variant = report["experiments"][current]["summary"]
        paired[label] = {"reference":reference,"variant":current,
            "definition":"signed variant minus reference; negative cost means the variant is cheaper",
            "cost_difference_yuan":variant["total_cost"]-base["total_cost"],
            "cost_percent_difference":100*(variant["total_cost"]/base["total_cost"]-1),
            "emergency_difference_kwh":variant["emergency_energy"]-base["emergency_energy"],
            "daily_total_cost_cvar95_difference_yuan":variant["daily_cost_cvar95"]-base["daily_cost_cvar95"],
            "final_energy_difference_kwh":variant["evaluation_final_energy"]-base["evaluation_final_energy"],
            "reference_mean_risk_weight":base["mean_realized_risk_weight"],
            "variant_mean_risk_weight":variant["mean_realized_risk_weight"],
            "reference_mean_reference_reserve_kwh":base["mean_realized_reference_reserve_kwh"],
            "variant_mean_reference_reserve_kwh":variant["mean_realized_reference_reserve_kwh"]}
    report["controlled_pair_comparisons"] = paired
    for kind in ("fixed","adaptive"):
        names = ["q2_expectation_only",f"q2_risk_{kind}_only",f"q2_reserve_{kind}_only",f"q2_risk_reserve_{kind}"]
        if all(report["experiments"].get(name,{}).get("status")=="complete" for name in names):
            e,r,s,j = [report["experiments"][name]["summary"] for name in names]
            report.setdefault("module_interactions",{})[kind] = {
                "definition":"actual joint cost minus risk-only cost minus reserve-only cost plus expectation-only cost; descriptive closed-loop interaction, not an independently causal estimate",
                "cost_difference_in_differences_yuan":j["total_cost"]-r["total_cost"]-s["total_cost"]+e["total_cost"]}


def save(report,daily):
    update_differences(report)
    OUT_JSON.write_text(json.dumps(native(report),ensure_ascii=False,indent=2),encoding="utf-8")
    np.savez_compressed(OUT_NPZ,**daily)


def run(args):
    import dispatch_v2 as model
    selection_path = ROOT / "data" / "selection_v2.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not selection.get("frozen"):
        raise RuntimeError("January selection is not frozen; experiments must wait for selection_v2.json frozen=true")
    specs, matched = experiment_specs(selection)
    if args.names:
        unknown = set(args.names)-{s[0] for s in specs}
        if unknown: raise ValueError(f"Unknown variants: {sorted(unknown)}")
        specs = [s for s in specs if s[0] in args.names]
    sources = {"selection_v2.json":digest(selection_path), "dispatch_v2.py":digest(ROOT/"code"/"dispatch_v2.py")}
    for case in CASES:
        for name in (f"prepared_v2_{case}.npz",f"{case}_v2.npz"):
            sources[name] = digest(ROOT/"data"/name)
    report = {"selection_is_frozen":True, "evaluation_start_day":31,"evaluation_end_day_exclusive":365,
              "sources_sha256":sources, "january_matched_parameters_q2":matched,
              "fairness_notes":[
                  "This is a diagnostic comparison after January-only forecast/policy choices are frozen; reporting outcomes never revise the selected primary policy.",
                  "All variants within a case share the exact primary opening SOC and prepared forecast arrays; later actual SOC and purchase decisions evolve causally and can differ.",
                  "Fixed/adaptive parameters are matched using January values supplied in selection_v2.json, never using February-December outcomes. Equal January means do not guarantee equal realized reporting-period means; both are reported.",
                  "Reference reserve is an optimization setting; its reported mean is not a claim that the actual controller always preserves that exact energy margin.",
                  "Cost differences are actual bills, exclude CVaR and terminal objective penalties, and must be read with terminal SOC differences.",
                  "Timing is measured for this experiment run; reused baseline timings come from the original primary run and are not controlled solver benchmarks.",
                  "The static outer stochastic model still approximates causal future control; the inner LP does not make it an exactly policy-consistent scenario tree."],
              "experiments":{}, "completed":False}
    daily = {}
    if args.resume and OUT_JSON.exists() and OUT_NPZ.exists():
        old = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        if old.get("sources_sha256") != sources:
            raise RuntimeError("Cannot resume: selection, model, prepared inputs or primary arrays changed")
        report = old
        daily = dict(np.load(OUT_NPZ,allow_pickle=False))
        report["completed"] = False
    jobs, aliases, config_cache = [], {}, {}
    allowed = {f.name for f in fields(model.Config)}
    primary_configs, primary_initial = {}, {}
    for case in CASES:
        config_values = selected_case(selection,case)["config"]
        invalid = set(config_values)-allowed
        if invalid: raise ValueError(f"Unknown Config fields in frozen {case} selection: {invalid}")
        config = asdict(model.Config(**config_values))
        primary_configs[case] = config
        result = dict(np.load(ROOT/"data"/f"{case}_v2.npz",allow_pickle=False))
        if len(result["plan"])!=334:
            raise AssertionError(f"Primary {case}_v2.npz must contain Feb-Dec only")
        primary_initial[case] = float(result["energy"][0,0])
        name = f"{case}_primary"
        report["experiments"][name] = {"case":case,"config":config,"source":f"{case}_v2.npz",
                                       "status":"complete","reused_primary":True,
                                       "summary":extended_summary(result,model)}
        for field,value in daily_arrays(result).items(): daily[f"{name}__{field}"] = value
        config_cache[(case,json.dumps(config,sort_keys=True),False)] = name
    for name,case,overrides,reason in specs:
        point_only = bool(overrides.get("__point_forecast_only",False))
        config = {**primary_configs[case],**{k:v for k,v in overrides.items() if not k.startswith("__")}}
        if report["experiments"].get(name,{}).get("status")=="complete":
            config_cache[(case,json.dumps(config,sort_keys=True),point_only)] = name
            continue
        report["experiments"][name] = {"case":case,"config":config,"overrides":overrides,
                                       "scenario_model":"point_forecast" if point_only else "historical_scenarios",
                                       "comparison_purpose":reason,"status":"pending"}
        key = (case,json.dumps(config,sort_keys=True),point_only)
        if key in config_cache:
            aliases[name] = config_cache[key]
        else:
            config_cache[key] = name
            jobs.append((name,case,config,primary_initial[case],31,365,point_only))
    print(f"Frozen experiment plan: {len(specs)} named variants, {len(jobs)} unique simulations, {len(aliases)} aliases",flush=True)
    if args.dry_run:
        for name,case,overrides,reason in specs: print(name,case,overrides,flush=True)
        return
    save(report,daily)
    started = time.perf_counter()
    def collect(answer):
        name,summary,values,elapsed = answer
        entry = report["experiments"][name]
        entry.update(status="complete",summary=summary,run_seconds=elapsed,reused_primary=False)
        for field,value in values.items(): daily[f"{name}__{field}"] = value
        save(report,daily)
        print(f"DONE {name}: cost={summary['total_cost']:.2f}, emergency={summary['emergency_energy']:.2f}, seconds={elapsed:.1f}",flush=True)
    if args.workers==1:
        for job in jobs:
            print(f"START {job[0]}",flush=True)
            collect(run_one(job))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            pending = {pool.submit(run_one,job):job[0] for job in jobs}
            for future in as_completed(pending): collect(future.result())
    # Resolve aliases only after their original simulations are available.
    for name,reference in aliases.items():
        entry = report["experiments"][name]
        source = report["experiments"][reference]
        if source.get("status")!="complete": raise RuntimeError(f"Missing alias source {reference}")
        entry.update(status="complete",summary=source["summary"],reused_identical_config=reference,
                     run_seconds=0., reused_primary=reference.endswith("_primary"))
        for key,value in list(daily.items()):
            if key.startswith(reference+"__"): daily[name+key[len(reference):]] = value
    report["completed"] = True
    report["last_run_elapsed_seconds"] = time.perf_counter()-started
    save(report,daily)
    print(f"Saved {OUT_JSON.name} and {OUT_NPZ.name}",flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,choices=(1,2),default=1)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--names",nargs="*")
    run(parser.parse_args())
