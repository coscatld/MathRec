"""Four preregistered timing ablations around the retained main/outer models.

Only q3 and q4_3 are varied, each with midnight-only and intraday-only
candidate screening. Unscreened authorized updates still solve the original
scenario LP. This driver never modifies mother-model or earlier result files.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

for thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                   "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[thread_var] = "1"

import numpy as np
import scipy
import dispatch_v2 as model
from outer_timing_policy import simulate_timing

ROOT = Path(__file__).resolve().parents[1]
EARLIER = ROOT / "outputs/outer_candidate_test"
SPECS = tuple((case, mode) for case in ("q3", "q4_3")
              for mode in ("midnight_only", "intraday_only"))


def now():
    return datetime.now(timezone.utc).isoformat()


def native(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k):native(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)):
        return [native(v) for v in value]
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(native(value), ensure_ascii=False,
                                    indent=2, allow_nan=False)+"\n", encoding="utf-8")


def file_record(path):
    path = Path(path)
    content = path.read_bytes()
    return {"path":str(path.relative_to(ROOT)),"bytes":len(content),
            "sha256":hashlib.sha256(content).hexdigest(),"exists":True}


def protected_records():
    paths = set(ROOT.glob("*.tex")) | set(ROOT.glob("*.cls")) | set(ROOT.glob("*.sty"))
    for folder in ("sections", "tables"):
        paths.update((ROOT/folder).rglob("*.tex"))
    for name in ("dispatch_v2.py", "optimize_dispatch.py", "outer_candidate_selector.py",
                 "run_outer_candidate_experiment.py", "check_outer_candidate_selector.py"):
        paths.add(ROOT/"code"/name)
    for folder in (ROOT/"data", EARLIER):
        paths.update(p for p in folder.rglob("*") if p.is_file())
    return {str(p.relative_to(ROOT)):file_record(p) for p in sorted(paths)}


def summarize(result, beta, target):
    total = float(result["total_cost"].sum())
    end = float(result["energy"][-1,-1])
    out = {"actual_bill_yuan":total,
           "plan_cost_yuan":float(result["plan_cost"].sum()),
           "adjustment_cost_yuan":float(result["adjustment_cost"].sum()),
           "emergency_cost_yuan":float(result["emergency_cost"].sum()),
           "emergency_kwh":float(result["emergency"].sum()),
           "initial_energy_kwh":float(result["energy"][0,0]),
           "final_energy_kwh":end,
           "endpoint_beta":beta,"endpoint_target_kwh":target,
           "endpoint_penalty_yuan":beta*abs(end-target),
           "bill_plus_endpoint_penalty_yuan":total+beta*abs(end-target)}
    months = np.array([str(d)[:7] for d in result["dates"]])
    out["monthly"] = [{"month":month,
                       "actual_bill_yuan":float(result["total_cost"][months==month].sum()),
                       "emergency_kwh":float(result["emergency"][months==month].sum())}
                      for month in sorted(set(months))]
    return out


def run_variant(case, mode, output_dir):
    begin = time.perf_counter()
    output_dir = Path(output_dir)
    selection = json.loads((ROOT/"data/selection_v2.json").read_text(encoding="utf-8"))
    assert selection["frozen"]
    cfg = model.Config(**selection["cases"][case]["config"])
    assert cfg.controller == "greedy" and cfg.risk_mode == cfg.reserve_mode == "none"
    assert tuple(cfg.updates) == (0,36,72,108) and not cfg.cross_day
    data = dict(np.load(ROOT/f"data/prepared_v2_{case}.npz",allow_pickle=False))
    primary = dict(np.load(ROOT/f"data/{case}_v2.npz",allow_pickle=False))
    incumbent = dict(np.load(EARLIER/f"{case}_selected.npz",allow_pickle=False))
    old_summary = json.loads((EARLIER/f"{case}_summary.json").read_text(encoding="utf-8"))
    start = int(np.flatnonzero(data["dates"] == "2025-02-01")[0])
    end = int(np.flatnonzero(data["dates"] == "2025-12-31")[0])+1
    initial = float(primary["energy"][0,0])
    assert abs(initial-float(incumbent["energy"][0,0])) < 1e-7
    result, decisions = simulate_timing(data,case,cfg,start,end,initial,
                                        screening_mode=mode,progress=True)
    assert len(result["dates"]) == 334 and len(decisions) == 1336
    assert np.array_equal(result["dates"], primary["dates"])
    assert np.array_equal(result["dates"], incumbent["dates"])
    active = [row for row in decisions if row["screening_active"]]
    expected = 334 if mode == "midnight_only" else 1002
    assert len(active) == expected
    for row in decisions:
        wanted = row["slot"] == 0 if mode == "midnight_only" else row["slot"] > 0
        assert row["screening_active"] == wanted
        if not wanted:
            assert row["selected_name"] == "scenario_lp" and not row["candidates"]
        else:
            choices = row["candidates"]
            chosen = next(c for c in choices if c["name"] == row["selected_name"])
            assert chosen["score"] <= min(c["score"] for c in choices)+1e-7
    beta = old_summary["comparison"]["common_final_inventory_beta"]
    totals = summarize(result,beta,6000.)
    totals.update({"schema_version":"1.0","generated_by":Path(__file__).name,
                   "generated_at":now(),"question_id":case,"mode":mode,
                   "period":["2025-02-01","2025-12-31"],"days":334,
                   "outer_decision_count":len(decisions),"screened_decision_count":len(active),
                   "ordinary_lp_decision_count":len(decisions)-len(active),
                   "screened_selection_counts":dict(Counter(r["selected_name"] for r in active)),
                   "frozen_config":asdict(cfg),"elapsed_seconds":time.perf_counter()-begin,
                   "bill_minus_main_yuan":totals["actual_bill_yuan"]-float(primary["total_cost"].sum()),
                   "bill_minus_all_screening_yuan":totals["actual_bill_yuan"]-float(incumbent["total_cost"].sum()),
                   "actual_improved_days_vs_all":int(np.sum(result["total_cost"]<incumbent["total_cost"]-1e-6)),
                   "actual_worsened_days_vs_all":int(np.sum(result["total_cost"]>incumbent["total_cost"]+1e-6)),
                   "validation":{"physical_constraints":True,"legal_updates_preserved":True,
                                 "screening_schedule":True,"common_initial_state":True,
                                 "bill_components":True,"selected_minima":True}})
    prefix = f"{case}_{mode}"
    np.savez_compressed(output_dir/f"{prefix}.npz",**result)
    write_json(output_dir/f"{prefix}_decisions.json",decisions)
    write_json(output_dir/f"{prefix}_summary.json",totals)
    print(f"DONE {prefix}: bill={totals['actual_bill_yuan']:.2f}, "
          f"vs_all={totals['bill_minus_all_screening_yuan']:+.2f}, "
          f"emergency={totals['emergency_kwh']:.2f}, E={totals['final_energy_kwh']:.2f}",flush=True)
    return prefix, totals


def final_selection(results, output_dir):
    """Apply declared incumbent-first, bill-first development selection rule."""
    table = {}
    for case in ("q2","q3","q4_2","q4_3"):
        old = json.loads((EARLIER/f"{case}_summary.json").read_text(encoding="utf-8"))
        beta = old["comparison"]["common_final_inventory_beta"]
        main_path = ROOT/f"data/{case}_v2.npz"
        main = dict(np.load(main_path,allow_pickle=False))
        choices = {"main": {**summarize(main,beta,6000.),"array_path":str(main_path.relative_to(ROOT))}}
        if case in ("q2","q4_2"):
            table[case] = {"selected_strategy":"main","reason":"user_fixed_day_ahead_main",
                           "selected":choices["main"],"alternatives":choices}
            continue
        all_path = EARLIER/f"{case}_selected.npz"
        enhanced = dict(np.load(all_path,allow_pickle=False))
        choices["all"] = {**summarize(enhanced,beta,6000.),"array_path":str(all_path.relative_to(ROOT))}
        for mode in ("midnight_only","intraday_only"):
            record = results[f"{case}_{mode}"]
            path = Path(output_dir)/f"{case}_{mode}.npz"
            choices[mode] = {**record,"array_path":str(path.relative_to(ROOT))}
        # An exact/numerical tie retains the currently accepted all-update enhancement.
        winner = "all"
        raw_best = min(choices, key=lambda name:choices[name]["actual_bill_yuan"])
        inventory_best = min(choices[name]["bill_plus_endpoint_penalty_yuan"] for name in choices)
        challenger = choices[raw_best]
        if (challenger["actual_bill_yuan"] < choices["all"]["actual_bill_yuan"]-.01
                and challenger["bill_plus_endpoint_penalty_yuan"] <= inventory_best+.01):
            winner = raw_best
        reason = ("strictly_lower_actual_bill_and_consistent_endpoint_ranking"
                  if winner != "all" else "retain_incumbent_no_qualifying_strict_improvement")
        table[case] = {"selected_strategy":winner,"reason":reason,"selected":choices[winner],
                       "alternatives":choices,"raw_bill_best":raw_best,
                       "interaction_yuan":choices["all"]["actual_bill_yuan"]
                       -choices["midnight_only"]["actual_bill_yuan"]
                       -choices["intraday_only"]["actual_bill_yuan"]+choices["main"]["actual_bill_yuan"],
                       "interaction_interpretation":"closed-loop nonadditivity, not a causal attribution"}
    return {"schema_version":"1.0","generated_by":Path(__file__).name,"generated_at":now(),
            "status":"computed_development_selection","mother_model":"main.tex and code/dispatch_v2.py",
            "selection_period":["2025-02-01","2025-12-31"],"numerical_tie_yuan":.01,
            "selection_rule":"retain all-update incumbent unless challenger lowers actual bill by more than one cent and endpoint-adjusted ranking agrees; daytime groups fixed to main by user",
            "evidence_boundary":"strategy selected on a previously examined development period, not independent holdout validation or future guarantee",
            "main_artifacts_replaced":False,"cases":table}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/outer_timing_ablation")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    output = args.output.resolve()
    output.relative_to(ROOT)
    output.mkdir(parents=True,exist_ok=True)
    protected = protected_records()
    code_paths = [Path(__file__).resolve(),ROOT/"code/outer_timing_policy.py"]
    codes = [file_record(p) for p in code_paths]
    began = now()
    report = {"schema_version":"1.0","generated_by":Path(__file__).name,"generated_at":began,
              "experiment":"four_outer_screening_timing_ablations","specs":SPECS,
              "period":["2025-02-01","2025-12-31"],"candidate_menu_unchanged":True,
              "screening_modes":{"midnight_only":"selector at 0; original scenario LP at 6/12/18",
                                 "intraday_only":"original scenario LP at 0; selector at 6/12/18"},
              "all_legal_updates_retained":True,
              "fixed_components":"main forecasts, up to 28 paired historical scenarios, greedy executor, zero risk, zero reserve, original settlement, common replay terminal term and same opening SOC",
              "selection_rule":"actual bill first; keep all-update incumbent on ties within one cent; a challenger also needs consistent endpoint-adjusted ranking",
              "frozen_day_ahead_cases":["q2","q4_2"],
              "retained_incumbent_directory":str(EARLIER.relative_to(ROOT)),
              "evidence_boundary":"development-period strategy selection, not untouched test",
              "protected_inputs":protected,"source_code":codes,
              "python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,
              "completed":False,"results":{}}
    write_json(output/"experiment.json",report)
    with ProcessPoolExecutor(max_workers=min(args.workers,4)) as pool:
        futures = [pool.submit(run_variant,case,mode,str(output)) for case,mode in SPECS]
        for future in as_completed(futures):
            name, result = future.result()
            report["results"][name] = result
            write_json(output/"experiment.json",report)
    assert protected_records() == protected, "Protected mother/earlier artifacts changed"
    assert [file_record(p) for p in code_paths] == codes, "Experiment code changed during run"
    report["completed"] = True
    report["completed_at"] = now()
    report["protected_inputs_unchanged"] = True
    write_json(output/"experiment.json",report)
    selection = final_selection(report["results"],output)
    write_json(output/"final_strategy.json",selection)
    artifacts = [output/"experiment.json",output/"final_strategy.json"]
    for case,mode in SPECS:
        artifacts.extend(output/f"{case}_{mode}{suffix}" for suffix in
                         (".npz","_decisions.json","_summary.json"))
    manifest = {"schema_version":"1.0","generated_by":Path(__file__).name,"generated_at":now(),
                "status":"PASS","question_ids":["q3","q4_3"],
                "run_command":[sys.executable,*sys.argv],"run_exit_code":0,
                "working_directory":str(ROOT),"python":platform.python_version(),
                "platform":platform.platform(),"started_at":began,"completed_at":now(),
                "source_code":codes,"inputs":list(protected.values()),
                "output_artifacts":[file_record(p) for p in artifacts],
                "claim":"four completed timing ablations and development-period strategy choice only"}
    write_json(output/"run_manifest.json",manifest)
    print(json.dumps({case:rec["selected_strategy"] for case,rec in selection["cases"].items()},ensure_ascii=False))


if __name__ == "__main__":
    main()
