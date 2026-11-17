"""Replay v1 historical scenarios through its actual causal storage executor.

Reads only the frozen versions/v1_before_settlement_revision implementation,
prepared inputs and actual-state trajectories. Re-solves each legal outer
decision and fixes that purchase schedule for all scenario replays. This is a
planning-versus-execution diagnostic, NOT a realized annual optimality gap:
rolling remaining-day windows overlap and must never be added into a bill.

Run: py code/diagnose_execution_gap.py
Outputs: data/execution_gap_v1.{json,csv,npz}; no workbook or paper mutation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "versions" / "v1_before_settlement_revision"
CASES = ("q2", "q3", "q4_2", "q4_3")


def frozen_model():
    path = FROZEN / "code" / "optimize_dispatch.py"
    spec = importlib.util.spec_from_file_location("frozen_v1_dispatch", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def distribution(values):
    x = np.asarray(values, float)
    return {"count": len(x), "min": float(x.min()), "mean": float(x.mean()),
            "median": float(np.median(x)), "p05": float(np.quantile(x, .05)),
            "p95": float(np.quantile(x, .95)), "max": float(x.max()),
            "negative_fraction": float(np.mean(x < -1e-6)),
            "positive_fraction": float(np.mean(x > 1e-6))}


def summarize(rows):
    return {key: distribution([r[key] for r in rows]) for key in (
        "delta_exec_yuan", "planning_expected_billed_cost_yuan",
        "replay_expected_billed_cost_yuan", "planning_expected_emergency_cost_yuan",
        "replay_expected_emergency_cost_yuan", "soc_weighted_rmse_kwh",
        "soc_max_abs_kwh", "soc_expected_terminal_difference_kwh")}


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=365)
    args = parser.parse_args()
    model = frozen_model()
    cfg = model.Settings()
    data = dict(np.load(FROZEN / "data" / "prepared.npz", allow_pickle=False))
    days = min(args.days, len(data["load"]))
    rows, examples, profiles, all_stats = [], {}, {}, {}
    tic = time.perf_counter()
    for case in CASES:
        saved = dict(np.load(FROZEN / "data" / f"{case}_full.npz", allow_pickle=False))
        updates = cfg.updates if case in ("q3", "q4_3") else (0,)
        # These profiles average over decisions sharing an update time and
        # over each decision's weighted scenarios; time 0 is the initial SOC.
        profile_sum_sq = np.zeros((4, 145))
        profile_sum_signed = np.zeros((4, 145))
        profile_sum_abs = np.zeros((4, 145))
        profile_count = np.zeros((4, 145), dtype=int)
        best_gap, best_soc = -np.inf, -np.inf
        for day in range(days):
            for start in updates:
                e0 = float(saved["energy"][day, start])
                ns, ps, weights, reserve, risk = model.scenario_inputs(data, day, start, case, cfg)
                reachable = np.maximum(0., e0-model.EMIN+model.ETA*model.PMAX*model.DT*np.arange(1, 145-start))
                reserve = np.minimum(reserve, reachable)
                reference = None if start == 0 else saved["plan"][day, start:]
                sol = model.solve_lp(ns, ps, weights, e0, cfg, reference=reference, reserve=reserve, risk=risk)
                grid = sol["grid"]
                scenarios, n = ns.shape
                replay_energy = np.zeros((scenarios, n+1))
                replay_emergency = np.zeros((scenarios, n))
                replay_charge = np.zeros((scenarios, n))
                replay_discharge = np.zeros((scenarios, n))
                replay_energy[:, 0] = e0
                for s in range(scenarios):
                    for t in range(n):
                        ch, dis, en, em, spill = model.execute_step(float(grid[t]), float(ns[s,t]), float(replay_energy[s,t]), cfg.no_storage)
                        replay_energy[s,t+1] = en
                        replay_emergency[s,t] = em
                        replay_charge[s,t], replay_discharge[s,t] = ch, dis
                # All costs below use true billing items ONLY. The initial
                # purchase and adjustment bill is shared by the two controls.
                if reference is None:
                    paid_quantity = grid
                else:
                    paid_quantity = (reference + 1.5*np.maximum(grid-reference,0)
                                     + .5*np.maximum(reference-grid,0))
                common_cost = ps @ paid_quantity
                planned_em_cost = (5*ps*sol["scenario_emergency"]).sum(axis=1)
                replay_em_cost = (5*ps*replay_emergency).sum(axis=1)
                planned_cost = common_cost + planned_em_cost
                replay_cost = common_cost + replay_em_cost
                errors = replay_energy-sol["energy"][None,:]
                delta = float(weights @ (replay_cost-planned_cost))
                # The old rolling simulation commits only until the next
                # decision; validate that this frozen re-solve recreates it.
                committed_n = min(36 if len(updates)>1 else 144, n)
                saved_grid = saved["adjusted"][day, start:start+committed_n]
                grid_deviation = float(np.max(np.abs(grid[:committed_n]-saved_grid)))
                if grid_deviation > 1e-4:
                    raise AssertionError(f"Frozen replay not reproduced: {case}, day={day}, start={start}, delta={grid_deviation}")
                soc_rmse = float(np.sqrt(weights @ np.mean(errors[:,1:]**2,axis=1)))
                soc_max = float(np.max(np.abs(errors)))
                row = {"case":case, "date":str(data["dates"][day]), "day_index":day,
                       "reporting_period":int(day>=31), "update_hour":start//6,
                       "physical_start_index":start, "horizon_intervals":n,
                       "scenarios":scenarios, "actual_initial_soc_kwh":e0,
                       "planning_expected_billed_cost_yuan":float(weights@planned_cost),
                       "replay_expected_billed_cost_yuan":float(weights@replay_cost),
                       "delta_exec_yuan":delta,
                       "planning_expected_emergency_cost_yuan":float(weights@planned_em_cost),
                       "replay_expected_emergency_cost_yuan":float(weights@replay_em_cost),
                       "soc_weighted_rmse_kwh":soc_rmse, "soc_max_abs_kwh":soc_max,
                       "soc_expected_terminal_difference_kwh":float(weights@errors[:,-1]),
                       "recreated_committed_grid_max_abs_kwh":grid_deviation}
                rows.append(row)
                if day>=31:
                    issue = start//36
                    profile_sum_sq[issue,start:] += weights@(errors**2)
                    profile_sum_signed[issue,start:] += weights@errors
                    profile_sum_abs[issue,start:] += weights@np.abs(errors)
                    profile_count[issue,start:] += 1
                    for label, value, best in (("largest_positive_gap",delta,best_gap), ("largest_soc_gap",soc_max,best_soc)):
                        if value > best:
                            key = f"{case}_{label}"
                            examples[key] = {"metadata":row.copy(), "weights":weights.copy(),
                                "net":ns.copy(), "price":ps.copy(), "grid":grid.copy(),
                                "reference_soc":sol["energy"].copy(), "replay_soc":replay_energy.copy(),
                                "reference_charge":sol["charge"].copy(), "reference_discharge":sol["discharge"].copy(),
                                "replay_charge":replay_charge.copy(), "replay_discharge":replay_discharge.copy(),
                                "planning_emergency":sol["scenario_emergency"].copy(), "replay_emergency":replay_emergency.copy(),
                                "planning_scenario_billed_cost":planned_cost.copy(), "replay_scenario_billed_cost":replay_cost.copy()}
                            if label == "largest_positive_gap": best_gap = value
                            else: best_soc = value
            if (day+1)%60 == 0 or day+1 == days:
                print(f"v1 diagnostic {case}: {day+1}/{days}; elapsed {time.perf_counter()-tic:.1f}s",flush=True)
        denom = np.maximum(profile_count,1)
        profiles[f"{case}_soc_profile_rmse_kwh"] = np.sqrt(profile_sum_sq/denom)
        profiles[f"{case}_soc_profile_mean_difference_kwh"] = profile_sum_signed/denom
        profiles[f"{case}_soc_profile_mean_abs_kwh"] = profile_sum_abs/denom
        profiles[f"{case}_soc_profile_decision_count"] = profile_count
        case_rows = [r for r in rows if r["case"]==case]
        all_stats[case] = {}
        for period, select in (("all_available_days",case_rows), ("reporting_feb_to_dec",[r for r in case_rows if r["day_index"]>=31])):
            if not select: continue
            all_stats[case][period] = {"all_updates":summarize(select), "by_update_hour":{}}
            for start in updates:
                all_stats[case][period]["by_update_hour"][str(start//6)] = summarize([r for r in select if r["physical_start_index"]==start])
    out = ROOT / "data"
    out.mkdir(exist_ok=True)
    with (out / "execution_gap_v1.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    example_metadata = {}
    for key, example in examples.items():
        example_metadata[key] = example.pop("metadata")
        for name, values in example.items():
            profiles[f"{key}_{name}"] = values
    np.savez_compressed(out / "execution_gap_v1.npz", **profiles)
    report = {
        "diagnostic": "Frozen v1 planning versus causal greedy execution on identical historical scenarios",
        "definition": "Delta_exec = sum_s pi_s (C_replay,s - C_planning,s), all costs are ordinary purchase + adjustment + five-times-price emergency bill",
        "exclusions": ["CVaR weight/penalty", "terminal inventory penalty", "numerical charge/discharge tie-break"],
        "interpretation": "Decision-point diagnostic on remaining-day windows. Windows overlap in rolling cases; do NOT sum these deltas into annual realized cost or call them global optimality gaps. Negative differences are possible because the greedy executor uses contemporaneous scenario demand whereas planning storage actions are shared across scenarios.",
        "replay_information": "Purchase schedule and initial observed SOC fixed per decision; each scenario is replayed causally with execute_step(grid_t, net_t, E_t). No future demand/price is provided to execute_step.",
        "time_mapping": "physical_start_index 0/36/72/108 = 00:00/06:00/12:00/18:00; SOC profiles at boundaries 0..144. Template labels are separate and do not change physical indexing.",
        "period": {"total_days":days, "reporting_start_day_index":31},
        "source_sha256":{name:file_sha256(FROZEN/name) for name in ["code/optimize_dispatch.py", "data/prepared.npz"]+[f"data/{c}_full.npz" for c in CASES]},
        "settings":model.asdict(cfg),
        "max_recreated_committed_grid_deviation_kwh":max(r["recreated_committed_grid_max_abs_kwh"] for r in rows),
        "summary":all_stats, "examples":example_metadata,
        "npz_profiles": "Shape (4,145): update issue 00/06/12/18 by physical SOC boundary. Profiles use reporting days only. Unused/unreachable entries have zero decision_count and must be masked.",
        "elapsed_seconds":time.perf_counter()-tic}
    (out / "execution_gap_v1.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Wrote {len(rows)} decision rows; elapsed {report['elapsed_seconds']:.1f}s",flush=True)


if __name__ == "__main__":
    main()
