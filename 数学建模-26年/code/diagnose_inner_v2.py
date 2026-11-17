"""Fixed-contract causal replay of planner, greedy and inner-LP controller.

Four prescribed Q3 dates x four outer decisions x all 28 historical scenarios.
This diagnostic never changes a policy, primary result, or workbook.
Future scenario realizations are withheld from the inner LP: only the current
sample is revealed, while every later forecast remains frozen at the issue.
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import numpy as np
import dispatch_v2 as model
import optimize_dispatch as legacy

DATES = ("2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21")
DATA = model.DATA


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(contract, realized_net, realized_price, frozen_net, frozen_price,
           initial_energy, terminal_scale, controller):
    """Causal replay: fixed contract; current realization; frozen future mean."""
    horizon = len(contract)
    energy = float(initial_energy)
    states = np.zeros(horizon+1)
    states[0] = energy
    emergency = np.zeros(horizon)
    calls = 0
    for k in range(horizon):
        gap = realized_net[k]-contract[k]
        if controller == "greedy":
            c, d, energy, h, spill = legacy.execute_step(
                contract[k], realized_net[k], energy)
        elif ((gap >= 0 and energy <= model.EMIN+1e-7)
              or (gap <= 0 and energy >= model.EMAX-1e-7)
              or abs(gap) < 1e-10):
            c = d = 0.
            h, spill = max(0., gap), max(0., -gap)
        else:
            remaining_net = frozen_net[k:].copy()
            remaining_price = frozen_price[k:].copy()
            # Do not copy realized_net[k+1:] or realized_price[k+1:].
            remaining_net[0] = realized_net[k]
            remaining_price[0] = realized_price[k]
            action = model.inner_lp(
                contract[k:], remaining_net, remaining_price, energy,
                terminal_scale=terminal_scale)
            c, d, energy, h, spill = (
                action[name] for name in
                ("charge", "discharge", "energy", "emergency", "spill"))
            calls += 1
        assert abs(contract[k]+h+d-c-realized_net[k]-spill) < 1e-5
        assert abs(energy-states[k]-model.ETA*c+d/model.ETA) < 1e-5
        assert model.EMIN-1e-5 <= energy <= model.EMAX+1e-5
        assert 0 <= c <= model.U+1e-5 and 0 <= d <= model.U+1e-5
        assert min(c, d) < 1e-5
        assert h >= -1e-5
        states[k+1] = energy
        emergency[k] = h
    return states, emergency, calls


def decision_diagnostic(data, primary, cfg, date, start):
    day = int(np.flatnonzero(data["dates"] == date)[0])
    output_day = int(np.flatnonzero(primary["dates"] == date)[0])
    initial = float(primary["energy"][output_day, start])
    ns, ps, weights, reserve, risk, base = model.outer_inputs(
        data, day, start, "q3", cfg)
    reachable = np.maximum(0., initial-model.EMIN
                           +model.ETA*model.U*np.arange(1,145-start))
    reserve = np.minimum(reserve, reachable)
    zero_plan = primary["plan"][output_day, start:]
    planned = legacy.solve_lp(
        ns, ps, weights, initial, base,
        reference=None if start == 0 else zero_plan,
        reserve=reserve, risk=risk)
    contract = planned["grid"]
    # The reconstructed contract is compared only over the part that actually
    # executed before the next update, not against future revised contracts.
    executed = primary["adjusted"][output_day, start:start+36]
    reconstruction_error = float(np.max(np.abs(contract[:len(executed)]-executed)))
    assert reconstruction_error < 1e-4, (date, start, reconstruction_error)
    fn, fp = model.point_forecast(data, day, start//36, "q3")
    frozen_net, frozen_price = fn[start:].copy(), fp[start:].copy()
    reference_energy = planned["energy"]
    if start == 0:
        original_plan = contract.copy()
    else:
        original_plan = zero_plan.copy()
    delta = contract-original_plan
    non_emergency = ps @ (original_plan+1.5*np.maximum(delta,0)
                          +.5*np.maximum(-delta,0))
    plan_emergency = 5*np.sum(ps*planned["scenario_emergency"], axis=1)
    plan_bills = non_emergency+plan_emergency
    rows = []
    traces = {"greedy":[], "lp":[]}
    total_calls = 0
    for s, weight in enumerate(weights):
        rec = {"date":date, "start_slot":start, "issue_hour":start//6,
               "scenario":s, "probability":float(weight),
               "initial_energy_kwh":initial,
               "non_emergency_bill_yuan":float(non_emergency[s]),
               "planned_emergency_bill_yuan":float(plan_emergency[s]),
               "planned_bill_yuan":float(plan_bills[s]),
               "planned_terminal_energy_kwh":float(reference_energy[-1])}
        for controller in ("greedy", "lp"):
            states, emergency, calls = replay(
                contract, ns[s], ps[s], frozen_net, frozen_price,
                initial, cfg.terminal_scale, controller)
            traces[controller].append(states)
            emergency_bill = float(5*ps[s]@emergency)
            bill = float(non_emergency[s]+emergency_bill)
            gap = bill-float(plan_bills[s])
            rec.update({
                f"{controller}_bill_yuan":bill,
                f"{controller}_emergency_bill_yuan":emergency_bill,
                f"{controller}_emergency_energy_kwh":float(emergency.sum()),
                f"{controller}_minus_plan_yuan":gap,
                f"{controller}_absolute_gap_yuan":abs(gap),
                f"{controller}_soc_rmse_kwh":float(
                    np.sqrt(np.mean((states-reference_energy)**2))),
                f"{controller}_terminal_energy_kwh":float(states[-1])})
            total_calls += calls
        rec["lp_minus_greedy_bill_yuan"] = rec["lp_bill_yuan"]-rec["greedy_bill_yuan"]
        rows.append(rec)
    def expectation(key):
        return float(weights @ np.array([r[key] for r in rows]))
    summary = {"date":date,"start_slot":start,"issue_hour":start//6,
               "scenario_count":len(weights),"remaining_slots":len(contract),
               "initial_energy_kwh":initial,
               "planned_bill_yuan":float(weights@plan_bills),
               "reconstructed_executed_segment_max_error":reconstruction_error,
               "inner_lp_calls":total_calls}
    for controller in ("greedy","lp"):
        states = np.asarray(traces[controller])
        summary.update({
            f"{controller}_bill_yuan":expectation(f"{controller}_bill_yuan"),
            f"{controller}_signed_mean_gap_yuan":expectation(f"{controller}_minus_plan_yuan"),
            f"{controller}_mean_absolute_scenario_gap_yuan":expectation(f"{controller}_absolute_gap_yuan"),
            f"{controller}_absolute_expected_gap_yuan":abs(expectation(f"{controller}_minus_plan_yuan")),
            f"{controller}_soc_rmse_kwh":float(np.sqrt(
                weights@np.mean((states-reference_energy[None,:])**2,axis=1))),
            f"{controller}_expected_terminal_energy_kwh":expectation(f"{controller}_terminal_energy_kwh"),
            f"{controller}_expected_soc_kwh":(weights@states).tolist()})
    summary["lp_minus_greedy_bill_yuan"] = expectation("lp_minus_greedy_bill_yuan")
    summary["reference_soc_kwh"] = reference_energy.tolist()
    summary["contract_kwh"] = contract.tolist()
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dates",nargs="+",default=list(DATES),choices=DATES)
    args = parser.parse_args()
    selection_path = DATA/"selection_v2.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    assert selection.get("frozen"), "Run after January selection is frozen."
    cfg = model.Config(**selection["cases"]["q3"]["config"])
    assert cfg.scenarios == 28 and not cfg.cancellation_refund
    data = dict(np.load(DATA/"prepared_v2_q3.npz",allow_pickle=False))
    primary = dict(np.load(DATA/"q3_v2.npz",allow_pickle=False))
    rows, decisions = [], []
    sources = {path.name:sha256(path) for path in
               (selection_path, DATA/"prepared_v2_q3.npz", DATA/"q3_v2.npz",
                model.ROOT/"code/dispatch_v2.py", Path(__file__).resolve())}
    report = {
        "design":"four prescribed dates by four Q3 decision times; same contracts, initial SOC and historical scenarios",
        "case":"q3","frozen_config":asdict(cfg),"sources_sha256":sources,
        "information_rule":"current scenario net demand and price are revealed one slot at a time; future inputs stay at the decision's lawful point forecasts; no subsequent forecasts or outer contract revisions",
        "bill_rule":"original plan plus adjustment relative to the original plan plus fivefold emergency purchases over the remaining window; excludes terminal penalties, CVaR, numerical tie breaks and advisory future purchases",
        "interpretation":"windows overlap and decision points use primary observed states; aggregate means are diagnostic means, not annual cost changes or global optimality gaps",
        "expected_soc_note":"SOC RMSE averages scenario squared errors against the common planning reference; terminal inventories are not forced to match",
        "decisions":decisions,"completed":False}
    began = time.perf_counter()
    json_path = DATA/"inner_gap_v2.json"
    csv_path = DATA/"inner_gap_v2.csv"
    for date in args.dates:
        for start in (0,36,72,108):
            tick = time.perf_counter()
            summary, scenario_rows = decision_diagnostic(data,primary,cfg,date,start)
            summary["elapsed_seconds"] = time.perf_counter()-tick
            decisions.append(summary)
            rows.extend(scenario_rows)
            json_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
            print(f"DONE {date} {start//6:02d}:00, "
                  f"greedy gap={summary['greedy_signed_mean_gap_yuan']:.2f}, "
                  f"LP gap={summary['lp_signed_mean_gap_yuan']:.2f}, "
                  f"seconds={summary['elapsed_seconds']:.1f}",flush=True)
    with csv_path.open("w",encoding="utf-8-sig",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregate = {"decision_count":len(decisions),"scenario_replays_per_controller":len(rows)}
    for controller in ("greedy","lp"):
        for metric in ("signed_mean_gap_yuan","mean_absolute_scenario_gap_yuan",
                       "absolute_expected_gap_yuan","soc_rmse_kwh"):
            key = f"{controller}_{metric}"
            aggregate[f"decision_mean_{key}"] = float(np.mean([d[key] for d in decisions]))
    aggregate["decision_mean_lp_minus_greedy_bill_yuan"] = float(
        np.mean([d["lp_minus_greedy_bill_yuan"] for d in decisions]))
    aggregate["decisions_lp_has_smaller_absolute_expected_gap"] = sum(
        d["lp_absolute_expected_gap_yuan"] < d["greedy_absolute_expected_gap_yuan"]
        for d in decisions)
    aggregate["decisions_lp_has_lower_replay_bill"] = sum(
        d["lp_minus_greedy_bill_yuan"] < -1e-7 for d in decisions)
    report.update(aggregate=aggregate,completed=True,
                  elapsed_seconds=time.perf_counter()-began)
    json_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(aggregate,ensure_ascii=False,indent=2),flush=True)


if __name__ == "__main__":
    main()
