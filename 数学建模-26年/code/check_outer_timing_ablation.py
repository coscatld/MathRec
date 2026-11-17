"""Independently audit saved timing ablations without altering model artifacts.

Recompute physical balances and actual settlement from saved arrays and prices;
reconstruct every installed contract; independently replay selected/no-adjust
plans; re-solve every unscreened LP; verify source/output hashes and final choice.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
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
import dispatch_v2 as model
import optimize_dispatch as legacy

ROOT = Path(__file__).resolve().parents[1]
SPECS = [(case, mode) for case in ("q3", "q4_3")
         for mode in ("midnight_only", "intraday_only")]
LEGAL = (0, 36, 72, 108)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(actual, expected, label, tolerance=1e-5):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape, (label, actual.shape, expected.shape)
    assert np.isfinite(actual).all() and np.isfinite(expected).all(), label
    error = float(np.max(np.abs(actual-expected), initial=0.))
    assert error < tolerance, (label, error)
    return error


def independent_replay(grid, net, prices, weight, energy, base, reference):
    """Use paired scenario prices and independently written storage equations."""
    weight = np.asarray(weight, dtype=float)
    weight = weight / weight.sum()
    state = np.full(len(net), float(energy))
    emergency_cost = np.zeros(len(net))
    emergency_kwh = np.zeros(len(net))
    cap = legacy.PMAX * legacy.DT
    for t, purchase in enumerate(grid):
        surplus = np.maximum(purchase-net[:, t], 0.)
        deficit = np.maximum(net[:, t]-purchase, 0.)
        charge = np.minimum(surplus, np.minimum(cap, np.maximum(legacy.EMAX-state, 0.) / legacy.ETA))
        discharge = np.minimum(deficit, np.minimum(cap, np.maximum(state-legacy.EMIN, 0.) * legacy.ETA))
        emergency = deficit-discharge
        state += charge*legacy.ETA-discharge/legacy.ETA
        emergency_kwh += emergency
        emergency_cost += 5. * prices[:, t] * emergency
    if reference is None:
        purchase_cost = np.sum(prices*grid[None, :], axis=1)
        adjustment_cost = np.zeros(len(net))
    else:
        purchase_cost = np.sum(prices*reference[None, :], axis=1)
        delta = grid-reference
        down = -.5 if base.cancellation_refund else .5
        quantity = 1.5*np.maximum(delta, 0.)+down*np.maximum(-delta, 0.)
        adjustment_cost = np.sum(prices*quantity[None, :], axis=1)
    beta = float(base.terminal_scale*np.median(weight @ prices))
    terminal = beta*np.abs(state-base.terminal_target)
    return {"score":float(weight @ (purchase_cost+adjustment_cost+emergency_cost+terminal)),
            "expected_purchase_cost":float(weight @ purchase_cost),
            "expected_adjustment_cost":float(weight @ adjustment_cost),
            "expected_emergency_cost":float(weight @ emergency_cost),
            "expected_bill":float(weight @ (purchase_cost+adjustment_cost+emergency_cost)),
            "expected_terminal_penalty":float(weight @ terminal),
            "expected_emergency_kwh":float(weight @ emergency_kwh),
            "expected_final_energy":float(weight @ state),
            "terminal_beta":beta}


def audit_variant(case, mode, folder):
    began = time.perf_counter()
    folder = Path(folder)
    prefix = f"{case}_{mode}"
    arrays = dict(np.load(folder/f"{prefix}.npz", allow_pickle=False))
    rows = read(folder/f"{prefix}_decisions.json")
    summary = read(folder/f"{prefix}_summary.json")
    raw = dict(np.load(ROOT/f"data/prepared_v2_{case}.npz", allow_pickle=False))
    main = dict(np.load(ROOT/f"data/{case}_v2.npz", allow_pickle=False))
    incumbent = dict(np.load(ROOT/f"outputs/outer_candidate_test/{case}_selected.npz", allow_pickle=False))
    selection = read(ROOT/"data/selection_v2.json")
    cfg = model.Config(**selection["cases"][case]["config"])
    assert selection["frozen"] and cfg.risk_mode == cfg.reserve_mode == "none"
    assert cfg.controller == "greedy" and not cfg.cross_day and not cfg.no_storage
    assert tuple(cfg.updates) == LEGAL
    data = model.feedback_data(raw, cfg.feedback)
    dates = arrays["dates"]
    nd = len(dates)
    assert nd == 334 and str(dates[0]) == "2025-02-01" and str(dates[-1]) == "2025-12-31"
    assert np.array_equal(dates, main["dates"]) and np.array_equal(dates, incumbent["dates"])
    start = int(np.flatnonzero(data["dates"] == dates[0])[0])
    assert np.array_equal(dates, data["dates"][start:start+nd])
    close(arrays["energy"][0, 0], main["energy"][0, 0], "main opening SOC")
    close(arrays["energy"][0, 0], incumbent["energy"][0, 0], "incumbent opening SOC")
    for key, value in arrays.items():
        if key != "dates":
            assert np.isfinite(value).all(), key
            assert value.shape == ((nd, 145) if key == "energy" else
                                   (nd,) if key.endswith("cost") else (nd, 144)), (key, value.shape)
    actual_net = (data["load"][start:start+nd]-data["pv"][start:start+nd])*legacy.DT
    balance = close(arrays["adjusted"]+arrays["emergency"]+arrays["discharge"],
                    arrays["charge"]+actual_net+arrays["spill"], "energy balance")
    storage = close(np.diff(arrays["energy"], axis=1),
                    legacy.ETA*arrays["charge"]-arrays["discharge"]/legacy.ETA,
                    "storage dynamics")
    continuity = close(arrays["energy"][1:, 0], arrays["energy"][:-1, -1], "day continuity")
    assert arrays["energy"].min() >= legacy.EMIN-1e-6
    assert arrays["energy"].max() <= legacy.EMAX+1e-6
    for key in ("plan", "adjusted", "charge", "discharge", "emergency", "spill"):
        assert arrays[key].min() >= -1e-6, key
    assert max(arrays["charge"].max(), arrays["discharge"].max()) <= legacy.PMAX*legacy.DT+1e-6
    assert np.minimum(arrays["charge"], arrays["discharge"]).max() <= 1e-6
    close(arrays["adjusted"][:, :36], arrays["plan"][:, :36], "midnight installed prefix")
    if not cfg.cancellation_refund:
        assert np.min(arrays["adjusted"]-arrays["plan"]) >= -1e-6
    # Directly verify all realized actions against the frozen scalar executor.
    executor_error = 0.
    for ix in range(nd):
        for slot in range(144):
            expected = legacy.execute_step(float(arrays["adjusted"][ix, slot]),
                                           float(actual_net[ix, slot]),
                                           float(arrays["energy"][ix, slot]))
            observed = [arrays["charge"][ix, slot], arrays["discharge"][ix, slot],
                        arrays["energy"][ix, slot+1], arrays["emergency"][ix, slot],
                        arrays["spill"][ix, slot]]
            executor_error = max(executor_error, close(observed, expected, "scalar execution"))
    prices = data["price"][start:start+nd] if case == "q4_3" else np.broadcast_to(data["fixed_price"], (nd, 144))
    delta = arrays["adjusted"]-arrays["plan"]
    down = -.5 if cfg.cancellation_refund else .5
    daily = {"plan_cost":np.sum(prices*arrays["plan"], axis=1),
             "adjustment_cost":np.sum(prices*(1.5*np.maximum(delta, 0.)+down*np.maximum(-delta, 0.)), axis=1),
             "emergency_cost":5.*np.sum(prices*arrays["emergency"], axis=1)}
    daily["total_cost"] = sum(daily.values())
    cost_errors = {key:close(arrays[key], value, key) for key, value in daily.items()}
    close(summary["actual_bill_yuan"], daily["total_cost"].sum(), "summary total")
    for key in ("plan_cost", "adjustment_cost", "emergency_cost"):
        close(summary[key+"_yuan"], daily[key].sum(), "summary "+key)
    close(summary["emergency_kwh"], arrays["emergency"].sum(), "summary emergency")
    close(summary["initial_energy_kwh"], arrays["energy"][0, 0], "summary opening")
    close(summary["final_energy_kwh"], arrays["energy"][-1, -1], "summary endpoint")
    close(summary["bill_minus_main_yuan"], daily["total_cost"].sum()-main["total_cost"].sum(), "main difference")
    close(summary["bill_minus_all_screening_yuan"], daily["total_cost"].sum()-incumbent["total_cost"].sum(), "incumbent difference")
    ns, ps, weights, _, _, base = model.outer_inputs(data, start+nd-1, 0, case, cfg)
    beta = float(base.terminal_scale*np.median(weights @ ps))
    endpoint_penalty = beta*abs(float(arrays["energy"][-1, -1])-base.terminal_target)
    close(summary["endpoint_beta"], beta, "endpoint beta")
    close(summary["endpoint_target_kwh"], base.terminal_target, "endpoint target")
    close(summary["endpoint_penalty_yuan"], endpoint_penalty, "endpoint penalty")
    close(summary["bill_plus_endpoint_penalty_yuan"], daily["total_cost"].sum()+endpoint_penalty, "endpoint corrected bill")
    for monthly in summary["monthly"]:
        mask = np.array([str(day).startswith(monthly["month"]) for day in dates])
        close(monthly["actual_bill_yuan"], daily["total_cost"][mask].sum(), "monthly bill")
        close(monthly["emergency_kwh"], arrays["emergency"][mask].sum(), "monthly emergency")
    assert len(summary["monthly"]) == 11
    assert summary["actual_improved_days_vs_all"] == int(np.sum(daily["total_cost"] < incumbent["total_cost"]-1e-6))
    assert summary["actual_worsened_days_vs_all"] == int(np.sum(daily["total_cost"] > incumbent["total_cost"]+1e-6))

    assert len(rows) == nd*4 == summary["outer_decision_count"]
    installed_error = lp_error = replay_error = no_adjust_error = components_error = 0.
    active_count = ordinary_count = 0
    chosen_names = Counter()
    for ix in range(nd):
        effective = np.zeros(144)
        for update, slot in enumerate(LEGAL):
            row = rows[4*ix+update]
            assert row["date"] == str(dates[ix]) and row["slot"] == slot and row["hour"] == slot//6
            assert row["screening_mode"] == mode
            active = slot == 0 if mode == "midnight_only" else slot > 0
            assert row["screening_active"] is active
            energy = float(arrays["energy"][ix, slot])
            close(row["initial_energy_kwh"], energy, "decision state")
            grid = np.asarray(row["selected_grid_kwh"], dtype=float)
            assert grid.shape == (144-slot,) and np.isfinite(grid).all() and grid.min() >= -1e-6
            reference = None if slot == 0 else arrays["plan"][ix, slot:].copy()
            current = None if slot == 0 else effective[slot:].copy()
            if slot == 0:
                installed_error = max(installed_error, close(grid, arrays["plan"][ix], "full midnight plan"))
            installed_error = max(installed_error, close(grid[:36], arrays["adjusted"][ix, slot:slot+36], "installed contract execution"))
            ns, ps, weights, reserve, risk, base = model.outer_inputs(data, start+ix, slot, case, cfg)
            reach = np.maximum(0., energy-model.EMIN+model.ETA*model.U*np.arange(1, 145-slot))
            reserve = np.minimum(reserve, reach)
            assert risk == 0. and np.all(reserve == 0.)
            if active:
                active_count += 1
                chosen_names[row["selected_name"]] += 1
                assert row["decision_source"] == "candidate_replay"
                candidates = row["candidates"]
                names = [entry["name"] for entry in candidates]
                expected_names = ["scenario_lp", "point_lp", "margin_0.25sigma", "margin_0.5sigma", "margin_1sigma"]
                if slot:
                    expected_names += ["no_adjust"]
                assert names == expected_names, names
                scores = np.asarray([entry["score"] for entry in candidates])
                selected = candidates[int(np.argmin(scores))]
                assert selected["name"] == row["selected_name"]
                close(row["baseline_score"], scores[0], "baseline score")
                close(row["estimated_score_reduction_yuan"], scores[0]-selected["score"], "estimated reduction")
                assert row["estimated_score_reduction_yuan"] >= -1e-6
                for entry in candidates:
                    for field, value in entry.items():
                        if field != "name":
                            assert np.isfinite(value), (field, value)
                    components_error = max(components_error, close(entry["expected_bill"],
                        entry["expected_purchase_cost"]+entry["expected_adjustment_cost"]+entry["expected_emergency_cost"], "candidate bill components"))
                    components_error = max(components_error, close(entry["score"],
                        entry["expected_bill"]+entry["expected_terminal_penalty"], "candidate score components"))
                independent = independent_replay(grid, ns, ps, weights, energy, base, reference)
                close(row["terminal_beta"], independent["terminal_beta"], "replay beta")
                for key, value in independent.items():
                    if key != "terminal_beta":
                        replay_error = max(replay_error, close(selected[key], value, "selected replay "+key))
                if slot:
                    no_adjust = candidates[-1]
                    independent_current = independent_replay(current, ns, ps, weights, energy, base, reference)
                    for key, value in independent_current.items():
                        if key != "terminal_beta":
                            no_adjust_error = max(no_adjust_error, close(no_adjust[key], value, "no-adjust replay "+key))
                    if selected["name"] == "no_adjust":
                        close(grid, current, "no-adjust retained contract")
            else:
                ordinary_count += 1
                assert row["decision_source"] == "ordinary_scenario_lp" and row["selected_name"] == "scenario_lp"
                assert not row["candidates"]
                for field in ("baseline_score", "estimated_score_reduction_yuan", "terminal_beta",
                              "generation_seconds", "replay_seconds"):
                    assert row[field] is None, field
                solution = legacy.solve_lp(ns, ps, weights, energy, base,
                                           reference=reference, reserve=reserve, risk=risk)
                lp_error = max(lp_error, close(grid, solution["grid"], "ordinary LP re-solve"))
                close(row["scenario_lp_objective"], solution["objective"], "ordinary LP objective")
            effective[slot:] = grid
    assert active_count == (334 if mode == "midnight_only" else 1002) == summary["screened_decision_count"]
    assert ordinary_count == summary["ordinary_lp_decision_count"]
    assert dict(chosen_names) == summary["screened_selection_counts"]
    print(f"AUDIT {prefix}: {nd} days, {active_count} replays, {ordinary_count} LP re-solves passed", flush=True)
    return prefix, {"passed":True,"days":nd,"actual_bill_yuan":float(daily["total_cost"].sum()),
        "opening_energy_kwh":float(arrays["energy"][0, 0]),"final_energy_kwh":float(arrays["energy"][-1, -1]),
        "emergency_kwh":float(arrays["emergency"].sum()),"max_daily_billing_error_yuan":cost_errors,
        "max_balance_error_kwh":balance,"max_storage_dynamics_error_kwh":storage,
        "max_day_continuity_error_kwh":continuity,"max_scalar_executor_error":executor_error,
        "legal_update_count":len(rows),"screened_decision_count":active_count,
        "ordinary_lp_resolved_count":ordinary_count,"max_installed_execution_error_kwh":installed_error,
        "max_ordinary_lp_grid_error_kwh":lp_error,"max_selected_replay_field_error":replay_error,
        "max_no_adjust_replay_field_error":no_adjust_error,"max_candidate_component_error_yuan":components_error,
        "candidate_argmin_and_tie_order":True,"paired_scenario_prices_in_independent_replay":True,
        "endpoint_adjustment_recomputed":True,"summary_and_monthly_metrics_recomputed":True,
        "elapsed_seconds":time.perf_counter()-began}


def verify_records(records):
    for item in records:
        path = ROOT/item["path"]
        assert path.is_file() and path.stat().st_size == item["bytes"], item["path"]
        assert digest(path) == item["sha256"], item["path"]
    return len(records)


def audit_selection(folder):
    chosen = read(folder/"final_strategy.json")
    assert not chosen["main_artifacts_replaced"]
    decisions = {}
    for case in ("q2", "q3", "q4_2", "q4_3"):
        entry = chosen["cases"][case]
        choices = entry["alternatives"]
        for name, choice in choices.items():
            arrays = dict(np.load(ROOT/choice["array_path"], allow_pickle=False))
            bill = float(arrays["total_cost"].sum())
            close(choice["actual_bill_yuan"], bill, "selection bill")
            penalty = choice["endpoint_beta"]*abs(float(arrays["energy"][-1, -1])-choice["endpoint_target_kwh"])
            close(choice["bill_plus_endpoint_penalty_yuan"], bill+penalty, "selection endpoint")
        if case in ("q2", "q4_2"):
            expected = "main"
            assert set(choices) == {"main"}
        else:
            assert set(choices) == {"main", "all", "midnight_only", "intraday_only"}
            expected = "all"
            raw_best = min(choices, key=lambda name:choices[name]["actual_bill_yuan"])
            if (choices[raw_best]["actual_bill_yuan"] < choices["all"]["actual_bill_yuan"]-.01
                    and choices[raw_best]["bill_plus_endpoint_penalty_yuan"] <=
                    min(option["bill_plus_endpoint_penalty_yuan"] for option in choices.values())+.01):
                expected = raw_best
            assert entry["raw_bill_best"] == raw_best
            close(entry["interaction_yuan"], choices["all"]["actual_bill_yuan"]-choices["midnight_only"]["actual_bill_yuan"]
                  -choices["intraday_only"]["actual_bill_yuan"]+choices["main"]["actual_bill_yuan"], "interaction")
        assert entry["selected_strategy"] == expected
        assert entry["selected"] == choices[expected]
        decisions[case] = expected
    return decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/outer_timing_ablation")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    folder = args.output.resolve()
    folder.relative_to(ROOT)
    audit = {"audit":"independent saved-output timing-ablation audit", "passed":False,
             "created_utc":datetime.now(timezone.utc).isoformat(),"python":platform.python_version(),
             "auditor_script_sha256":digest(__file__),"audit_command":[sys.executable, *sys.argv],
             "cases":{},"protected_sources_read_only":True}
    try:
        experiment = read(folder/"experiment.json")
        manifest = read(folder/"run_manifest.json")
        assert experiment["completed"] and experiment["protected_inputs_unchanged"]
        assert manifest["run_exit_code"] == 0 and manifest["status"] == "PASS"
        assert {tuple(spec) for spec in experiment["specs"]} == set(SPECS)
        audit["verified_protected_input_count"] = verify_records(manifest["inputs"])
        audit["verified_source_count"] = verify_records(manifest["source_code"])
        audit["verified_output_count"] = verify_records(manifest["output_artifacts"])
        assert manifest["source_code"] == experiment["source_code"]
        assert {item["path"]:item for item in manifest["inputs"]} == experiment["protected_inputs"]
        assert set(experiment["results"]) == {f"{case}_{mode}" for case, mode in SPECS}
        for case, mode in SPECS:
            prefix = f"{case}_{mode}"
            assert experiment["results"][prefix] == read(folder/f"{prefix}_summary.json")
        with ProcessPoolExecutor(max_workers=min(args.workers, 4)) as pool:
            jobs = [pool.submit(audit_variant, case, mode, str(folder)) for case, mode in SPECS]
            for job in as_completed(jobs):
                name, result = job.result()
                audit["cases"][name] = result
        audit["final_strategy_recomputed"] = audit_selection(folder)
        verify_records(manifest["inputs"]+manifest["source_code"]+manifest["output_artifacts"])
        audit["hashes_still_match_after_audit"] = True
        audit["passed"] = True
        audit["scope_note"] = ("All 192384 executed slots, 5344 legal updates, 2672 screened selected-plan replays, "
            "all applicable no-adjust replays and 2672 inactive LP re-solves checked. Scores of unselected LP candidates "
            "were checked for component arithmetic and argmin order but their unsaved grids were not regenerated. "
            "Information timing code was reviewed; earlier causal perturbation and two-day none/all equivalence checks "
            "remain separate supporting evidence. This verifies developmental backtest records, not future performance.")
    except Exception as exc:
        audit["failure"] = repr(exc)
        raise
    finally:
        (folder/"audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
