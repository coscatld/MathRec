"""Reproduce compatibility, legal-update gating and future-isolation checks.

Run from the project root:
    python code/check_outer_timing_policy.py

Writes only outputs/outer_timing_ablation/policy_checks.json. Prior source,
prepared data, frozen decisions and earlier experiment results are read-only.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

for thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                   "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[thread_var] = "1"

import numpy as np
import scipy

import dispatch_v2 as model
import optimize_dispatch as legacy
from outer_timing_policy import simulate_timing, screening_active
from run_outer_candidate_experiment import simulate_selected


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/outer_timing_ablation/policy_checks.json"
CASES = ("q3", "q4_3")
TIMING_MODES = ("midnight_only", "intraday_only")
MODES = ("none", "all", *TIMING_MODES)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact_array(left, right, label):
    np.testing.assert_array_equal(left, right, err_msg=label)


def check_compatibility_and_gating(case, selection):
    cfg = model.Config(**selection["cases"][case]["config"])
    data = dict(np.load(ROOT / f"data/prepared_v2_{case}.npz", allow_pickle=False))
    primary = dict(np.load(ROOT / f"data/{case}_v2.npz", allow_pickle=False))
    start = int(np.flatnonzero(data["dates"] == "2025-02-01")[0])
    stored = int(np.flatnonzero(primary["dates"] == "2025-02-01")[0])
    initial = float(primary["energy"][stored, 0])
    baseline = model.simulate(data, case, cfg, start, start + 2, initial, progress=False)
    with contextlib.redirect_stdout(io.StringIO()):
        retained, retained_decisions = simulate_selected(
            data, case, cfg, start, start + 2, initial)
    checked = {}
    for mode in MODES:
        result, decisions = simulate_timing(
            data, case, cfg, start, start + 2, initial, mode, progress=False)
        if mode in ("none", "all"):
            target = baseline if mode == "none" else retained
            for key, value in result.items():
                exact_array(value, target[key], f"{case}/{mode}/{key}")
        assert len(decisions) == 8
        screened_count = sum(row["screening_active"] for row in decisions)
        assert screened_count == {
            "none": 0, "all": 8, "midnight_only": 2, "intraday_only": 6}[mode]
        unscreened_count = 0
        for row in decisions:
            slot = row["slot"]
            assert row["screening_active"] == screening_active(mode, slot)
            if row["screening_active"]:
                assert row["baseline_score"] is not None
                assert row["estimated_score_reduction_yuan"] >= -1e-5
                assert len(row["candidates"]) == (5 if slot == 0 else 6)
            else:
                assert row["selected_name"] == "scenario_lp"
                assert row["candidates"] == []
                for key in ("baseline_score", "estimated_score_reduction_yuan", "terminal_beta"):
                    assert row[key] is None
                day = int(np.flatnonzero(data["dates"] == row["date"])[0])
                ix = day - start
                ns, ps, weights, reserve, risk, base = model.outer_inputs(
                    data, day, slot, case, cfg)
                reach = np.maximum(
                    0., row["initial_energy_kwh"] - model.EMIN
                    + model.ETA * model.U * np.arange(1, 145 - slot))
                solution = legacy.solve_lp(
                    ns, ps, weights, row["initial_energy_kwh"], base,
                    reference=None if slot == 0 else result["plan"][ix, slot:],
                    reserve=np.minimum(reserve, reach), risk=risk)
                exact_array(row["selected_grid_kwh"], solution["grid"],
                            f"{case}/{mode}/slot{slot}/ordinary LP installed suffix")
                exact_array(result["adjusted"][ix, slot:slot + 36], solution["grid"][:36],
                            f"{case}/{mode}/slot{slot}/ordinary LP executed segment")
                unscreened_count += 1
        if mode == "all":
            for row, previous in zip(decisions, retained_decisions):
                for key in ("date", "slot", "hour", "initial_energy_kwh", "selected_name",
                            "baseline_score", "estimated_score_reduction_yuan",
                            "terminal_beta", "candidates"):
                    assert row[key] == previous[key], (case, key)
        checked[mode] = {
            "passed": True,
            "screened_count": screened_count,
            "unscreened_lp_independently_resolved_count": unscreened_count,
            "bill_yuan": float(result["total_cost"].sum()),
            "retained_or_main_exact_array_equality": True if mode in ("none", "all") else None,
            "physical_and_bill_validation": True,
        }
    return {"case": case, "period": ["2025-02-01", "2025-02-02"],
            "initial_energy_kwh": initial, "modes": checked}


def strip_timings(decision):
    return {key: value for key, value in decision.items()
            if key not in ("generation_seconds", "replay_seconds")}


def check_prefix_information_isolation(case, selection):
    frozen = selection["cases"][case]
    cfg = model.Config(**frozen["config"])
    raw = dict(np.load(ROOT / "data/prepared.npz", allow_pickle=False))
    altered = {key: value.copy() for key, value in raw.items()}
    day, cutoff = 200, 72
    issue = cutoff // 36
    # Procurement at noon precedes revealing the noon realization. Current
    # and earlier published PV forecast vintages remain exactly unchanged.
    altered["load"][day, cutoff:] *= 1.7
    altered["pv"][day, cutoff:] *= .1
    altered["price"][day, cutoff:] *= 2.3
    altered["pv_issue_forecast"][day, issue + 1:] *= .2
    altered["pv_hourly_raw"][day, issue + 1:] *= .2
    prepared = [model.make_forecast_data(source, **frozen["forecasts"])
                for source in (raw, altered)]
    saved = dict(np.load(ROOT / f"data/{case}_v2.npz", allow_pickle=False))
    stored = int(np.flatnonzero(saved["dates"] == raw["dates"][day])[0])
    initial = float(saved["energy"][stored, 0])

    future_forecast_difference = float(np.nanmax(np.abs(
        prepared[0]["forecast_price_issue"][day, 3, 108:]
        - prepared[1]["forecast_price_issue"][day, 3, 108:])))
    future_pv_difference = float(np.nanmax(np.abs(
        prepared[0]["pv_issue_forecast"][day, 3]
        - prepared[1]["pv_issue_forecast"][day, 3])))
    assert future_forecast_difference > 0.
    assert future_pv_difference > 0.
    input_difference = 0.
    for slot in (0, 36, cutoff):
        bundles = []
        for data in prepared:
            ns, ps, weights, reserve, risk, _ = model.outer_inputs(data, day, slot, case, cfg)
            net, price = model.point_forecast(data, day, slot // 36, case)
            bundles.append((ns, ps, weights, reserve, risk, net[slot:], price[slot:]))
        for left, right in zip(*bundles):
            exact_array(left, right, f"{case}/slot{slot}/lawful outer inputs")
            input_difference = max(input_difference,
                                   float(np.max(np.abs(np.asarray(left)-np.asarray(right)))))

    checked = {}
    for mode in TIMING_MODES:
        runs = [simulate_timing(data, case, cfg, day, day + 1, initial,
                                mode, progress=False) for data in prepared]
        before, after = (run[0] for run in runs)
        prefix_fields = ("adjusted", "charge", "discharge", "emergency", "spill")
        for key in prefix_fields:
            exact_array(before[key][:, :cutoff], after[key][:, :cutoff],
                        f"{case}/{mode}/{key}/execution prefix")
        exact_array(before["energy"][:, :cutoff + 1], after["energy"][:, :cutoff + 1],
                    f"{case}/{mode}/energy/prefix")
        exact_array(before["plan"], after["plan"], f"{case}/{mode}/midnight plan")
        prefix_decisions = [
            [strip_timings(row) for row in run[1] if row["slot"] <= cutoff]
            for run in runs]
        assert len(prefix_decisions[0]) == len(prefix_decisions[1]) == 3
        assert prefix_decisions[0] == prefix_decisions[1], (case, mode, "decision prefix")
        changed_future = max(float(np.max(np.abs(
            before[key][:, cutoff:] - after[key][:, cutoff:])))
            for key in prefix_fields)
        assert changed_future > 0.
        checked[mode] = {
            "passed": True, "execution_prefix_slots": cutoff,
            "decision_prefix_slots": [0, 36, cutoff],
            "screened_decisions_in_prefix": sum(row["screening_active"]
                                                for row in prefix_decisions[0]),
            "execution_prefix_exact_equality": True,
            "full_midnight_plan_exact_equality": True,
            "full_decision_and_candidate_metrics_exact_equality_ignoring_timings": True,
            "positive_control_changed_execution_suffix_max_kwh": changed_future,
        }
    return {
        "case": case, "date": str(raw["dates"][day]), "cutoff_slot": cutoff,
        "initial_energy_kwh": initial, "causal_forecasts_rebuilt": True,
        "lawful_outer_input_max_difference": input_difference,
        "positive_control_future_price_issue_max_difference": future_forecast_difference,
        "positive_control_unpublished_pv_issue_max_difference": future_pv_difference,
        "scope": "Closed-loop current-day execution prefix plus decisions through noon; perturbations start before revealing noon actual values.",
        "modes": checked,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    began = time.perf_counter()
    selection = json.loads((ROOT / "data/selection_v2.json").read_text(encoding="utf-8"))
    assert selection.get("frozen")
    paths = [Path(__file__).resolve(), ROOT / "code/outer_timing_policy.py",
             ROOT / "code/run_outer_candidate_experiment.py", ROOT / "code/dispatch_v2.py",
             ROOT / "code/optimize_dispatch.py", ROOT / "code/outer_candidate_selector.py",
             ROOT / "data/selection_v2.json", ROOT / "data/prepared.npz"]
    for case in CASES:
        paths.extend((ROOT / f"data/prepared_v2_{case}.npz", ROOT / f"data/{case}_v2.npz"))
    source_hashes = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    report = {
        "schema_version": "1.0", "generated_by": "check_outer_timing_policy.py",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Preservation and information-isolation regression checks for the timing ablation only.",
        "passed": True, "checks": {},
        "execution_provenance": {
            "run_command": [sys.executable, *sys.argv], "working_directory": str(ROOT),
            "python_version": platform.python_version(), "numpy_version": np.__version__,
            "scipy_version": scipy.__version__, "source_sha256": source_hashes,
        },
    }
    for case in CASES:
        for name, check in (("compatibility_and_gating", check_compatibility_and_gating),
                            ("prefix_information_isolation", check_prefix_information_isolation)):
            key = f"{case}/{name}"
            try:
                report["checks"][key] = {"passed": True, **check(case, selection)}
            except Exception as exc:
                report["passed"] = False
                report["checks"][key] = {"passed": False, "error": repr(exc),
                                           "traceback": traceback.format_exc()}
            print(json.dumps({"check": key, "passed": report["checks"][key]["passed"]}),
                  flush=True)
    unchanged = source_hashes == {str(path.relative_to(ROOT)): digest(path) for path in paths}
    report["source_inputs_unchanged"] = unchanged
    report["passed"] = bool(report["passed"] and unchanged)
    report["execution_provenance"]["run_exit_code"] = 0 if report["passed"] else 1
    report["elapsed_seconds"] = time.perf_counter() - began
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "report": str(output)}), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
