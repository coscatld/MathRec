"""Independent accounting and information checks for the proposed selector.

Writes only outputs/outer_candidate_test/checks.json. Existing model results,
source datasets, tables, and the manuscript remain untouched.
"""
from __future__ import annotations

import json
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np

import dispatch_v2 as model
import optimize_dispatch as legacy
import outer_candidate_selector as selector


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "outer_candidate_test" / "checks.json"
TOL = 1e-7


def assert_close(actual, expected, *, atol=TOL):
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=atol)


def scalar_accounting(grids, nets, prices, weights, initial, cfg,
                      reference=None, terminal_beta=None):
    """Call the legacy scalar executor separately for every candidate/path.

    The external oracle is the already established physical execution routine;
    bills are accumulated here slot by slot, without selector helper functions.
    """
    beta = (cfg.terminal_scale * float(np.median(weights @ prices))
            if terminal_beta is None else terminal_beta)
    shape = (len(grids), len(nets))
    result = {key: np.zeros(shape) for key in (
        "bill", "terminal_penalty", "score", "emergency_kwh", "final_energy")}
    for k, grid in enumerate(grids):
        for s, (net, price) in enumerate(zip(nets, prices)):
            energy = initial
            bill = emergency_sum = 0.0
            for t, purchase in enumerate(grid):
                charge, discharge, energy, emergency, spill = legacy.execute_step(
                    float(purchase), float(net[t]), float(energy), cfg.no_storage)
                assert abs(purchase + discharge + emergency - charge - net[t] - spill) < TOL
                if reference is None:
                    paid = purchase
                else:
                    delta = purchase - reference[t]
                    paid = (reference[t] + 1.5 * max(delta, 0.)
                            + (-.5 if cfg.cancellation_refund else .5) * max(-delta, 0.))
                bill += float(price[t]) * (paid + 5.0 * emergency)
                emergency_sum += emergency
            terminal = beta * abs(energy - cfg.terminal_target)
            result["bill"][k, s] = bill
            result["terminal_penalty"][k, s] = terminal
            result["score"][k, s] = bill + terminal
            result["emergency_kwh"][k, s] = emergency_sum
            result["final_energy"][k, s] = energy
    return result


def check_scalar_equivalence():
    grids = np.array([
        [0., 4000., 0., 10., 500., 0., 1200., 200.],
        [150., 0., 900., 0., 2400., 100., 0., 0.],
        [500., 500., 500., 500., 500., 500., 500., 500.],
    ])
    nets = np.array([
        [-500., 0., 2800., 10., 10., 800., -100., 200.],
        [800., 2500., -500., 1000., 4500., -1000., 2000., 1200.],
        [0., 200., 0., 0., 50., 1200., 0., -3000.],
        [3500., -4000., 500., 500., 500., 500., 500., 500.],
    ])
    prices = np.array([
        [.01, .1, .2, .5, 1., 2., .8, .3],
        [1., .6, .2, .1, .05, .02, .8, 2.],
        [.05, .3, 1., .2, .5, .7, .8, .4],
        [2., 1., .5, .3, .2, .1, .01, .05],
    ])
    weights = np.array([.1, .2, .3, .4])
    ref = np.array([100., 1000., 500., 20., 600., 0., 1000., 100.])
    max_error = 0.
    runs = 0
    mapping = {"bill": "expected_bill", "terminal_penalty": "expected_terminal_penalty",
               "score": "score", "emergency_kwh": "expected_emergency_kwh",
               "final_energy": "expected_final_energy"}
    for e0 in (legacy.EMIN, 6000., legacy.EMAX):
        for no_storage in (False, True):
            for refund in (False, True):
                cfg = legacy.Settings(no_storage=no_storage, cancellation_refund=refund,
                                      terminal_scale=.7)
                for reference in (None, ref):
                    actual = selector.replay_candidates(
                        grids, nets, prices, weights, e0, cfg, reference=reference)
                    expected = scalar_accounting(
                        grids, nets, prices, weights, e0, cfg, reference=reference)
                    for key, aggregate in mapping.items():
                        assert_close(actual["per_scenario"][key], expected[key])
                        assert_close(actual[aggregate], expected[key] @ weights)
                        max_error = max(max_error, float(np.max(np.abs(
                            actual["per_scenario"][key] - expected[key]))))
                    runs += 1
    return {"parameter_combinations": runs, "candidate_scenario_trajectories": runs * 12,
            "maximum_absolute_error": max_error,
            "coverage": "SOC bounds, charge/discharge power cap, surplus, deficit, spill, exact balance, no storage, both settlement conventions"}


def check_paired_price_shortage():
    cfg = legacy.Settings(no_storage=True, terminal_scale=0.)
    grid = np.zeros((1, 1))
    nets = np.array([[0.], [10.]])
    prices = np.array([[1.], [10.]])
    weights = np.array([.5, .5])
    actual = selector.replay_candidates(grid, nets, prices, weights, legacy.EMIN, cfg)
    assert_close(actual["expected_bill"], [250.])
    mean_price_score = 5. * float(weights @ prices[:, 0]) * float(weights @ nets[:, 0])
    assert abs(actual["expected_bill"][0] - mean_price_score) > 100.
    reversed_prices = selector.replay_candidates(
        grid, nets, prices[::-1].copy(), weights, legacy.EMIN, cfg)
    assert_close(reversed_prices["expected_bill"], [25.])
    return {"paired_expected_bill": 250., "incorrect_mean_price_bill": mean_price_score,
            "same_marginals_reversed_pairing_bill": float(reversed_prices["expected_bill"][0])}


def check_original_reference_and_no_adjust():
    cfg = legacy.Settings(terminal_scale=0., risk_scale=0., reserve_scale=0.)
    reference = np.array([10., 20.])
    current = np.array([14., 12.])
    nets = current[None, :]
    prices = np.array([[2., 3.]])
    weights = np.ones(1)
    replay = selector.replay_candidates(
        current[None, :], nets, prices, weights, 6000., cfg, reference=reference)
    # Original plan = 80, prior upward revision = 12, prior downward fee = 12.
    assert_close(replay["expected_bill"], [104.])
    selected = selector.select_candidate(
        nets, prices, weights, 6000., cfg, current.copy(), prices[0].copy(),
        reference=reference, current_contract=current, margin_scales=())
    matches = [candidate for candidate in selected["candidates"]
               if np.allclose(candidate["grid"], current, rtol=0, atol=TOL)
               and "adjust" in candidate["name"]]
    assert len(matches) == 1, [candidate["name"] for candidate in selected["candidates"]]
    assert_close(matches[0]["expected_bill"], 104.)
    return {"no_adjust_candidate": matches[0]["name"], "original_plan_bill": 80.,
            "existing_adjustment_bill": 24., "no_adjust_total_bill": 104.,
            "incorrect_rebased_bill": float(prices[0] @ current)}


def check_actual_terminal_and_common_beta():
    cfg = legacy.Settings(terminal_scale=.4)
    grids = np.array([[0., 0.], [100., 100.]])
    nets = np.array([[-100., -100.], [100., 100.]])
    prices = np.array([[1., 2.], [5., 6.]])
    weights = np.array([.5, .5])
    beta = 2.
    actual = selector.replay_candidates(
        grids, nets, prices, weights, 6000., cfg, terminal_beta=beta)
    final = np.array([[6180., 6000. - 200. / legacy.ETA], [6360., 6000.]])
    expected = beta * np.abs(final - cfg.terminal_target)
    assert_close(actual["per_scenario"]["final_energy"], final)
    assert_close(actual["per_scenario"]["terminal_penalty"], expected)
    assert_close(actual["expected_terminal_penalty"], expected @ weights)
    assert_close(actual["score"], actual["expected_bill"] + expected @ weights)
    penalty_on_mean_soc = beta * abs(float(final[0] @ weights) - cfg.terminal_target)
    assert actual["expected_terminal_penalty"][0] > penalty_on_mean_soc + 300.
    return {"common_beta_yuan_per_kwh": beta, "actual_final_energy_by_candidate_scenario": final.tolist(),
            "expected_terminal_penalty": actual["expected_terminal_penalty"].tolist(),
            "incorrect_penalty_at_mean_soc_first_candidate": penalty_on_mean_soc}


def check_nested_candidate_sets():
    cfg = legacy.Settings(no_storage=True, terminal_scale=0., risk_scale=0., reserve_scale=0.)
    nets = np.array([[0.], [10.]])
    prices = np.ones_like(nets)
    weights = np.array([.5, .5])
    small = selector.replay_candidates(np.array([[0.], [2.]]), nets, prices, weights, legacy.EMIN, cfg)
    large = selector.replay_candidates(np.array([[0.], [2.], [10.]]), nets, prices, weights, legacy.EMIN, cfg)
    assert_close(large["score"][:2], small["score"])
    assert float(np.min(large["score"])) <= float(np.min(small["score"])) + TOL
    assert_close(np.min(small["score"]), 22.)
    assert_close(np.min(large["score"]), 10.)
    n = np.array([[300., -100., 1400., 0.], [0., 150., 200., 800.],
                  [700., 500., -100., 2200.]])
    p = np.array([[1., .5, 1.5, .2], [1., .5, 1.5, .2], [1., .5, 1.5, .2]])
    w = np.array([.2, .3, .5])
    operational = replace(cfg, no_storage=False, terminal_scale=1.)
    base = selector.select_candidate(n, p, w, 6000., operational, w @ n, w @ p,
                                     margin_scales=())
    expanded = selector.select_candidate(n, p, w, 6000., operational, w @ n, w @ p,
                                         margin_scales=(.25, .5, 1.))
    by_name = {entry["name"]: entry for entry in expanded["candidates"]}
    for entry in base["candidates"]:
        assert entry["name"] in by_name
        assert_close(entry["grid"], by_name[entry["name"]]["grid"])
        assert_close(entry["score"], by_name[entry["name"]]["score"])
    base_min = min(entry["score"] for entry in base["candidates"])
    expanded_min = min(entry["score"] for entry in expanded["candidates"])
    assert expanded_min <= base_min + TOL
    chosen = by_name[expanded["selected_name"]]
    assert_close(chosen["score"], expanded_min)
    assert_close(expanded["grid"], chosen["grid"])
    return {"golden_small_minimum": 22., "golden_expanded_minimum": 10.,
            "solver_base_candidate_count": len(base["candidates"]),
            "solver_expanded_candidate_count": len(expanded["candidates"]),
            "solver_base_minimum": float(base_min), "solver_expanded_minimum": float(expanded_min),
            "guarantee_scope": "same fixed empirical score and retained old candidates only"}


def check_future_information_leakage():
    case, day, start = "q4_3", 200, 72
    issue = start // 36
    selection = json.loads((model.DATA / "selection_v2.json").read_text(encoding="utf-8"))
    frozen = selection["cases"][case]
    cfg = model.Config(**frozen["config"])
    raw = dict(np.load(model.DATA / "prepared.npz", allow_pickle=False))
    altered = {key: value.copy() for key, value in raw.items()}
    # τ's outer decision precedes revealing τ's realization. All changes here
    # are at τ or later; already published noon PV forecasts are held fixed.
    altered["load"][day, start:] *= 1.7
    altered["pv"][day, start:] *= .1
    altered["price"][day, start:] *= 2.3
    altered["pv_issue_forecast"][day, issue + 1:] *= .2
    altered["pv_hourly_raw"][day, issue + 1:] *= .2
    prepared = [model.make_forecast_data(source, **frozen["forecasts"])
                for source in (raw, altered)]
    saved = dict(np.load(model.DATA / f"{case}_v2.npz", allow_pickle=False))
    row = int(np.flatnonzero(saved["dates"] == raw["dates"][day])[0])
    initial = float(saved["energy"][row, start])
    reference = saved["plan"][row, start:].copy()
    # Reconstruct the still-outstanding 06:00 contract. Saved adjusted values
    # after noon already include the noon decision and must not be reused here.
    ns, ps, w, reserve, risk, base = model.outer_inputs(prepared[0], day, 36, case, cfg)
    previous_initial = float(saved["energy"][row, 36])
    reserve = np.minimum(reserve, np.maximum(
        0., previous_initial - model.EMIN + model.ETA * model.U * np.arange(1, 109)))
    previous = legacy.solve_lp(ns, ps, w, previous_initial, base,
                              reference=saved["plan"][row, 36:], reserve=reserve, risk=risk)
    current = previous["grid"][36:].copy()
    bundles, results = [], []
    for data in prepared:
        ns, ps, w, reserve, risk, base = model.outer_inputs(data, day, start, case, cfg)
        reserve = np.minimum(reserve, np.maximum(
            0., initial - model.EMIN + model.ETA * model.U * np.arange(1, 145 - start)))
        point_net, point_price = model.point_forecast(data, day, issue, case)
        bundles.append((ns, ps, w, reserve, risk, point_net[start:], point_price[start:]))
        results.append(selector.select_candidate(
            ns, ps, w, initial, base, point_net[start:], point_price[start:],
            reference=reference, current_contract=current, reserve=reserve, risk=risk))
    input_differences = [float(np.max(np.abs(np.asarray(left) - np.asarray(right))))
                         for left, right in zip(*bundles)]
    assert max(input_differences) == 0., input_differences
    before, after = results
    assert before["selected_name"] == after["selected_name"]
    assert_close(before["grid"], after["grid"])
    assert len(before["candidates"]) == len(after["candidates"])
    max_score_difference = 0.
    for left, right in zip(before["candidates"], after["candidates"]):
        assert left["name"] == right["name"]
        assert_close(left["grid"], right["grid"])
        for key in ("score", "expected_bill", "expected_terminal_penalty",
                    "expected_emergency_kwh", "expected_final_energy"):
            assert_close(left[key], right[key])
        max_score_difference = max(max_score_difference, abs(float(left["score"] - right["score"])))
    # Positive control: future issue forecasts did change after causal rebuild,
    # so the perturbation probe is not operating on identical prepared arrays.
    future_difference = float(np.nanmax(np.abs(
        prepared[0]["forecast_price_issue"][day, 3, 108:]
        - prepared[1]["forecast_price_issue"][day, 3, 108:])))
    assert future_difference > 0.
    return {"case": case, "date": str(raw["dates"][day]), "decision_slot": start,
            "causal_forecasts_rebuilt": True, "candidate_count": len(before["candidates"]),
            "selected_name": before["selected_name"],
            "outer_input_max_difference": max(input_differences),
            "candidate_score_max_difference": max_score_difference,
            "selected_grid_max_difference": float(np.max(np.abs(before["grid"] - after["grid"]))),
            "positive_control_future_issue_price_difference": future_difference,
            "scope": "fixed noon decision; root runner separately checks closed-loop execution"}


def main():
    checks = (
        ("scalar_executor_equivalence", check_scalar_equivalence),
        ("paired_price_shortage", check_paired_price_shortage),
        ("original_reference_and_no_adjust", check_original_reference_and_no_adjust),
        ("actual_terminal_and_common_beta", check_actual_terminal_and_common_beta),
        ("nested_candidate_sets", check_nested_candidate_sets),
        ("future_information_leakage", check_future_information_leakage),
    )
    report = {"scope": "isolated regression checks for an unvalidated outer candidate selector",
              "checks": {}, "passed": True}
    for name, check in checks:
        try:
            result = check()
            report["checks"][name] = {"passed": True, **result}
        except Exception as exc:
            report["passed"] = False
            report["checks"][name] = {"passed": False, "error": repr(exc),
                                      "traceback": traceback.format_exc()}
        print(json.dumps({"check": name, **report["checks"][name]}, ensure_ascii=False), flush=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "report": str(OUTPUT)}, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
