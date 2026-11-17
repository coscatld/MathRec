"""Ablate replay-selection timing while preserving ordinary outer LP updates.

The frozen main model remains the scenario-LP mother model. ``none`` uses
that LP at every authorized update, ``all`` reproduces the retained outer
selector, and the two timing ablations only gate candidate replay selection.
An unscreened update still solves and installs the ordinary scenario LP using
this strategy's own current storage state and original midnight plan.

No controller, risk, reserve, candidate-menu, scenario, or settlement change
is introduced here. This module deliberately leaves the prior experiment's
source files and results untouched.
"""
from __future__ import annotations

import time

import numpy as np

import dispatch_v2 as model
import optimize_dispatch as legacy
from outer_candidate_selector import select_candidate
from run_outer_candidate_experiment import MARGINS, native


SCREENING_MODES = ("none", "all", "midnight_only", "intraday_only")
ARRAY_KEYS = ("plan", "adjusted", "charge", "discharge", "emergency", "spill")
COST_KEYS = ("plan_cost", "adjustment_cost", "emergency_cost", "total_cost")


def screening_active(mode, slot):
    """Return whether a legal outer update uses the replay candidate selector."""
    if mode not in SCREENING_MODES:
        raise ValueError(f"Unknown screening_mode: {mode!r}")
    return (mode == "all" or (mode == "midnight_only" and slot == 0)
            or (mode == "intraday_only" and slot > 0))


def simulate_timing(data, case, cfg, start_day, end_day, initial_energy,
                    screening_mode="all", progress=True):
    """Return execution arrays and an audit row for every authorized update.

    The four updates remain 00:00, 06:00, 12:00 and 18:00 in every mode.
    Replay scores exist only on screened updates; unscreened rows explicitly
    carry null scores and an empty candidate list. ``selected_grid_kwh`` is
    the complete remaining-day contract installed at the update, enabling
    reconstruction of the outstanding contract before subsequent updates.

    Intraday settlement always references this strategy's own midnight plan.
    The no-adjust candidate preserves the outstanding effective suffix just
    before the current update, including any earlier authorized revisions.
    """
    if screening_mode not in SCREENING_MODES:
        raise ValueError(f"Unknown screening_mode: {screening_mode!r}")
    if case not in ("q3", "q4_3"):
        raise ValueError("Timing ablations apply only to q3 and q4_3.")
    if cfg.controller != "greedy" or cfg.cross_day or cfg.no_storage:
        raise ValueError("Timing ablations freeze the original greedy executor with storage.")
    if cfg.risk_mode != "none" or cfg.reserve_mode != "none":
        raise ValueError("Timing ablations require zero risk and reserve.")
    updates = tuple(cfg.updates)
    if updates != (0, 36, 72, 108):
        raise ValueError("Timing ablations preserve all four original legal updates.")
    if not 0 <= start_day < end_day <= len(data["dates"]):
        raise ValueError("Invalid backtest interval.")
    energy = float(initial_energy)
    if not np.isfinite(energy) or not model.EMIN <= energy <= model.EMAX:
        raise ValueError("Initial storage must lie within the physical bounds.")

    data = model.feedback_data(data, cfg.feedback)
    nd = end_day - start_day
    arrays = {key: np.zeros((nd, 144)) for key in ARRAY_KEYS}
    arrays["energy"] = np.zeros((nd, 145))
    for key in COST_KEYS:
        arrays[key] = np.zeros(nd)
    arrays["dates"] = data["dates"][start_day:end_day].copy()
    dynamic = case == "q4_3"
    decisions = []
    began = time.perf_counter()

    for ix, day in enumerate(range(start_day, end_day)):
        arrays["energy"][ix, 0] = energy
        effective = np.zeros(144)
        for slot in range(144):
            if slot in updates:
                ns, ps, weights, reserve, risk, base = model.outer_inputs(
                    data, day, slot, case, cfg)
                reachable = np.maximum(
                    0., energy - model.EMIN
                    + model.ETA * model.U * np.arange(1, 145 - slot))
                reserve = np.minimum(reserve, reachable)
                active = screening_active(screening_mode, slot)
                reference = (None if slot == 0
                             else arrays["plan"][ix, slot:].copy())
                # Capture the outstanding contract before installing this
                # update, never from later realized executions.
                current_contract = (None if slot == 0
                                    else effective[slot:].copy())
                record = {
                    "date": str(data["dates"][day]), "slot": slot,
                    "hour": slot // 6, "initial_energy_kwh": energy,
                    "screening_mode": screening_mode,
                    "screening_active": active,
                    "selected_name": "scenario_lp",
                    "decision_source": "candidate_replay" if active else "ordinary_scenario_lp",
                    "baseline_score": None,
                    "estimated_score_reduction_yuan": None,
                    "terminal_beta": None,
                    "generation_seconds": None, "replay_seconds": None,
                    "candidates": [],
                }
                if active:
                    point_net, point_price = model.point_forecast(
                        data, day, slot // 36, case)
                    choice = select_candidate(
                        ns, ps, weights, energy, base,
                        point_net[slot:], point_price[slot:],
                        reference=reference, current_contract=current_contract,
                        reserve=reserve, risk=risk, margin_scales=MARGINS)
                    grid = choice["grid"]
                    record.update(
                        selected_name=choice["selected_name"],
                        baseline_score=float(choice["baseline_score"]),
                        estimated_score_reduction_yuan=float(choice["score_improvement"]),
                        terminal_beta=float(choice["terminal_beta"]),
                        generation_seconds=float(choice["generation_seconds"]),
                        replay_seconds=float(choice["replay_seconds"]),
                        candidates=[{k: native(v) for k, v in candidate.items()
                                     if k != "grid"}
                                    for candidate in choice["candidates"]])
                    assert record["estimated_score_reduction_yuan"] >= -1e-5
                else:
                    solution = legacy.solve_lp(
                        ns, ps, weights, energy, base,
                        reference=reference, reserve=reserve, risk=risk)
                    grid = solution["grid"]
                    # This internal LP objective is not a replay score.
                    record["scenario_lp_objective"] = float(solution["objective"])
                effective[slot:] = grid
                if slot == 0:
                    arrays["plan"][ix] = effective.copy()
                record["selected_grid_kwh"] = native(effective[slot:].copy())
                decisions.append(record)

            # Procurement is selected first; the execution step then observes
            # the current realized net load, exactly as in the retained runner.
            actual_net = (data["load"][day, slot] - data["pv"][day, slot]) * model.DT
            grid = float(effective[slot])
            c, d, energy, h, spill = legacy.execute_step(grid, actual_net, energy)
            for key, value in (("adjusted", grid), ("charge", c), ("discharge", d),
                               ("emergency", h), ("spill", spill)):
                arrays[key][ix, slot] = value
            arrays["energy"][ix, slot + 1] = energy

        prices = data["price"][day] if dynamic else data["fixed_price"]
        original = arrays["plan"][ix]
        delta = arrays["adjusted"][ix] - original
        down_rate = -.5 if cfg.cancellation_refund else .5
        arrays["plan_cost"][ix] = prices @ original
        arrays["adjustment_cost"][ix] = prices @ (
            1.5 * np.maximum(delta, 0) + down_rate * np.maximum(-delta, 0))
        arrays["emergency_cost"][ix] = 5 * prices @ arrays["emergency"][ix]
        arrays["total_cost"][ix] = sum(arrays[key][ix] for key in COST_KEYS[:-1])
        if progress and ((ix + 1) % 30 == 0 or ix == nd - 1):
            print(f"{case}/{screening_mode}: {ix+1}/{nd} days, "
                  f"bill={arrays['total_cost'][:ix+1].sum():.2f}, "
                  f"elapsed={time.perf_counter()-began:.1f}s", flush=True)

    model.validate(data, case, arrays, cfg, start_day)
    assert abs(arrays["energy"][0, 0] - initial_energy) < 1e-7
    assert np.allclose(arrays["total_cost"], arrays["plan_cost"]
                       + arrays["adjustment_cost"] + arrays["emergency_cost"])
    assert len(decisions) == nd * len(updates)
    expected_active = nd * sum(screening_active(screening_mode, slot) for slot in updates)
    assert sum(row["screening_active"] for row in decisions) == expected_active
    return arrays, decisions
