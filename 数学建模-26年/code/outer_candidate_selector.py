"""Isolated outer-plan selection with a frozen causal greedy executor.

This experiment does not change the legacy optimizer or its controller. It
compares a fixed set of purchase plans on the same supplied historical paths.
Each replay holds its plan fixed through the remaining day; it does not assume
another update or access any future realization from the evaluation day.

Inputs use kWh and currency/kWh, as in ``optimize_dispatch.solve_lp``. Prices
and net loads retain their scenario-row pairing. The score is the expected
settled bill plus one common absolute terminal-energy penalty. It is an
estimate, not a guarantee about a future realized bill. This first experiment
requires zero LP risk and reserve terms, so those terms cannot silently change
the candidate-generation comparison.
"""
from __future__ import annotations

import time

import optimize_dispatch as old
import numpy as np


def _vector(value, name, n, *, nonnegative=False):
    out = np.asarray(value, dtype=float)
    if out.shape != (n,) or not np.isfinite(out).all():
        raise ValueError(f"{name} must be a finite vector of shape ({n},)")
    if nonnegative and np.any(out < 0):
        raise ValueError(f"{name} must be nonnegative")
    return out


def _validated_inputs(net_scenarios, price_scenarios, weights,
                      initial_energy, base_settings, reference):
    net = np.asarray(net_scenarios, dtype=float)
    prices = np.asarray(price_scenarios, dtype=float)
    if net.ndim != 2 or min(net.shape) < 1 or not np.isfinite(net).all():
        raise ValueError("net_scenarios must be a nonempty finite (S, n) array")
    if prices.shape != net.shape or not np.isfinite(prices).all():
        raise ValueError("price_scenarios must have the same finite (S, n) shape")
    if np.any(prices <= 0):
        raise ValueError("scenario prices must be strictly positive")
    weight = _vector(weights, "weights", len(net), nonnegative=True)
    weight_sum = float(weight.sum())
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("weights must have a finite positive sum")
    weight = weight / weight_sum
    energy = float(initial_energy)
    if (not np.isfinite(energy)
            or not old.EMIN - 1e-7 <= energy <= old.EMAX + 1e-7):
        raise ValueError("initial_energy must be within the storage bounds")
    scale = float(base_settings.terminal_scale)
    target = float(base_settings.terminal_target)
    if not np.isfinite(scale) or scale < 0:
        raise ValueError("terminal_scale must be finite and nonnegative")
    if not np.isfinite(target) or not old.EMIN <= target <= old.EMAX:
        raise ValueError("terminal_target must be within the storage bounds")
    ref = (None if reference is None else
           _vector(reference, "reference", net.shape[1], nonnegative=True))
    beta = scale * float(np.median(weight @ prices))
    if not np.isfinite(beta):
        raise ValueError("terminal penalty coefficient must be finite")
    return net, prices, weight, energy, ref, beta


def _replay_validated(grids, net, prices, weights, initial_energy,
                      base_settings, reference, terminal_beta):
    """Vectorize legacy execute_step across K candidates and S scenarios."""
    count, horizon = grids.shape
    energy = np.full((count, len(net)), initial_energy, dtype=float)
    emergency_kwh = np.zeros_like(energy)
    emergency_cost = np.zeros_like(energy)
    cap = 0.0 if base_settings.no_storage else old.PMAX * old.DT
    for t in range(horizon):
        gap = grids[:, t, None] - net[None, :, t]
        charge = np.minimum(np.minimum(np.maximum(gap, 0.0), cap),
                            np.maximum(0.0, (old.EMAX - energy) / old.ETA))
        discharge = np.minimum(np.minimum(np.maximum(-gap, 0.0), cap),
                               np.maximum(0.0, (energy - old.EMIN) * old.ETA))
        emergency = np.maximum(0.0, -gap - discharge)
        energy = energy + old.ETA * charge - discharge / old.ETA
        emergency_kwh += emergency
        # Each net-load path is billed at its own paired price path.
        emergency_cost += 5.0 * prices[None, :, t] * emergency

    if reference is None:
        purchase_cost = grids @ prices.T
        adjustment_cost = np.zeros_like(purchase_cost)
    else:
        purchase_cost = np.broadcast_to(reference @ prices.T,
                                       (count, len(net))).copy()
        delta = grids - reference
        down_rate = -0.5 if base_settings.cancellation_refund else 0.5
        adjustment_quantity = (1.5 * np.maximum(delta, 0.0)
                               + down_rate * np.maximum(-delta, 0.0))
        adjustment_cost = adjustment_quantity @ prices.T
    bill = purchase_cost + adjustment_cost + emergency_cost
    terminal_penalty = terminal_beta * np.abs(
        energy - float(base_settings.terminal_target))
    score = bill + terminal_penalty
    if (not np.isfinite(score).all()
            or energy.min() < old.EMIN - 1e-6
            or energy.max() > old.EMAX + 1e-6):
        raise AssertionError("Replay produced nonfinite costs or invalid storage energy")
    paths = {
        "purchase_cost": purchase_cost,
        "adjustment_cost": adjustment_cost,
        "emergency_cost": emergency_cost,
        "bill": bill,
        "terminal_penalty": terminal_penalty,
        "score": score,
        "emergency_kwh": emergency_kwh,
        "final_energy": energy,
    }
    result = {"expected_" + key: value @ weights
              for key, value in paths.items() if key != "score"}
    result["score"] = score @ weights
    result["per_scenario"] = paths
    result["terminal_beta"] = terminal_beta
    return result


def replay_candidates(grids, net_scenarios, price_scenarios, weights,
                      initial_energy, base_settings, *, reference=None,
                      terminal_beta=None):
    """Score supplied plans using the unchanged greedy storage policy.

    ``grids`` has shape (K, n). Returned ``expected_*`` arrays and ``score``
    have shape (K,); ``per_scenario`` contains the corresponding (K, S)
    diagnostic arrays. ``reference`` is the original day-ahead plan for the
    remaining horizon. Its purchase cost is included in every intraday bill;
    adjustments are settled relative to that original plan, including for a
    no-adjust candidate that preserves a previously modified contract.

    By default every candidate uses terminal_scale times the median of the
    scenario-weighted slot prices as its common terminal penalty coefficient.
    """
    net, prices, weight, energy, ref, beta = _validated_inputs(
        net_scenarios, price_scenarios, weights, initial_energy,
        base_settings, reference)
    plans = np.asarray(grids, dtype=float)
    if (plans.ndim != 2 or plans.shape[0] < 1
            or plans.shape[1] != net.shape[1]
            or not np.isfinite(plans).all() or np.any(plans < 0)):
        raise ValueError("grids must be a nonnegative finite (K, n) array")
    if terminal_beta is not None:
        beta = float(terminal_beta)
        if not np.isfinite(beta) or beta < 0:
            raise ValueError("terminal_beta must be finite and nonnegative")
    return _replay_validated(plans, net, prices, weight, energy,
                             base_settings, ref, beta)


def select_candidate(net_scenarios, price_scenarios, weights, initial_energy,
                     base_settings, point_net, point_prices, *, reference=None,
                     current_contract=None, reserve=None, risk=0.0,
                     margin_scales=(0.25, 0.5, 1.0)):
    """Generate a fixed candidate set and minimize common expected replay cost.

    The baseline is the scenario LP on the supplied paths. Other LPs use the
    lawful point forecast, optionally increased slotwise by a fixed multiple
    of weighted historical-scenario net-load standard deviation. Intraday,
    ``current_contract`` is required and contributes ``no_adjust``; it must
    represent the currently outstanding contract, not merely ``reference``.

    All LPs use identical initial storage and Settings. Their internal LP
    objectives are diagnostics only: selection always uses original scenario
    paths and the common replay score. LP objectives are not directly
    comparable, since their input paths and price forecasts differ. Exact
    score ties retain the earlier candidate, hence prefer the baseline.
    """
    net, prices, weight, energy, ref, beta = _validated_inputs(
        net_scenarios, price_scenarios, weights, initial_energy,
        base_settings, reference)
    n = net.shape[1]
    forecast_net = _vector(point_net, "point_net", n)
    forecast_prices = _vector(point_prices, "point_prices", n)
    if np.any(forecast_prices <= 0):
        raise ValueError("point_prices must be strictly positive")
    if not np.isfinite(risk) or float(risk) != 0.0:
        raise ValueError("This isolated outer test requires risk=0")
    rho = np.zeros(n) if reserve is None else _vector(reserve, "reserve", n)
    if np.any(rho != 0.0):
        raise ValueError("This isolated outer test requires zero reserve")
    if ref is None:
        if current_contract is not None:
            raise ValueError("current_contract requires an intraday reference")
        current = None
    else:
        if current_contract is None:
            raise ValueError("An intraday decision requires current_contract")
        current = _vector(current_contract, "current_contract", n,
                          nonnegative=True)
    scales = tuple(float(scale) for scale in margin_scales)
    if (any(not np.isfinite(scale) or scale <= 0 for scale in scales)
            or len(set(scales)) != len(scales)):
        raise ValueError("margin_scales must contain distinct positive finite values")

    generation_start = time.perf_counter()
    candidates = []

    def add_lp(name, candidate_net, candidate_prices, candidate_weights):
        solution = old.solve_lp(candidate_net, candidate_prices,
                                candidate_weights, energy, base_settings,
                                reference=ref, reserve=rho, risk=0.0)
        grid = _vector(solution["grid"], f"{name} grid", n, nonnegative=True)
        candidates.append({"name": name, "grid": grid.copy(),
                           "plan_objective": float(solution["objective"])})

    add_lp("scenario_lp", net, prices, weight)
    add_lp("point_lp", forecast_net[None, :], forecast_prices[None, :],
           np.ones(1))
    mean_net = weight @ net
    sigma = np.sqrt(weight @ ((net - mean_net) ** 2))
    for scale in scales:
        add_lp(f"margin_{scale:g}sigma", (forecast_net + scale * sigma)[None, :],
               forecast_prices[None, :], np.ones(1))
    if current is not None:
        candidates.append({"name": "no_adjust", "grid": current.copy()})
    generation_seconds = time.perf_counter() - generation_start

    replay_start = time.perf_counter()
    replay = _replay_validated(np.stack([c["grid"] for c in candidates]),
                               net, prices, weight, energy, base_settings,
                               ref, beta)
    for k, candidate in enumerate(candidates):
        for key, value in replay.items():
            if key not in ("per_scenario", "terminal_beta"):
                candidate[key] = float(value[k])
    selected_index = int(np.argmin(replay["score"]))
    chosen = candidates[selected_index]
    baseline_score = float(replay["score"][0])
    tolerance = 1e-9 * max(1.0, abs(baseline_score))
    if (chosen["score"] > baseline_score + tolerance
            or chosen["score"] > float(np.min(replay["score"])) + tolerance):
        raise AssertionError("Selected score is not the candidate-set minimum")
    replay_seconds = time.perf_counter() - replay_start
    return {
        "grid": chosen["grid"].copy(),
        "selected_name": chosen["name"],
        "candidates": candidates,
        "baseline_score": baseline_score,
        "score_improvement": baseline_score - chosen["score"],
        "terminal_beta": beta,
        "generation_seconds": generation_seconds,
        "replay_seconds": replay_seconds,
    }
