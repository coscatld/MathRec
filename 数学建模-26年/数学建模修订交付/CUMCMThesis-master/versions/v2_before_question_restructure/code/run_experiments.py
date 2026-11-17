"""Reproduce control-policy ablations without overwriting primary schedules.

All cases simulate 1 January--31 December, including a causal January warmup.
Reported comparison metrics use 1 February--31 December. The energy carried
into February and remaining after December are reported explicitly.

Run: py code/run_experiments.py
Outputs: data/experiments.json and data/experiment_daily_metrics.npz.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from optimize_dispatch import ROOT, Settings, simulate, summary


EXPERIMENTS = [
    ("q2_no_storage", "q2", {"no_storage": True, "risk_scale": 0., "reserve_scale": 0.}),
    ("q2_point_forecast", "q2", {"deterministic": True, "risk_scale": 0., "reserve_scale": 0.}),
    ("q2_sp_expectation", "q2", {"risk_scale": 0., "reserve_scale": 0.}),
    ("q2_cvar_no_reserve", "q2", {"reserve_scale": 0.}),
    ("q3_midnight_only", "q3", {"updates": (0,)}),
    ("q3_00_06", "q3", {"updates": (0, 36)}),
    ("q3_00_06_12", "q3", {"updates": (0, 36, 72)}),
    ("q3_reserve_0p5", "q3", {"reserve_scale": .5}),
    ("q3_reserve_1p5", "q3", {"reserve_scale": 1.5}),
    ("q3_risk_0", "q3", {"risk_scale": 0.}),
    ("q3_risk_0p2", "q3", {"risk_scale": .2}),
    ("q3_terminal_0", "q3", {"terminal_scale": 0.}),
    ("q3_refund_interpretation", "q3", {"cancellation_refund": True}),
]


def extended_summary(result: dict) -> dict:
    out = summary(result, start=31)
    r = {k: v[31:] for k, v in result.items()}
    delta = r["adjusted"] - r["plan"]
    out.update({
        "adjustment_up_energy": float(np.maximum(delta, 0).sum()),
        "adjustment_down_energy": float(np.maximum(-delta, 0).sum()),
        "adjustment_up_intervals": int((delta > 1e-6).sum()),
        "adjustment_down_intervals": int((delta < -1e-6).sum()),
        "stored_energy_change": float(r["energy"][-1, -1] - r["energy"][0, 0]),
        "simulation_initial_energy": float(result["energy"][0, 0]),
        "evaluation_initial_energy": float(r["energy"][0, 0]),
        "evaluation_final_energy": float(r["energy"][-1, -1]),
        "full_year_total_cost": float(result["total_cost"].sum()),
        "full_year_emergency_energy": float(result["emergency"].sum()),
        "january_warmup_cost": float(result["total_cost"][:31].sum()),
    })
    return out


def differences(experiments: dict) -> dict:
    """Signed differences avoid presenting deteriorations as improvements."""
    out = {}
    for name, result in experiments.items():
        base_name = "q2_proposed" if result["case"] == "q2" else "q3_proposed"
        if base_name not in experiments or name == base_name:
            continue
        reference = experiments[base_name]["summary"]
        current = result["summary"]
        out[name] = {
            "reference": base_name,
            "cost_minus_reference": current["total_cost"] - reference["total_cost"],
            "cost_percent_vs_reference": 100 * (current["total_cost"] / reference["total_cost"] - 1),
            "emergency_kwh_minus_reference": current["emergency_energy"] - reference["emergency_energy"],
            "daily_cost_cvar95_minus_reference": current["daily_cost_cvar95"] - reference["daily_cost_cvar95"],
            "final_energy_minus_reference": current["evaluation_final_energy"] - reference["evaluation_final_energy"],
        }
    return out


def save_report(report: dict, daily: dict, output: Path):
    report["differences_vs_proposed"] = differences(report["experiments"])
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(output.parent / "experiment_daily_metrics.npz", **daily)


def collect_daily(daily: dict, name: str, result: dict):
    for field in ("total_cost", "plan_cost", "adjustment_cost", "emergency_cost", "risk"):
        daily[f"{name}__{field}"] = result[field][31:]
    for field in ("plan", "adjusted", "emergency", "spill", "charge", "discharge"):
        daily[f"{name}__{field}_energy"] = result[field][31:].sum(axis=1)
    daily[f"{name}__initial_energy"] = result["energy"][31:, 0]
    daily[f"{name}__final_energy"] = result["energy"][31:, -1]


def run(output: Path, selected: list[str] | None = None):
    data_path = ROOT / "data/prepared.npz"
    data = dict(np.load(data_path, allow_pickle=False))
    report = {
        "simulation_period": ["2025-01-01", "2025-12-31"],
        "evaluation_period": ["2025-02-01", "2025-12-31"],
        "evaluation_days": 334,
        "january_role": "Causal warmup, forecast calibration and state carryover; excluded from reported policy performance.",
        "fairness_notes": [
            "Every case starts on 1 January at 6000 kWh and uses the same causal forecast data. February opening states can differ because each policy has its own January warmup; these states are reported.",
            "Update ablations share the same midnight PV forecast source, solver and default parameters. Each policy evolves its own actual energy state across the year, so its later daily midnight purchase quantities need not coincide.",
            "Terminal-penalty removal can lower expenditure partly by consuming stored energy; report both opening and closing inventories, and do not equate expenditure reduction with inventory-neutral savings.",
            "The primary settlement keeps original planned purchase charges and adds 0.5 times price on reductions. The alternate settlement assumes half-price refunds instead; it is a separate interpretation sensitivity.",
            "No evaluation-period parameter optimization is performed. All requested variants are retained, including variants that worsen cost or emergency purchases.",
        ],
        "sources": {
            "prepared_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            "solver_sha256": hashlib.sha256((ROOT / "code/optimize_dispatch.py").read_bytes()).hexdigest(),
        },
        "experiments": {},
    }
    daily = {"dates": data["dates"][31:]}
    for case in ("q2", "q3"):
        path = ROOT / f"data/{case}_full.npz"
        if path.exists():
            result = dict(np.load(path, allow_pickle=False))
            if len(result["plan"]) == 365:
                name = f"{case}_proposed"
                report["experiments"][name] = {
                    "case": case, "settings": asdict(Settings()),
                    "source": path.name, "summary": extended_summary(result),
                    "reuse_primary": True,
                }
                collect_daily(daily, name, result)
    save_report(report, daily, output)
    for name, case, overrides in EXPERIMENTS:
        if selected and name not in selected:
            continue
        print(f"START {name}: {overrides}", flush=True)
        started = time.perf_counter()
        cfg = Settings(**overrides)
        result = simulate(data, case, cfg, days=365, progress=False)
        elapsed = time.perf_counter() - started
        report["experiments"][name] = {
            "case": case, "settings": asdict(cfg),
            "summary": extended_summary(result), "run_seconds": elapsed,
            "reuse_primary": False,
        }
        collect_daily(daily, name, result)
        save_report(report, daily, output)
        s = report["experiments"][name]["summary"]
        print(f"DONE {name}: cost={s['total_cost']:.4f}, emergency={s['emergency_energy']:.4f}, "
              f"initialE={s['evaluation_initial_energy']:.4f}, finalE={s['evaluation_final_energy']:.4f}, "
              f"seconds={elapsed:.2f}", flush=True)
    report["completed"] = True
    save_report(report, daily, output)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/experiments.json")
    parser.add_argument("--cases", nargs="*", choices=[v[0] for v in EXPERIMENTS])
    args = parser.parse_args()
    run(args.output, args.cases)
