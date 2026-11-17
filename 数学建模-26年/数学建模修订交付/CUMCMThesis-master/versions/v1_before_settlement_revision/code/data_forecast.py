"""Read the contest inputs and generate causal, reproducible forecasts.

Run from any directory with Python 3.11+ and numpy/openpyxl installed::

    python code/data_forecast.py

Inputs are read-only.  All powers are kW, prices are yuan/kWh.  Index t = 0
denotes 00:00--00:10, whose right endpoint is the supplied label 00:10.
Forecasts used for evaluation are chosen on January only.  Every forecast
on day d reads measurements with day index strictly smaller than d.
The published PV forecast additionally uses measurements already available
at its issue time, never measurements after that time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import openpyxl


HERE = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = HERE.parent / "附件"
MODEL_NAMES = (
    "last_day", "mean_3", "mean_7", "mean_14", "mean_28",
    "last_week", "weekday_2", "weekday_4", "weekday_blend_75",
    "weekday_blend_50", "weekday_trend", "mean_7_trend",
)
ENDPOINT_HOURS = np.arange(1, 145, dtype=float) / 6.0


def read_rows(path: Path, sheet: int = 0) -> list[tuple]:
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        return list(book.worksheets[sheet].values)
    finally:
        book.close()


def read_daily(path: Path, sheet: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rows = read_rows(path, sheet)
    dates = np.array([r[0].strftime("%Y-%m-%d") for r in rows[1:]])
    values = np.asarray([r[1:145] for r in rows[1:]], dtype=float)
    assert values.shape == (365, 144), (path, sheet, values.shape)
    assert np.all(np.isfinite(values)), (path, "nonfinite observations")
    assert np.all(values >= 0), (path, "negative observations")
    expected = np.array([(datetime(2025, 1, 1) + timedelta(days=d)).strftime("%Y-%m-%d")
                         for d in range(365)])
    assert np.array_equal(dates, expected), (path, "dates missing or unordered")
    for i, v in enumerate(rows[0][1:144]):
        assert v.hour * 60 + v.minute == (i + 1) * 10, (path, i, v)
    assert str(rows[0][144]) == "0:00+1", (path, "last endpoint")
    return dates, values


def forecast_one(history: np.ndarray, day: int, name: str,
                 prior: np.ndarray) -> np.ndarray:
    """Forecast one day. The only permitted actual values are history[:day]."""
    if day == 0:
        return prior.copy()
    recent7 = history[max(0, day - 7):day].mean(axis=0)
    if name == "last_day":
        return history[day - 1].copy()
    if name.startswith("mean_") and not name.endswith("trend"):
        window = int(name.split("_")[1])
        return history[max(0, day - window):day].mean(axis=0)
    if name == "last_week":
        return history[day - 7].copy() if day >= 7 else recent7
    week_indices = np.arange(day - 7, max(-1, day - 29), -7, dtype=int)
    if len(week_indices):
        same_weekday = history[week_indices].mean(axis=0)
    else:
        same_weekday = recent7
    if name.startswith("weekday_") and name.split("_")[-1].isdigit() and "blend" not in name:
        n = int(name.split("_")[-1])
        return history[week_indices[:n]].mean(axis=0) if len(week_indices) else recent7
    if name.startswith("weekday_blend_"):
        weight = int(name.split("_")[-1]) / 100
        return weight * same_weekday + (1 - weight) * recent7
    if name in ("weekday_trend", "mean_7_trend"):
        # Past seven-day level relative to the preceding seven days; clipped
        # to avoid excessive extrapolation from very small past PV output.
        if day >= 14:
            ratio = float(np.clip(history[day - 7:day].mean() /
                                  max(history[day - 14:day - 7].mean(), 1e-8), 0.85, 1.15))
        else:
            ratio = 1.0
        base = history[day - 7] if name == "weekday_trend" and day >= 7 else recent7
        return base * ratio
    raise ValueError(name)


def metrics(actual: np.ndarray, prediction: np.ndarray) -> dict:
    valid = np.isfinite(actual) & np.isfinite(prediction)
    if not np.any(valid):
        return {"n": 0, "mae": None, "rmse": None, "bias_actual_minus_forecast": None}
    error = actual[valid] - prediction[valid]
    return {"n": int(valid.sum()), "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "bias_actual_minus_forecast": float(np.mean(error))}


def select_and_forecast(actual: np.ndarray, prior: np.ndarray) -> tuple[np.ndarray, dict]:
    """Select on 15--31 January; evaluate on 1 February--31 December.

    January predictions use an online selection from strictly earlier January
    backtest errors, initially mean_7. From February onward the selected model
    is frozen, with its historical inputs updated once per day. January
    residuals therefore remain available as causal scenario calibration data.
    """
    candidates = {name: np.stack([forecast_one(actual, d, name, prior)
                                  for d in range(365)]) for name in MODEL_NAMES}
    tuning = {name: metrics(actual[14:31], pred[14:31])
              for name, pred in candidates.items()}
    selected = min(MODEL_NAMES, key=lambda k: tuning[k]["rmse"])
    forecast = candidates[selected].copy()
    january_model = []
    for d in range(31):
        if d <= 14:
            name = "mean_7"
        else:
            name = min(MODEL_NAMES,
                       key=lambda k: np.mean((actual[14:d] - candidates[k][14:d]) ** 2))
        forecast[d] = candidates[name][d]
        january_model.append(name)
    forecast = np.maximum(forecast, 0)
    month = np.array([int((datetime(2025, 1, 1) + timedelta(days=d)).month)
                      for d in range(365)])
    return forecast, {
        "selected_model": selected,
        "selection_metric": "RMSE",
        "selection_period": ["2025-01-15", "2025-01-31"],
        "candidate_january_metrics": tuning,
        "january_causal_models": january_model,
        "evaluation_period": ["2025-02-01", "2025-12-31"],
        "evaluation": metrics(actual[31:], forecast[31:]),
        "monthly_evaluation": {str(m): metrics(actual[month == m], forecast[month == m])
                               for m in range(2, 13)},
        "baseline_last_day_evaluation": metrics(actual[31:], candidates["last_day"][31:]),
        "baseline_mean_7_evaluation": metrics(actual[31:], candidates["mean_7"][31:]),
    }


def published_pv(input_dir: Path, dates: np.ndarray,
                 actual_pv: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    rows = read_rows(input_dir / "附件3.xlsx")[1:]
    assert len(rows) == 365 * 4
    raw = np.empty((365, 4, 24), dtype=float)
    output = np.full((365, 4, 144), np.nan)
    carried_date = None
    for i, row in enumerate(rows):
        if row[0]:
            carried_date = datetime.strptime(str(row[0]), "%Y-%m-%d").strftime("%Y-%m-%d")
        d, issue = divmod(i, 4)
        assert carried_date == dates[d]
        issue_hour = issue * 6
        assert int(str(row[1]).split(":")[0]) == issue_hour
        raw[d, issue] = np.asarray(row[2:26], dtype=float)
        assert np.all(np.isfinite(raw[d, issue]))
        assert np.all(raw[d, issue] >= 0)
        # At 06:00 the measurement labelled 06:00 has already been observed.
        # It is the anchor, not an unavailable newly issued 0-hour forecast.
        if issue_hour == 0:
            anchor = actual_pv[d - 1, -1] if d else 0.0
        else:
            anchor = actual_pv[d, issue_hour * 6 - 1]
        xp = issue_hour + np.arange(25, dtype=float)
        fp = np.r_[anchor, raw[d, issue]]
        valid = ENDPOINT_HOURS > issue_hour
        output[d, issue, valid] = np.maximum(np.interp(ENDPOINT_HOURS[valid], xp, fp), 0)
    reports = {}
    for issue in range(4):
        h = issue * 6
        reports[f"issue_{h:02d}h_remaining_day"] = metrics(actual_pv[31:], output[31:, issue])
        start, stop = h * 6, min((h + 6) * 6, 144)
        reports[f"issue_{h:02d}h_next_6h"] = metrics(actual_pv[31:, start:stop],
                                                    output[31:, issue, start:stop])
        reports[f"issue_{h:02d}h_vs_midnight_same_future"] = {
            "updated": metrics(actual_pv[31:, start:], output[31:, issue, start:]),
            "midnight": metrics(actual_pv[31:, start:], output[31:, 0, start:]),
        }
    return raw, output, reports


def summary(values: np.ndarray) -> dict:
    return {"shape": list(values.shape), "missing": int(np.isnan(values).sum()),
            "negative": int((values < 0).sum()), "zero": int((values == 0).sum()),
            "min": float(np.nanmin(values)), "max": float(np.nanmax(values)),
            "mean": float(np.nanmean(values)), "std": float(np.nanstd(values))}


def update_issue_forecast(actual: np.ndarray, midnight: np.ndarray,
                          positive_floor: float = 0.0) -> np.ndarray:
    """Correct future baseline using only the last two observed hours.

    Bias is actual minus the original midnight forecast, not minus a future
    revision. The correction decays with a six-hour time constant and is
    clipped pointwise to 30% of that future baseline.
    """
    updated = np.full((365, 4, 144), np.nan)
    updated[:, 0, :] = midnight
    for issue in range(1, 4):
        start = issue * 36
        bias = (actual[:, start - 12:start] - midnight[:, start - 12:start]).mean(axis=1)
        lead = np.arange(1, 145 - start, dtype=float)
        baseline = midnight[:, start:]
        correction = bias[:, None] * np.exp(-lead[None, :] / 36.0)
        correction = np.clip(correction, -0.30 * baseline, 0.30 * baseline)
        updated[:, issue, start:] = np.maximum(baseline + correction, positive_floor)
    return updated


def residual_dependence(load_residual: np.ndarray, pv_residual: np.ndarray,
                        price_residual: np.ndarray, start: int = 31) -> dict:
    residuals = {"load": load_residual[start:], "pv": pv_residual[start:],
                 "price": price_residual[start:]}
    return {
        "period": "2025-02-01--2025-12-31, descriptive only, not model tuning",
        "within_day_lag1_correlation": {
            k: float(np.corrcoef(v[:, :-1].ravel(), v[:, 1:].ravel())[0, 1])
            for k, v in residuals.items()},
        "same_interval_cross_correlation": {
            f"{a}_{b}": float(np.corrcoef(residuals[a].ravel(), residuals[b].ravel())[0, 1])
            for a, b in (("load", "pv"), ("load", "price"), ("pv", "price"))},
    }


def causality_checks(actual: np.ndarray, baseline: np.ndarray,
                      prior: np.ndarray, selected: str) -> dict:
    """Perturb unavailable actual values and check forecasts stay identical."""
    daily_checks = 0
    issue_checks = 0
    for day in (31, 78, 171, 265, 364):
        perturbed = actual.copy()
        perturbed[day:] += 123456.0
        assert np.array_equal(forecast_one(actual, day, selected, prior),
                              forecast_one(perturbed, day, selected, prior))
        daily_checks += 1
        for issue in (1, 2, 3):
            start = issue * 36
            perturbed = actual.copy()
            perturbed[day, start:] += 123456.0
            perturbed[day + 1:] += 123456.0
            before = update_issue_forecast(actual, baseline)[day, issue, start:]
            after = update_issue_forecast(perturbed, baseline)[day, issue, start:]
            assert np.array_equal(before, after)
            issue_checks += 1
    return {"daily_future_perturbation_passed": daily_checks,
            "intraday_future_perturbation_passed": issue_checks}


def prepare(input_dir: Path = DEFAULT_INPUT, output_dir: Path = HERE / "data") -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    dates, load = read_daily(input_dir / "附件2.xlsx", 0)
    dates_pv, pv = read_daily(input_dir / "附件2.xlsx", 1)
    dates_price, price = read_daily(input_dir / "附件4.xlsx")
    assert np.array_equal(dates, dates_pv) and np.array_equal(dates, dates_price)
    q1 = np.asarray([r[1:4] for r in read_rows(input_dir / "附件1.xlsx")[1:]], dtype=float)
    assert q1.shape == (144, 3) and np.all(np.isfinite(q1))
    fixed_price, q1_load, q1_pv = q1.T
    forecast_load, load_report = select_and_forecast(load, q1_load)
    forecast_pv, pv_report = select_and_forecast(pv, q1_pv)
    forecast_price, price_report = select_and_forecast(price, fixed_price)
    forecast_price = np.maximum(forecast_price, 1e-6)
    raw_pv, issue_pv, issue_report = published_pv(input_dir, dates, pv)
    issue_load = update_issue_forecast(load, forecast_load)
    issue_price = update_issue_forecast(price, forecast_price, positive_floor=1e-6)
    arrays = {
        "load": load, "pv": pv, "price": price, "dates": dates,
        "fixed_price": fixed_price, "q1_load": q1_load, "q1_pv": q1_pv,
        "forecast_load": forecast_load, "forecast_pv_history": forecast_pv,
        "forecast_price": forecast_price, "pv_issue_forecast": issue_pv,
        "forecast_load_issue": issue_load, "forecast_price_issue": issue_price,
        "pv_hourly_raw": raw_pv,
        "residual_load": load - forecast_load,
        "residual_pv_history": pv - forecast_pv,
        "residual_price": price - forecast_price,
        "residual_pv_issue": pv[:, None, :] - issue_pv,
        "residual_load_issue": load[:, None, :] - issue_load,
        "residual_price_issue": price[:, None, :] - issue_price,
        "residual_net_history": (load - pv) - (forecast_load - forecast_pv),
        "residual_net_issue": ((load - pv)[:, None, :] - (issue_load - issue_pv)),
        "endpoint_hours": ENDPOINT_HOURS, "evaluation_day_indices": np.arange(31, 365),
    }
    np.savez_compressed(output_dir / "prepared.npz", **arrays)
    months = np.array([int(d[5:7]) for d in dates])
    report = {
        "sources": {f"附件{k}.xlsx": {"sha256": hashlib.sha256((input_dir / f"附件{k}.xlsx").read_bytes()).hexdigest()}
                    for k in range(1, 5)},
        "conventions": {
            "interval": "The right endpoint 00:10 represents 00:00--00:10; 24:00 is included in the preceding day.",
            "power_to_energy": "10-minute interval energy = supplied power / 6 (kWh). Each supplied sample is treated as interval mean power.",
            "daily_forecasts": "All actual inputs have day index < forecast day. Model chosen only from 15--31 January RMSE and frozen for February--December; coefficients use expanding available history within their fixed lookback.",
            "january_warmup": "Day 1 uses attachment 1 as a fixed prior; days 2--15 use mean of previous up to 7 days. January later forecasts choose from previously observed January errors only. January is not in performance evaluation.",
            "pv_issues": "24 forecast columns are issue+1h through issue+24h. Linear interpolation to 10-minute right endpoints, anchored on the observed PV at issue time (midnight uses previous-day 24:00; first January midnight uses zero). Only same-day future intervals are retained; earlier indices are NaN.",
            "load_price_issue_updates": "At 06/12/18h, mean(actual minus midnight prediction) over the 12 already observed preceding intervals corrects the future forecast. Correction at lead h intervals is bias*exp(-h/36), clipped pointwise to +/-30% of the future midnight baseline; resulting load >=0 and price >=1e-6. Pre-issue intervals are NaN. Parameters fixed a priori, without evaluation-period tuning.",
            "residuals": "actual minus forecast; whole days retain temporal and load-PV-price dependence. A decision on day d may only sample residual rows j < d. For issue-specific residuals, retain the same issue index; NaN indicates unavailable past intervals.",
            "missing_and_outliers": "No missing or negative raw load, PV or price values. All observed extremes are retained; no retrospective outlier deletion or interpolation.",
            "prices": "Attachment 4 is treated as realized transaction prices, not a known day-ahead future price path. At midnight use historical price prediction; settlement uses actual transaction price.",
        },
        "raw_summary": {"load_kw": summary(load), "pv_kw": summary(pv),
                        "price_yuan_per_kwh": summary(price), "pv_hourly_forecast_kw": summary(raw_pv)},
        "monthly_means": {str(m): {"load_kw": float(load[months == m].mean()),
                                   "pv_kw": float(pv[months == m].mean()),
                                   "price_yuan_per_kwh": float(price[months == m].mean())}
                          for m in range(1, 13)},
        "forecast_load": load_report, "forecast_pv_history": pv_report,
        "forecast_price": price_report, "published_pv_evaluation": issue_report,
        "load_issue_evaluation": {str(i * 6): metrics(load[31:], issue_load[31:, i]) for i in range(4)},
        "price_issue_evaluation": {str(i * 6): metrics(price[31:], issue_price[31:, i]) for i in range(4)},
        "issue_updates_same_future_comparison": {
            label: {str(i * 6): {"updated": metrics(actual[31:, i * 36:], updated[31:, i, i * 36:]),
                                 "midnight": metrics(actual[31:, i * 36:], midnight[31:, i * 36:])}
                    for i in range(1, 4)}
            for label, actual, updated, midnight in (
                ("load", load, issue_load, forecast_load),
                ("price", price, issue_price, forecast_price))},
        "residual_dependence": residual_dependence(load - forecast_load, pv - forecast_pv,
                                                    price - forecast_price),
        "causality_checks": {
            "load": causality_checks(load, forecast_load, q1_load, load_report["selected_model"]),
            "price": causality_checks(price, forecast_price, fixed_price, price_report["selected_model"]),
        },
        "interface": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()},
    }
    (output_dir / "data_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=HERE / "data")
    args = parser.parse_args()
    r = prepare(args.input_dir, args.output_dir)
    for field in ("forecast_load", "forecast_pv_history", "forecast_price"):
        print(field, r[field]["selected_model"], r[field]["evaluation"])
    print("Wrote", args.output_dir / "prepared.npz")
    print("Wrote", args.output_dir / "data_report.json")
