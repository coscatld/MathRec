"""Read-only numerical audit of the final delivery, without annual optimization.

The sole output is outputs/final_strategy/audit.json.  Its original immutable
source snapshot is retained across runs.  Old manifests are archival evidence:
authorized manuscript, figure and attachment edits need not match old hashes.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import sys
import traceback

import numpy as np
import openpyxl

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/final_strategy"
REPORT = OUT / "audit.json"
CASES = ("q2", "q3", "q4_2", "q4_3")
SOURCES = {c: (f"outputs/outer_candidate_test/{c}_selected.npz"
               if c in ("q3", "q4_3") else f"data/{c}_v2.npz") for c in CASES}
WORKBOOKS = {"q2": "result2.xlsx", "q3": "result3.xlsx",
             "q4_2": "result4-2.xlsx", "q4_3": "result4-3.xlsx"}
DATES = ("2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21")
TOL = 1e-5


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(actual, expected, label, atol=TOL):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    assert a.shape == b.shape, (label, a.shape, b.shape)
    assert np.isfinite(a).all() and np.isfinite(b).all(), label
    error = float(np.max(np.abs(a-b), initial=0.))
    assert error < atol, (label, error, atol)
    return error


def arrays(path):
    with np.load(ROOT / path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def metrics(a):
    # Sort the worst 5% explicitly, including its fractional boundary day.
    cost = np.sort(a["total_cost"])[::-1]
    tail = .05*len(cost)
    whole = int(tail)
    cvar = (float(cost[:whole].sum())+(tail-whole)*float(cost[whole]))/tail
    return {
        "days": len(cost),
        **{key: float(a[key].sum()) for key in
           ("plan_cost", "adjustment_cost", "emergency_cost", "total_cost")},
        **{key+"_energy": float(a[key].sum()) for key in
           ("plan", "adjusted", "emergency", "spill", "charge", "discharge")},
        "emergency_days": int(np.count_nonzero(a["emergency"].sum(axis=1)>1e-6)),
        "emergency_intervals": int(np.count_nonzero(a["emergency"]>1e-6)),
        "daily_cost_cvar95": cvar,
        "daily_cost_max": float(cost[0]),
        "initial_energy": float(a["energy"][0, 0]),
        "final_energy": float(a["energy"][-1, -1]),
        "max_actual_grid_power": float((a["adjusted"]+a["emergency"]).max()*6),
    }


def physical_and_bill(case, a):
    prepared = arrays(f"data/prepared_v2_{case}.npz")
    ids = np.array([list(prepared["dates"]).index(x) for x in a["dates"]])
    assert len(ids) == 334
    assert str(a["dates"][0]) == "2025-02-01" and str(a["dates"][-1]) == "2025-12-31"
    net = (prepared["load"][ids]-prepared["pv"][ids])/6
    prices = prepared["price"][ids] if case.startswith("q4") else prepared["fixed_price"][None, :]
    state = a["energy"][:, :-1]
    surplus = np.maximum(a["adjusted"]-net, 0.)
    deficit = np.maximum(net-a["adjusted"], 0.)
    charge = np.minimum(surplus, np.minimum(5000/6, np.maximum(10800-state, 0.)/.9))
    discharge = np.minimum(deficit, np.minimum(5000/6, np.maximum(state-1200, 0.)*.9))
    errors = {
        "balance_kwh": close(a["adjusted"]+a["emergency"]+a["discharge"],
                             net+a["charge"]+a["spill"], case+" balance"),
        "charge_kwh": close(a["charge"], charge, case+" greedy charge"),
        "discharge_kwh": close(a["discharge"], discharge, case+" greedy discharge"),
        "emergency_kwh": close(a["emergency"], deficit-discharge, case+" emergency"),
        "spill_kwh": close(a["spill"], surplus-charge, case+" spill"),
        "soc_kwh": close(np.diff(a["energy"], axis=1),
                         .9*a["charge"]-a["discharge"]/.9, case+" storage dynamics"),
        "continuity_kwh": close(a["energy"][1:, 0], a["energy"][:-1, -1], case+" continuity"),
    }
    assert a["energy"].min() >= 1200-TOL and a["energy"].max() <= 10800+TOL
    delta = a["adjusted"]-a["plan"]
    assert delta.min() >= -TOL, case+" frozen no-refund contract cannot economically decrease"
    if case in ("q2", "q4_2"):
        close(delta, np.zeros_like(delta), case+" day-ahead unchanged plan")
    else:
        close(delta[:, :36], np.zeros((334, 36)), case+" midnight installed plan")
    # Entire original plan is settled even when electricity is unused/spilled.
    bills = {"plan_cost": np.sum(prices*a["plan"], axis=1),
             "adjustment_cost": np.sum(prices*(1.5*np.maximum(delta, 0.)+.5*np.maximum(-delta, 0.)), axis=1),
             "emergency_cost": np.sum(5*prices*a["emergency"], axis=1)}
    bills["total_cost"] = sum(bills.values())
    errors.update({key+"_yuan": close(a[key], value, case+" "+key) for key, value in bills.items()})
    return {"passed": True, "source_path": SOURCES.get(case), "days": len(ids),
            "maximum_errors": errors, "metrics": metrics(a),
            "terminal_penalty_excluded_from_realized_bill": True,
            "power_to_energy_dt_hours": 1/6,
            "balance_inequality_excess_equals_spill": True}


def immutable_check(snapshot):
    mismatches = [p for p, digest in snapshot.items() if not (ROOT/p).exists() or sha(ROOT/p) != digest]
    assert not mismatches, ("immutable evidence changed", mismatches)
    for case in CASES:
        summary = read(ROOT/f"outputs/outer_candidate_test/{case}_summary.json")
        for path, digest in summary["source_sha256"].items():
            assert sha(ROOT/path) == digest, (case, path)
    return {"passed": True, "verified_current_snapshot_count": len(snapshot),
            "candidate_run_source_hashes_match": True,
            "old_manifest_sha256": sha(ROOT/"data/delivery_manifest.json"),
            "archival_timing_manifest_sha256": sha(ROOT/"outputs/outer_timing_ablation/run_manifest.json"),
            "policy": "Old manifest files themselves are immutable; old manuscript and presentation hashes are not current-delivery constraints."}


def source_checks():
    selection = read(ROOT/"data/selection_v2.json")
    result = {}
    for case in CASES:
        cfg = selection["cases"][case]["config"]
        assert cfg["controller"] == "greedy" and cfg["risk_mode"] == cfg["reserve_mode"] == "none"
        assert not cfg["cross_day"] and not cfg["no_storage"] and not cfg["cancellation_refund"]
        a = arrays(SOURCES[case])
        original = arrays(f"data/{case}_v2.npz")
        close(a["energy"][0, 0], original["energy"][0, 0], case+" common opening SOC")
        result[case] = physical_and_bill(case, a)
    return result


def decision_checks():
    out = {}
    for case in ("q3", "q4_3"):
        a = arrays(SOURCES[case])
        rows = read(ROOT/f"outputs/outer_candidate_test/{case}_decisions.json")
        assert len(rows) == 334*4
        max_error = 0.
        counts = {}
        for i, date in enumerate(a["dates"]):
            day_rows = rows[4*i:4*i+4]
            assert [r["slot"] for r in day_rows] == [0, 36, 72, 108]
            for row in day_rows:
                slot = row["slot"]
                assert row["date"] == str(date)
                close(row["initial_energy_kwh"], a["energy"][i, slot], case+" actual decision SOC")
                entries = row["candidates"]
                names = [x["name"] for x in entries]
                expected = ["scenario_lp", "point_lp", "margin_0.25sigma", "margin_0.5sigma", "margin_1sigma"]
                assert names == expected+(["no_adjust"] if slot else []), names
                score = np.asarray([entry["score"] for entry in entries])
                winner = entries[int(np.argmin(score))]
                assert winner["name"] == row["selected_name"], row
                counts[winner["name"]] = counts.get(winner["name"], 0)+1
                close(row["estimated_score_reduction_yuan"], score[0]-score.min(), case+" score difference")
                for entry in entries:
                    max_error = max(max_error,
                        close(entry["expected_bill"], sum(entry[key] for key in (
                            "expected_purchase_cost", "expected_adjustment_cost", "expected_emergency_cost")), case+" replay bill"),
                        close(entry["score"], entry["expected_bill"]+entry["expected_terminal_penalty"], case+" replay score"))
        out[case] = {"passed": True, "legal_updates": len(rows), "candidate_count_midnight": 5,
                     "candidate_count_intraday": 6, "argmin_and_first_tie_match": True,
                     "same_current_soc_verified": True, "max_component_error_yuan": max_error,
                     "selected_counts": counts}
    # Focused independent oracles call the unchanged scalar executor; no annual rerun.
    import check_outer_candidate_selector as checks
    out["focused_regressions"] = {
        "scalar_replay": checks.check_scalar_equivalence(),
        "paired_q4_prices": checks.check_paired_price_shortage(),
        "no_adjust_and_original_reference": checks.check_original_reference_and_no_adjust(),
        "scenario_terminal_penalty": checks.check_actual_terminal_and_common_beta(),
    }
    assert read(ROOT/"outputs/outer_timing_ablation/audit.json")["passed"]
    assert read(ROOT/"outputs/outer_candidate_test/checks.json")["passed"]
    return out


def timing_checks():
    expected = {"q3": [13975810.86, 13911957.70, 13947087.88, 13871471.44],
                "q4_3": [14769232.53, 14712596.45, 14734593.23, 14670237.87]}
    out = {}
    for case, bills in expected.items():
        paths = [f"data/{case}_v2.npz", f"outputs/outer_timing_ablation/{case}_midnight_only.npz",
                 f"outputs/outer_timing_ablation/{case}_intraday_only.npz", SOURCES[case]]
        variants = [arrays(path) for path in paths]
        total = [float(a["total_cost"].sum()) for a in variants]
        for a in variants:
            close(a["energy"][1:, 0], a["energy"][:-1, -1], case+" independent continuous SOC")
            close(a["energy"][0, 0], variants[0]["energy"][0, 0], case+" common initial SOC")
        close(total, bills, case+" four timing totals", atol=.0051)
        assert int(np.argmin(total)) == 3
        out[case] = {"passed": True, "order": ["mother", "midnight", "intraday", "all"],
                     "actual_bills_yuan": total, "midnight_minus_all_yuan": total[1]-total[3],
                     "independent_continuous_states": True}
    close(out["q3"]["midnight_minus_all_yuan"], 40486.25, "unrounded midnight difference", atol=.0051)
    for case, percent in (("q2", .161), ("q4_2", .428)):
        main = arrays(f"data/{case}_v2.npz")["total_cost"].sum()
        selected = arrays(f"outputs/outer_candidate_test/{case}_selected.npz")["total_cost"].sum()
        increase = float(100*(selected/main-1))
        close(increase, percent, case+" rejected day-ahead increase", atol=.00051)
        out[case] = {"retained_mother": True, "rejected_screening_increase_percent": increase}
    return out


def export_checks():
    export = read(OUT/"results_export.json")
    old = read(ROOT/"data/results_export.json")
    assert export["q1"] == old["q1"], "Q1 export changed"
    for case in CASES:
        a = arrays(SOURCES[case])
        b = export["cases"][case]
        for key, value in a.items():
            if key == "dates":
                assert list(value) == b[key], (case, key)
            else:
                close(b[key], value, case+" export "+key)
        # Enhanced arrays have no saved LP-reference trajectory: no invented RMSE.
        if case in ("q3", "q4_3"):
            assert "reference_energy" not in b, case+" inherited mother reference trajectory"
    summary = read(OUT/"summary.json")
    cases = summary.get("cases", summary)
    for case in CASES:
        actual = cases[case].get("summary", cases[case])
        a = arrays(SOURCES[case])
        required = {"days","plan_cost","adjustment_cost","emergency_cost","total_cost",
                    "emergency_energy","emergency_days","emergency_intervals",
                    "daily_cost_cvar95","daily_cost_max","initial_energy","final_energy"}
        for key, value in metrics(a).items():
            if key not in required and key not in actual:
                continue
            assert key in actual, (case, "missing final metric", key)
            close(actual[key], value, case+" summary "+key)
        for month in actual["monthly"]:
            mask = np.array([str(date)[:7] == month["month"] for date in a["dates"]])
            close(month["total_cost"], a["total_cost"][mask].sum(), case+" monthly bill")
            close(month["emergency_energy"], a["emergency"][mask].sum(), case+" monthly emergency")
        assert len(actual["monthly"]) == 11
        assert actual["max_cost_date"] == str(a["dates"][np.argmax(a["total_cost"])])
        assert actual["max_emergency_date"] == str(a["dates"][np.argmax(a["emergency"].sum(axis=1))])
        close(actual["emergency_segments"], sum(len(segments(day)) for day in a["emergency"]), case+" segment count")
    with (OUT/"daily_summary.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1336
    for case in CASES:
        a = arrays(SOURCES[case])
        selected = [row for row in rows if row["case"] == case]
        assert [row["date"] for row in selected] == list(a["dates"])
        values = {"plan_kwh": a["plan"].sum(axis=1), "adjusted_kwh": a["adjusted"].sum(axis=1),
                  "emergency_kwh": a["emergency"].sum(axis=1), "energy_start_kwh": a["energy"][:, 0],
                  "energy_end_kwh": a["energy"][:, -1],
                  **{k+"_yuan": a[k] for k in ("plan_cost", "adjustment_cost", "emergency_cost", "total_cost")}}
        for key, expected in values.items():
            close([float(row[key]) for row in selected], expected, case+" daily CSV "+key)
    return {"passed": True, "exact_array_export": True, "q1_unchanged": True,
            "daily_csv_rows": len(rows), "required_summary_metrics_per_case": 13,
            "monthly_metrics_recomputed":True, "maximum_day_selection_recomputed":True}


def clock(slot):
    return f"{slot//6:02d}:{slot%6*10:02d}"


def segments(day, separator="-"):
    out = []
    t = 0
    while t < 144:
        if day[t] <= 1e-7:
            t += 1
            continue
        start = t
        while t < 144 and day[t] > 1e-7:
            t += 1
        out.append((clock(start)+separator+clock(t), float(day[start:t].sum())))
    return out


def workbook_checks():
    out = {}
    for case in CASES:
        path = ROOT.parent/"附件/附件5"/WORKBOOKS[case]
        a = arrays(SOURCES[case])
        with_cache = openpyxl.load_workbook(path, data_only=True, read_only=True)
        formulas = openpyxl.load_workbook(path, data_only=False, read_only=True)
        template = openpyxl.load_workbook(ROOT/"originals/附件5"/WORKBOOKS[case], read_only=True)
        assert with_cache.sheetnames == template.sheetnames
        checked = 0
        for sheet, key, cost in (("计划购电量", "plan", "plan_cost"), ("调整购电量", "adjusted", "total_cost")):
            if sheet not in with_cache.sheetnames:
                continue
            values = list(with_cache[sheet].iter_rows(values_only=True))
            formula_rows = list(formulas[sheet].iter_rows(values_only=True))
            assert values[0][:145] == tuple(c.value for c in next(template[sheet].iter_rows()))[:145]
            for i, row in enumerate(values[1:335]):
                assert row[0].date().isoformat() == str(a["dates"][i]), (case, sheet, i)
                checked += len(row[1:145])
                close(row[1:145], a[key][i], case+" workbook "+sheet)
                close(row[145:147], [a[key][i].sum(), a[cost][i]], case+" workbook totals")
                assert formula_rows[i+1][145] == f"=SUM(B{i+2}:EO{i+2})", (case, sheet, i)
        rows = list(with_cache["充放电量"].iter_rows(values_only=True))[1:]
        assert len(rows) == 2004
        for i, date in enumerate(a["dates"]):
            for j in range(6):
                row = rows[6*i+j]
                assert row[0].date().isoformat() == str(date)
                close(row[2:4], [a["charge"][i,j*24:(j+1)*24].sum(), a["discharge"][i,j*24:(j+1)*24].sum()], case+" workbook battery")
            close([rows[6*i][5], rows[6*i+1][5]], [a["energy"][i,0], a["energy"][i,-1]], case+" workbook SOC")
        emergency = list(with_cache["紧急购电量"].iter_rows(min_row=2, max_col=3, values_only=True))
        emergency = [row for row in emergency if row[0] is not None]
        expected = [(str(date), label, amount) for date, day in zip(a["dates"],a["emergency"])
                    for label, amount in (segments(day) or [("无",0.)])]
        assert len(emergency) == len(expected), (case,"emergency rows")
        for row, target in zip(emergency, expected):
            assert (row[0].date().isoformat(), row[1]) == target[:2], (case,row,target)
            close(row[2], target[2], case+" workbook emergency segment")
        for wb in (with_cache, formulas, template):
            wb.close()
        out[case] = {"passed": True, "path": str(path), "sha256":sha(path), "days":334,
                     "ten_minute_quantities_checked": checked, "battery_rows":2004,
                     "emergency_rows":len(expected)}
    return out


def table_checks():
    def require(text, fragment, label):
        assert fragment in text, (label, fragment)
    annual = (ROOT/"tables/annual_summary.tex").read_text()
    labels = [("计划费用/万元","plan_cost",1e4),("调整费用/万元","adjustment_cost",1e4),
              ("紧急费用/万元","emergency_cost",1e4),("实际总费用/万元","total_cost",1e4),
              ("紧急电量/kWh","emergency_energy",1),("紧急购电天数","emergency_days",1),
              ("日总费用CVaR95/元","daily_cost_cvar95",1),("最高日费用/元","daily_cost_max",1),
              ("2月1日期初库存/kWh","initial_energy",1),("12月31日期末库存/kWh","final_energy",1)]
    m = {case:metrics(arrays(SOURCES[case])) for case in CASES}
    for label,key,scale in labels:
        values = [str(m[c][key]) if key == "emergency_days" else f"{m[c][key]/scale:.2f}" for c in CASES]
        require(annual, " & ".join([label]+values), "annual "+label)
    assert "参考与实际SOC的RMSE" not in annual, "Unavailable enhanced reference SOC metric"
    for case in CASES:
        a = arrays(SOURCES[case])
        selected = (ROOT/f"tables/{case}_selected.tex").read_text()
        decomp = (ROOT/f"tables/{case}_decomposition.tex").read_text()
        for date in DATES:
            i = list(a["dates"]).index(date)
            parts = selected.split(date)
            assert len(parts) >= 4, (case, date, "purchase/storage/emergency headings")
            purchase, battery = parts[1], parts[2]
            for slot in (60,72,84,96,108,120):
                require(purchase, f"{clock(slot)}--{clock(slot+1)} & {a['adjusted'][i,slot]:.2f}",case+date)
            require(purchase, f"全天购电量 & {a['adjusted'][i].sum():.2f}", case+date)
            require(purchase, f"{{{a['total_cost'][i]:.2f}}}", case+date+" bill")
            for j in range(6):
                start,stop=j*24,(j+1)*24
                require(battery, f"{clock(start)}--{clock(stop)} & {a['charge'][i,start:stop].sum():.2f} & {a['discharge'][i,start:stop].sum():.2f}", case+date+" battery")
            require(battery, f"{{{a['energy'][i,0]:.2f}}}", case+date+" opening SOC")
            require(battery, f"{{{a['energy'][i,-1]:.2f}}}", case+date+" final SOC")
            require(decomp, " & ".join([date,f"{a['plan'][i].sum():.2f}",f"{a['adjusted'][i].sum():.2f}"]+
                    [f"{a[k][i]:.2f}" for k in ("plan_cost","adjustment_cost","emergency_cost")]),case+date+" decomposition")
        totals = " & ".join(f"合计 & {a['emergency'][list(a['dates']).index(d)].sum():.2f}" for d in DATES)
        require(selected, totals, case+" emergency totals")
    prior = ROOT/"outputs/final_strategy/before_root/tables"
    unchanged = []
    for path in prior.glob("v2_*.tex"):
        # Captions/notes may be edited; the old ablation values remain exact.
        def numerical_rows(text):
            return [re.findall(r"(?<![A-Za-z])[-+]?\d+\.\d+", line)
                    for line in text.splitlines() if " & " in line and re.search(r"\d+\.\d+",line)]
        old = numerical_rows(path.read_text())
        new = numerical_rows((ROOT/"tables"/path.name).read_text())
        assert old == new, ("mother ablation numerical data changed",path.name)
        unchanged.append(path.name)
    return {"passed":True, "annual_metrics":len(labels),"specified_dates_per_case":4,
            "mother_ablation_tables_numerically_unchanged":unchanged}


def asset_provenance_checks():
    manifest = read(OUT/"asset_manifest.json")
    assert manifest["generator_sha256"] == sha(ROOT/"code/publish_final_strategy.py")
    for path,digest in manifest["inputs_sha256"].items():
        assert sha(ROOT/path) == digest, ("asset input",path)
    for path,entry in manifest["products"].items():
        assert sha(ROOT/path) == entry["sha256"], ("asset product",path)
        for source,digest in entry["sources_sha256"].items():
            assert sha(ROOT/source) == digest, ("asset source",source)
    required = {"v2_control_difference":[SOURCES["q3"],"data/q3_v2.npz"],
                "v2_adjustments":[SOURCES["q3"]],
                "v2_monthly":list(SOURCES.values()),
                "q4_cost_waterfall":[SOURCES["q4_2"],SOURCES["q4_3"]]}
    for name,sources in required.items():
        for ext in ("png","pdf"):
            entry = manifest["products"][f"figures/results/{name}.{ext}"]
            assert set(sources) <= set(entry["sources_sha256"]), (name,"final array provenance")
    timing = (ROOT/"tables/outer_timing_ablation.tex").read_text().replace(",", "")
    for row in timing_checks().values():
        for cost in row.get("actual_bills_yuan",[]):
            assert f"{cost:.2f}" in timing, ("timing table",cost)
    return {"passed":True,"products_verified":len(manifest["products"]),
            "final_execution_figures_verified":list(required),
            "scope":"Source mappings and artifact hashes; visual and semantic QA remains with root agent."}


def main():
    prior = read(REPORT)
    snapshot = prior["immutable_before"]
    report = {"audit":"independent final-strategy delivery audit", "status":"RUNNING",
              "passed":False, "created_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
              "snapshot_at_utc":prior["snapshot_at_utc"], "immutable_before":snapshot,
              "auditor_script_sha256":sha(Path(__file__)), "annual_models_rerun":False,
              "checks":{}, "failures":[],
              "scope":"Independent arithmetic, array, decision, table and workbook checks; root agent handles compilation, visual QA and prose semantics."}
    checks = {"immutable_sources":lambda:immutable_check(snapshot), "source_arrays":source_checks,
              "candidate_decisions_and_score_oracles":decision_checks, "four_timing_groups":timing_checks,
              "final_exports_and_metrics":export_checks, "attachment_workbooks":workbook_checks,
              "paper_tables":table_checks,"asset_provenance":asset_provenance_checks}
    for name, check in checks.items():
        try:
            report["checks"][name] = check()
            print("PASS",name,flush=True)
        except Exception as exc:
            report["checks"][name] = {"passed":False,"error":str(exc),"traceback":traceback.format_exc()}
            report["failures"].append({"check":name,"error":str(exc)})
            print("FAIL",name,str(exc),flush=True)
    report["input_hashes"] = {str(p.relative_to(ROOT)):sha(p) for p in
        [OUT/"results_export.json",OUT/"summary.json",OUT/"daily_summary.csv",OUT/"asset_manifest.json",ROOT/"main.tex",
         *sorted((ROOT/"sections").glob("*.tex")), *sorted((ROOT/"tables").glob("*.tex"))]
        if p.exists()}
    report["passed"] = not report["failures"]
    report["status"] = "PASS" if report["passed"] else "FAIL"
    REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(report["status"],str(REPORT),flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
