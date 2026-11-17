"""Choose forecasts and economic control only on January; freeze for Feb-Dec.

All validation candidates share each case's frozen v1 state at 15 January.
Actual invoice is the selection score. Closing inventories are exposed.
The production prefix is the already fixed v1 January policy, never rerun
retrospectively using the selected February settings.
"""
from dataclasses import asdict,replace
import json
from pathlib import Path
import numpy as np
from dispatch_v2 import ROOT,DATA,Config,simulate,summary_v2,make_forecast_data,january_matching


def forecast_metrics(data,case):
    out={}
    def metric(actual,pred):
        err=actual-pred;return {"mae":float(np.mean(np.abs(err))),"rmse":float(np.sqrt(np.mean(err**2)))}
    for var,key in [("load","forecast_load"),("pv","forecast_pv_history"),("price","forecast_price")]:
        out[var]=metric(data[var][31:],data[key][31:])
    if case in ("q3","q4_3"):
        out["published_pv_midnight"]=metric(data["pv"][31:],data["pv_issue_forecast"][31:,0])
    return out


def main():
    raw=dict(np.load(DATA/"prepared.npz",allow_pickle=False))
    report={"calibration_period":["2025-01-15","2025-01-31"],
       "evaluation_period":["2025-02-01","2025-12-31"],
       "score":"sum of realized billed cost on January validation; no CVaR or terminal term included",
       "search":"coordinate search over forecasts; then economic controller comparison; no Feb-Dec selection",
       "january_production":"frozen v1 January operating prefix, identical across new variants within a case",
       "cases":{}}
    output=DATA/"selection_v2.json"
    for case in ("q2","q3","q4_2","q4_3"):
        forecasts={"load_model":"weekday_4","pv_model":"mean_7","price_model":"weekday_4"}
        if case=="q4_2":forecasts.update(report["cases"]["q2"]["forecasts"])
        if case=="q4_3":forecasts.update(report["cases"]["q3"]["forecasts"])
        records=[];cache={}
        def evaluate(fc,cfg,stage):
            key=json.dumps([fc,asdict(cfg)],sort_keys=True)
            if key in cache:return cache[key]
            data=make_forecast_data(raw,**fc)
            result=simulate(data,case,cfg,start_day=14,end_day=31)
            s=summary_v2(result)
            rec={"stage":stage,"forecasts":dict(fc),"config":asdict(cfg),"summary":s}
            records.append(rec);cache[key]=rec
            print(f"VALIDATE {case} {stage} {fc} {cfg.controller} cross={cfg.cross_day}: {s['total_cost']:.2f}, Eend={s['final_energy']:.2f}",flush=True)
            return rec
        cfg=Config()
        stages=[]
        if case in ("q2","q3"):
            stages.append(("load_model",["last_week","weekday_2","weekday_4"]))
        if case=="q2":stages.append(("pv_model",["mean_3","mean_7","mean_14"]))
        if case in ("q4_2","q4_3"):
            stages.append(("price_model",["last_week","weekday_2","weekday_4"]))
        for field,choices in stages:
            tests=[evaluate({**forecasts,field:value},cfg,"forecast_"+field) for value in choices]
            winner=min(tests,key=lambda v:v["summary"]["total_cost"])
            forecasts=dict(winner["forecasts"])
        alternatives=[Config(controller="greedy"),Config(controller="lp")]
        if case in ("q3","q4_3"):alternatives.append(Config(controller="lp",cross_day=True))
        control_tests=[evaluate(forecasts,c,"controller") for c in alternatives]
        best=min(control_tests,key=lambda v:v["summary"]["total_cost"])
        cfg=Config(**best["config"])
        data=make_forecast_data(raw,**forecasts)
        np.savez_compressed(DATA/f"prepared_v2_{case}.npz",**data)
        report["cases"][case]={"forecasts":forecasts,"config":asdict(cfg),
             "validation_candidates":records,"validation_winner":best["summary"],
             "january_calibration":january_matching(data,case),
             "forecast_metrics_evaluation_descriptive_only":forecast_metrics(data,case)}
        output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
        print(f"FROZEN {case}: {forecasts}; {asdict(cfg)}",flush=True)
    report["frozen"]=True
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__=="__main__":main()
