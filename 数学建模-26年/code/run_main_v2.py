"""Execute frozen economic policies and publish one result source."""
import json
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import numpy as np
from dispatch_v2 import ROOT,DATA,Config,simulate,save_result,summary_v2
from optimize_dispatch import jsonable


def run(case):
    selection=json.loads((DATA/"selection_v2.json").read_text(encoding="utf-8"))
    assert selection.get("frozen"),"January selection must be frozen first"
    cfg=Config(**selection["cases"][case]["config"])
    data=dict(np.load(DATA/f"prepared_v2_{case}.npz",allow_pickle=False))
    r=simulate(data,case,cfg,progress=True)
    save_result(case,r,cfg)
    return {"case":case,**summary_v2(r)}


def export():
    selection=json.loads((DATA/"selection_v2.json").read_text(encoding="utf-8"))
    export={"version":"v2_settlement_control","q1":json.loads((DATA/"q1.json").read_text(encoding="utf-8")),
       "selection":selection,"cases":{}}
    for case in ("q2","q3","q4_2","q4_3"):
        r=dict(np.load(DATA/f"{case}_v2.npz",allow_pickle=False))
        assert len(r["dates"])==334
        export["cases"][case]=jsonable(r)
    (DATA/"results_export.json").write_text(json.dumps(export,ensure_ascii=False,separators=(",",":")),encoding="utf-8")
    print("Written authoritative v2 results_export.json",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--workers",type=int,default=2)
    p.add_argument("--only",choices=["q2","q3","q4_2","q4_3"]);p.add_argument("--export-only",action="store_true")
    args=p.parse_args()
    if not args.export_only:
        cases=[args.only] if args.only else ["q2","q3","q4_2","q4_3"]
        if args.workers==1:
            for c in cases:print(json.dumps(run(c)),flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for result in pool.map(run,cases):print(json.dumps(result),flush=True)
    if all((DATA/f"{c}_v2.npz").exists() for c in ["q2","q3","q4_2","q4_3"]):export()
