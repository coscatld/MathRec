"""只读核对附件 5：逐数比对求解 JSON、Excel 数值/公式缓存、日期与汇总。

本程序不调用 openpyxl.save，不参与工作簿写入。
用法：python code/verify_workbooks.py
      python code/verify_workbooks.py --inventory-only
"""
from __future__ import annotations

import datetime as dt
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import openpyxl
from time_index import (EMERGENCY_EPS, aggregate_four_hours, emergency_segments,
                        load_time_mapping)

PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT.parent / '附件' / '附件5'
SOURCE = PROJECT / 'originals' / '附件5'
DATA_PATH = PROJECT / 'data' / 'results_export.json'
FILES = {'q1': 'result1.xlsx', 'q2': 'result2.xlsx', 'q3': 'result3.xlsx',
         'q4_2': 'result4-2.xlsx', 'q4_3': 'result4-3.xlsx'}
TOL = 1e-5
MAPPING = load_time_mapping()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory():
    """记录本机指定输出文件的真实路径、哈希和有限数值数量，不据此评价模型。"""
    files = []
    for key, filename in FILES.items():
        path = OUTPUT / filename
        record = {'key': key, 'filename': filename, 'absolute_path': str(path.resolve()),
                  'exists': path.exists()}
        if not path.exists():
            files.append(record)
            continue
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        regions = []
        for name in ('计划购电量', '调整购电量'):
            if name not in wb.sheetnames:
                continue
            options = {'min_row': 2, 'max_row': 145 if key == 'q1' else 335,
                       'min_col': 2, 'max_col': 2 if key == 'q1' else 145,
                       'values_only': True}
            values = [v for row in wb[name].iter_rows(**options) for v in row]
            numeric = sum(isinstance(v, (int, float)) and not isinstance(v, bool)
                          and math.isfinite(v) for v in values)
            regions.append({'sheet': name, 'range': 'B2:B145' if key == 'q1' else 'B2:EO335',
                            'expected_numeric_count': len(values), 'numeric_count': numeric,
                            'blank_count': sum(v is None for v in values),
                            'nonzero_numeric_count': sum(isinstance(v, (int, float)) and v != 0 for v in values)})
        record.update(sha256=sha256(path), size_bytes=path.stat().st_size,
                      modified_utc=dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc).isoformat(),
                      sheets=wb.sheetnames, result_regions=regions)
        wb.close()
        files.append(record)
    return {'observed_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
            'scope': '本机指定路径的实际文件；不能据此判断其他位置或外部收到的副本。', 'files': files}


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--inventory-only', action='store_true')
ARGS = parser.parse_args()
if ARGS.inventory_only:
    result = inventory()
    report_path = PROJECT / 'outputs' / 'workbook_previews' / 'initial_workbook_inventory.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f'Inventory saved: {report_path}')
    raise SystemExit(0)
DATA = json.loads(DATA_PATH.read_text(encoding='utf-8'))


def close(actual, expected, label):
    a, e = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    assert a.shape == e.shape, f'{label}: shapes {a.shape} != {e.shape}'
    assert np.isfinite(a).all(), f'{label}: nonfinite output'
    error = float(np.max(np.abs(a-e))) if a.size else 0.0
    assert error < TOL, f'{label}: max error {error}'
    return error


report = []
for key, filename in FILES.items():
    data = DATA['q1'] if key == 'q1' else DATA.get('cases', DATA)[key]
    p = OUTPUT / filename
    wb = openpyxl.load_workbook(p, data_only=True)
    wf = openpyxl.load_workbook(p, data_only=False)
    template = openpyxl.load_workbook(SOURCE / filename, read_only=True)
    assert sha256(SOURCE / filename) == MAPPING['original_templates'][key]['sha256'], f'{key}: original template fingerprint changed'
    assert wb.sheetnames == template.sheetnames, f'{filename}: worksheet order changed'
    for sheet in wb:
        source_sheet = template[sheet.title]
        assert [c.value for c in sheet[1]] == [c.value for c in source_sheet[1]], f'{filename}/{sheet.title}: original headers changed'
        assert sheet.freeze_panes == 'B2', f'{filename}/{sheet.title}: incorrect freeze {sheet.freeze_panes}'
        errors = [(cell.coordinate, cell.value) for row in sheet for cell in row if cell.data_type == 'e']
        assert not errors, f'{filename}/{sheet.title}: Excel errors {errors[:5]}'

    if key == 'q1':
        s = wb['计划购电量']
        assert [s.cell(t+2, 1).value for t in range(144)] == [template['计划购电量'].cell(t+2,1).value for t in range(144)]
        assert [s.cell(t+2, 1).value for t in range(144)] == [r['template_labels'][key]['计划购电量'] for r in MAPPING['slots']]
        close([s.cell(t+2,2).value for t in range(144)], data['plan'], f'{key}.plan')
        close([s['B147'].value, s['B148'].value], [sum(data['plan']),data['cost']], f'{key}.totals')
        b = wb['充放电量']
        charge_blocks, discharge_blocks = aggregate_four_hours(data['charge']), aggregate_four_hours(data['discharge'])
        for block in range(6):
            close([b.cell(block+2,2).value, b.cell(block+2,3).value],
                  [charge_blocks[block], discharge_blocks[block]], f'{key}.battery[{block}]')
        close([b['E2'].value,b['E3'].value], [data['energy'][0], data['energy'][-1]], f'{key}.energy')
        info = {'key':key, 'days':1, 'cost':data['cost']}
    else:
        dates = [dt.datetime.fromisoformat(str(d)[:10]) for d in data['dates']]
        for sheet_name, quantity_key, cost_key in [('计划购电量','plan','plan_cost'),('调整购电量','adjusted','total_cost')]:
            if sheet_name not in wb.sheetnames:
                continue
            s = wb[sheet_name]
            assert [s.cell(1,t+2).value for t in range(144)] == [template[sheet_name].cell(1,t+2).value for t in range(144)], f'{key}: original time labels'
            assert [s.cell(1,t+2).value for t in range(144)] == [r['template_labels'][key][sheet_name] for r in MAPPING['slots']], f'{key}: frozen time labels'
            assert [s.cell(i+2,1).value for i in range(334)] == dates, f'{key}: date labels'
            close([[s.cell(i+2,t+2).value for t in range(144)] for i in range(334)],data[quantity_key],f'{key}.{quantity_key}')
            close([s.cell(i+2,146).value for i in range(334)],np.sum(data[quantity_key],axis=1),f'{key}.{quantity_key}.sum')
            close([s.cell(i+2,147).value for i in range(334)],data[cost_key],f'{key}.{cost_key}')
            assert all(wf[sheet_name].cell(i+2,146).value == f'=SUM(B{i+2}:EO{i+2})' for i in range(334)), f'{key}: total formulas'
        b = wb['充放电量']
        assert b.max_row == 2005, f'{key}: incomplete storage dates'
        for i, date in enumerate(dates):
            charge_blocks, discharge_blocks = aggregate_four_hours(data['charge'][i]), aggregate_four_hours(data['discharge'][i])
            for block in range(6):
                row = 2 + 6*i + block
                assert b.cell(row,1).value == date
                assert b.cell(row,2).value == template['充放电量'].cell(block+2,2).value
                close([b.cell(row,3).value,b.cell(row,4).value],
                      [charge_blocks[block], discharge_blocks[block]], f'{key}.battery[{i},{block}]')
            close([b.cell(2+6*i,6).value,b.cell(3+6*i,6).value],
                  [data['energy'][i][0],data['energy'][i][-1]],f'{key}.energy[{i}]')
        emergency = wb['紧急购电量']
        actual = {d:0.0 for d in dates}
        expected_rows = []
        for i, date in enumerate(dates):
            segments = emergency_segments(data['emergency'][i], eps=EMERGENCY_EPS)
            expected_rows.extend((date, s['label'], s['kwh']) for s in segments)
            if not segments:
                expected_rows.append((date, '无', 0.0))
        actual_rows = [r for r in emergency.iter_rows(min_row=2, max_col=3, values_only=True) if r[0] is not None]
        assert len(actual_rows) == len(expected_rows), f'{key}: emergency segment count'
        for row, expected in zip(actual_rows, expected_rows):
            date, segment, energy = row
            assert row[:2] == expected[:2], f'{key}: emergency physical label/order {row[:2]} != {expected[:2]}'
            close([energy], [expected[2]], f'{key}.emergency.{date}.{segment}')
            actual[date] += energy
        close(list(actual.values()),np.sum(data['emergency'],axis=1),f'{key}.emergency totals')
        info = {'key':key,'days':334,'storage_rows':2004,'emergency_rows':emergency.max_row-1,
                'plan_cost':sum(data['plan_cost']),'total_cost':sum(data['total_cost']),
                'emergency_kwh':sum(actual.values())}
    info.update(filename=filename,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                original_sha256=hashlib.sha256((SOURCE/filename).read_bytes()).hexdigest(),
                absolute_path=str(p.resolve()), size_bytes=p.stat().st_size,
                results_source=str(DATA_PATH.resolve()), results_source_sha256=sha256(DATA_PATH),
                time_mapping_sha256=sha256(PROJECT/'data'/'time_mapping.json'),
                plan_numeric_count=144 if key=='q1' else 48096,
                first_template_label=MAPPING['slots'][0]['template_labels'][key]['计划购电量'],
                last_template_label=MAPPING['slots'][-1]['template_labels'][key]['计划购电量'],
                sheets=wb.sheetnames,status='passed')
    report.append(info)
    print(json.dumps(info,ensure_ascii=False))

report_file = PROJECT/'outputs'/'workbook_previews'/'workbook_verification.json'
report_file.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
manifest = report_file.with_name('workbook_file_manifest.csv')
with manifest.open('w', newline='', encoding='utf-8-sig') as stream:
    fields = ['key','filename','absolute_path','size_bytes','sha256','plan_numeric_count','days',
              'first_template_label','last_template_label','results_source_sha256','time_mapping_sha256','status']
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(report)
print(f'All five workbooks verified: {report_file}')
