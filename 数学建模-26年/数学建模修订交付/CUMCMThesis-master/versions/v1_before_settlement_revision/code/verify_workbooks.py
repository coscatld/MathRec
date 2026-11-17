"""只读核对附件 5：逐数比对求解 JSON、Excel 数值/公式缓存、日期与汇总。

本程序不调用 openpyxl.save，不参与工作簿写入。
用法：python code/verify_workbooks.py
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import openpyxl

PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT.parent / '附件' / '附件5'
SOURCE = PROJECT / 'originals' / '附件5'
DATA = json.loads((PROJECT / 'data' / 'results_export.json').read_text(encoding='utf-8'))
FILES = {'q1': 'result1.xlsx', 'q2': 'result2.xlsx', 'q3': 'result3.xlsx',
         'q4_2': 'result4-2.xlsx', 'q4_3': 'result4-3.xlsx'}
TOL = 1e-5


def close(actual, expected, label):
    a, e = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    assert a.shape == e.shape, f'{label}: shapes {a.shape} != {e.shape}'
    assert np.isfinite(a).all(), f'{label}: nonfinite output'
    error = float(np.max(np.abs(a-e))) if a.size else 0.0
    assert error < TOL, f'{label}: max error {error}'
    return error


def clock(slot):
    return f'{slot // 6:02d}:{slot % 6 * 10:02d}'


def label(begin, end):
    return f'{clock(begin)}-{clock(end)}'


report = []
for key, filename in FILES.items():
    data = DATA['q1'] if key == 'q1' else DATA.get('cases', DATA)[key]
    p = OUTPUT / filename
    wb = openpyxl.load_workbook(p, data_only=True)
    wf = openpyxl.load_workbook(p, data_only=False)
    template = openpyxl.load_workbook(SOURCE / filename, read_only=True)
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
        close([s.cell(t+2,2).value for t in range(144)], data['plan'], f'{key}.plan')
        close([s['B147'].value, s['B148'].value], [sum(data['plan']),data['cost']], f'{key}.totals')
        b = wb['充放电量']
        for block in range(6):
            close([b.cell(block+2,2).value, b.cell(block+2,3).value],
                  [sum(data['charge'][24*block:24*(block+1)]),sum(data['discharge'][24*block:24*(block+1)])], f'{key}.battery[{block}]')
        close([b['E2'].value,b['E3'].value], [data['energy'][0], data['energy'][-1]], f'{key}.energy')
        info = {'key':key, 'days':1, 'cost':data['cost']}
    else:
        dates = [dt.datetime.fromisoformat(str(d)[:10]) for d in data['dates']]
        for sheet_name, quantity_key, cost_key in [('计划购电量','plan','plan_cost'),('调整购电量','adjusted','total_cost')]:
            if sheet_name not in wb.sheetnames:
                continue
            s = wb[sheet_name]
            assert [s.cell(1,t+2).value for t in range(144)] == [template[sheet_name].cell(1,t+2).value for t in range(144)], f'{key}: original time labels'
            assert [s.cell(i+2,1).value for i in range(334)] == dates, f'{key}: date labels'
            close([[s.cell(i+2,t+2).value for t in range(144)] for i in range(334)],data[quantity_key],f'{key}.{quantity_key}')
            close([s.cell(i+2,146).value for i in range(334)],np.sum(data[quantity_key],axis=1),f'{key}.{quantity_key}.sum')
            close([s.cell(i+2,147).value for i in range(334)],data[cost_key],f'{key}.{cost_key}')
            assert all(wf[sheet_name].cell(i+2,146).value == f'=SUM(B{i+2}:EO{i+2})' for i in range(334)), f'{key}: total formulas'
        b = wb['充放电量']
        assert b.max_row == 2005, f'{key}: incomplete storage dates'
        for i, date in enumerate(dates):
            for block in range(6):
                row = 2 + 6*i + block
                assert b.cell(row,1).value == date
                assert b.cell(row,2).value == template['充放电量'].cell(block+2,2).value
                close([b.cell(row,3).value,b.cell(row,4).value],
                      [sum(data['charge'][i][24*block:24*(block+1)]),sum(data['discharge'][i][24*block:24*(block+1)])], f'{key}.battery[{i},{block}]')
            close([b.cell(2+6*i,6).value,b.cell(3+6*i,6).value],
                  [data['energy'][i][0],data['energy'][i][-1]],f'{key}.energy[{i}]')
        emergency = wb['紧急购电量']
        actual = {d:0.0 for d in dates}
        seen = set()
        for date, segment, energy in emergency.iter_rows(min_row=2,values_only=True):
            if date is None:
                continue
            assert date in actual, f'{key}: emergency date outside requested range'
            seen.add(date)
            actual[date] += energy
            if segment != '无':
                begin,end = segment.split('-')
                h,m = map(int,begin.split(':')); left = h*6 + m//10
                h,m = map(int,end.split(':')); right = h*6 + m//10
                source = data['emergency'][dates.index(date)]
                assert all(x > 1e-7 for x in source[left:right]), f'{key}: non-contiguous emergency segment'
                assert (left==0 or source[left-1]<=1e-7) and (right==144 or source[right]<=1e-7), f'{key}: not a maximal segment'
                close([energy],[sum(source[left:right])],f'{key}.emergency.{date}.{segment}')
        assert seen == set(dates), f'{key}: emergency dates missing'
        close(list(actual.values()),np.sum(data['emergency'],axis=1),f'{key}.emergency totals')
        info = {'key':key,'days':334,'storage_rows':2004,'emergency_rows':emergency.max_row-1,
                'plan_cost':sum(data['plan_cost']),'total_cost':sum(data['total_cost']),
                'emergency_kwh':sum(actual.values())}
    info.update(filename=filename,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                original_sha256=hashlib.sha256((SOURCE/filename).read_bytes()).hexdigest(),
                sheets=wb.sheetnames,status='passed')
    report.append(info)
    print(json.dumps(info,ensure_ascii=False))

report_file = PROJECT/'outputs'/'workbook_previews'/'workbook_verification.json'
report_file.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(f'All five workbooks verified: {report_file}')
