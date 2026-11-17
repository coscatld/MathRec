"""Reformat prescribed-date tables from frozen final arrays; no model runs.

Run after publish_final_strategy.py to apply the compact manuscript layout.
Original spreadsheet layout and numeric arrays remain unchanged.
"""
from pathlib import Path
import hashlib
import json
import re
from collections import Counter
import numpy as np
from time_index import physical_slice, four_hour_blocks, clock, intervals

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/framework_revision'
DATES = ['2025-03-20', '2025-06-21', '2025-09-23', '2025-12-21']
LABELS = {'q2': '问题二', 'q3': '问题三', 'q4_2': '问题四日前版', 'q4_3': '问题四滚动版'}


def f(x):
    return f'{float(x):.2f}'


def block(caption, label, headers, rows, note='', spec='lRRRR', size=r'\small'):
    result = [r'\begin{table}[!htbp]\centering' + size,
              r'\caption{' + caption + '}', r'\label{' + label + '}',
              r'\setlength{\tabcolsep}{4pt}',
              r'\begin{tabularx}{\linewidth}{' + spec + r'}\toprule',
              ' & '.join(headers) + r'\\\midrule']
    for row in rows:
        if row is None:
            result.append(r'\midrule')
        else:
            result.append(' & '.join(row) + r'\\')
    result.append(r'\bottomrule\end{tabularx}')
    if note:
        result.append(r'\par\smallskip\parbox{\linewidth}{\footnotesize ' + note + '}')
    result.append(r'\end{table}')
    return '\n'.join(result)


def main():
    records = {}
    for case, label in LABELS.items():
        source = (ROOT / f'outputs/outer_candidate_test/{case}_selected.npz'
                  if case in ('q3', 'q4_3') else ROOT / f'data/{case}_v2.npz')
        a = dict(np.load(source, allow_pickle=False))
        ids = [list(a['dates']).index(d) for d in DATES]
        arr = a['adjusted'] if case in ('q3', 'q4_3') else a['plan']
        rows = []
        for hour in [10, 12, 14, 16, 18, 20]:
            t = physical_slice(f'{hour:02}:00', f'{hour:02}:10').start
            rows.append([f'{clock(t)}--{clock(t+1)}', *[f(arr[i, t]) for i in ids]])
        rows += [None, ['全天常规量', *[f(arr[i].sum()) for i in ids]],
                 ['全天实际费用/元', *[f(a['total_cost'][i]) for i in ids]]]
        dates = [d[5:] for d in DATES]
        noun = '最终生效常规量' if case in ('q3', 'q4_3') else '零时计划量'
        purchase = block(label + '指定日期购电结果（电量：kWh）',
                         f'tab:{case}-purchase', ['物理时段', *dates], rows,
                         f'日期均为2025年；购电量为{noun}，紧急量单列。实际费用包含计划、调整及紧急三项账单。')
        rows = []
        for b in four_hour_blocks():
            t, end = b['begin'], b['end']
            rows.append([f'{clock(t)}--{clock(end)}', *[
                f(a['charge'][i, t:end].sum()) + ' / ' + f(a['discharge'][i, t:end].sum())
                for i in ids]])
        rows += [None, ['期初 / 期末库存', *[
            f(a['energy'][i, 0]) + ' / ' + f(a['energy'][i, -1]) for i in ids]]]
        battery = block(label + '指定日期储能结果（kWh）', f'tab:{case}-battery',
                        ['物理时段', *dates], rows,
                        '六个时段行按“充电量 / 放电量”列示；末行为当日0时 / 24时库存。',
                        size=r'\fontsize{10}{12}\selectfont')
        rows = []
        for date, i in zip(dates, ids):
            group = intervals(a['emergency'][i])
            if group:
                rows += [[date if j == 0 else '', span, f(value)]
                         for j, (span, value) in enumerate(group)]
            else:
                rows.append([date, '无', '0.00'])
        rows += [None, ['四日合计', '', f(a['emergency'][ids].sum())]]
        # Each day's sum is retained explicitly, including nonzero days with
        # multiple emergency intervals; displayed sums use unrounded values.
        note = '各日合计依次为' + '、'.join(f(a['emergency'][i].sum()) for i in ids) + ' kWh。'
        emergency = block(label + '指定日期紧急购电', f'tab:{case}-emergency',
                          ['日期', '物理区间', '紧急电量/kWh'], rows, note, spec='lLR')
        content = '\n\n'.join([purchase, battery, emergency]) + '\n'
        target = ROOT / f'tables/{case}_selected.tex'
        # Independent preservation check: every original two-decimal numeric
        # data cell must occur in the compact result with at least its old count.
        old = (OUT / 'before' / 'tables' / target.name).read_text(encoding='utf-8-sig')
        expected = Counter(re.findall(r'(?<![\d.])\d+\.\d{2}(?![\d.])', old))
        actual = Counter(re.findall(r'(?<![\d.])\d+\.\d{2}(?![\d.])', content))
        assert not expected - actual, (case, expected - actual)
        target.write_text(content, encoding='utf-8')
        records[case] = {'source': str(source.relative_to(ROOT)),
                         'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                         'table': str(target.relative_to(ROOT)),
                         'table_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                         'preserved_original_numeric_cells': sum(expected.values()),
                         'all_original_values_preserved': True}
    (OUT / 'compact_tables.json').write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')
    print('PASS: compact tables preserve every original numeric cell for all four cases.')


if __name__ == '__main__':
    main()
