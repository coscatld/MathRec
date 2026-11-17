"""统一的十分钟物理索引与附件 5 显示映射。

物理假设：输入的 00:10 采样值是 [00:00, 00:10) 内的平均功率，
不是赛题已明确规定的事实。数组 t=0 按源样本列序填入原模板第一个
购电时段。保留模板显示标签，绝不据此平移计算数组。

命令：python code/time_index.py --freeze
仅从 originals/附件5 读取原始标签，生成 data/time_mapping.json/csv。
openpyxl 只用于读取原模板；本模块不创建或保存 Excel 文件。
"""
from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

PROJECT = Path(__file__).resolve().parents[1]
MAPPING_PATH = PROJECT / 'data' / 'time_mapping.json'
SLOTS_PER_DAY = 144
MINUTES_PER_SLOT = 10
DT_HOURS = 1 / 6
EMERGENCY_EPS = 1e-7
FILES = {'q1': 'result1.xlsx', 'q2': 'result2.xlsx', 'q3': 'result3.xlsx',
         'q4_2': 'result4-2.xlsx', 'q4_3': 'result4-3.xlsx'}


def clock(boundary: int) -> str:
    """物理边界编号 0..144 -> 00:00..24:00，午夜不绕回 00:00。"""
    if int(boundary) != boundary or not 0 <= boundary <= SLOTS_PER_DAY:
        raise ValueError('Physical boundary must be an integer in 0..144')
    minutes = int(boundary) * MINUTES_PER_SLOT
    return f'{minutes // 60:02d}:{minutes % 60:02d}'


def _boundary(value: str | int) -> int:
    if isinstance(value, str):
        h, m = map(int, value.split(':'))
        if not (0 <= h <= 24 and 0 <= m < 60 and m % 10 == 0):
            raise ValueError(f'Not a ten-minute physical boundary: {value}')
        value = (h * 60 + m) // MINUTES_PER_SLOT
    clock(value)
    return int(value)


@lru_cache(maxsize=1)
def load_time_mapping() -> dict:
    mapping = json.loads(MAPPING_PATH.read_text(encoding='utf-8'))
    assert len(mapping['slots']) == SLOTS_PER_DAY
    assert mapping['physical_boundaries'] == [clock(t) for t in range(145)]
    for t, row in enumerate(mapping['slots']):
        assert row['t'] == t and row['excel_column_number'] == t + 2
        assert row['source_endpoint'] == clock(t + 1)
        assert row['physical_start'] == clock(t) and row['physical_end'] == clock(t + 1)
    return mapping


def physical_interval(begin: int, end: int, sep: str = '-') -> str:
    """半开物理区间 [begin,end) 的标签；可用 sep='--' 写入 LaTeX。"""
    begin, end = _boundary(begin), _boundary(end)
    if not begin < end:
        raise ValueError('Expected begin < end for a same-day physical interval')
    boundaries = load_time_mapping()['physical_boundaries']
    return f'{boundaries[begin]}{sep}{boundaries[end]}'


def physical_slice(start: str | int, end: str | int) -> slice:
    """指定物理时段 -> 源数组切片，例如 ('10:00','10:10') -> 60:61。"""
    begin, stop = _boundary(start), _boundary(end)
    physical_interval(begin, stop)
    return slice(begin, stop)


def slot_for_endpoint(endpoint: str) -> int:
    """源采样右端点 -> 0-based 数组索引，例如 '00:10' -> 0。"""
    boundary = _boundary(endpoint)
    if boundary == 0:
        raise ValueError('Use 24:00 for the final source endpoint, not 00:00')
    return boundary - 1


def four_hour_blocks() -> list[dict]:
    """六组 begin/end 索引、物理标签，以及各原模板四小时标签。"""
    return load_time_mapping()['four_hour_blocks']


def aggregate_four_hours(values: Sequence[float]) -> list[float]:
    if len(values) != SLOTS_PER_DAY:
        raise ValueError('Expected 144 ten-minute interval energy values')
    return [math.fsum(values[b['begin']:b['end']]) for b in four_hour_blocks()]


def emergency_segments(values: Sequence[float], eps: float = EMERGENCY_EPS,
                       sep: str = '-') -> list[dict]:
    """同一自然日最大连续正紧急电量段；返回 begin/end/label/kwh。"""
    if len(values) != SLOTS_PER_DAY or not all(math.isfinite(x) for x in values):
        raise ValueError('Expected 144 finite emergency energy values')
    if any(x < -eps for x in values):
        raise ValueError('Emergency energy cannot be negative')
    segments, t = [], 0
    while t < SLOTS_PER_DAY:
        if values[t] <= eps:
            t += 1
            continue
        begin = t
        while t < SLOTS_PER_DAY and values[t] > eps:
            t += 1
        segments.append({'begin': begin, 'end': t,
                         'label': physical_interval(begin, t, sep),
                         'kwh': math.fsum(values[begin:t])})
    return segments


def intervals(values: Sequence[float], sep: str = '--') -> list[tuple[str, float]]:
    """论文表格兼容接口，与工作簿使用相同紧急区间判据。"""
    return [(x['label'], x['kwh']) for x in emergency_segments(values, sep=sep)]


def freeze_mapping() -> None:
    import openpyxl  # read only
    from openpyxl.utils import get_column_letter

    original = PROJECT / 'originals' / '附件5'
    labels, block_labels, fingerprints = {}, {}, {}
    for key, name in FILES.items():
        path = original / name
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        labels[key] = {}
        for sheet_name in ['计划购电量', '调整购电量']:
            if sheet_name not in wb.sheetnames:
                continue
            sheet = wb[sheet_name]
            labels[key][sheet_name] = ([sheet.cell(t + 2, 1).value for t in range(144)]
                                      if key == 'q1' else
                                      [sheet.cell(1, t + 2).value for t in range(144)])
            assert all(isinstance(x, str) for x in labels[key][sheet_name])
            assert labels[key][sheet_name][0] == '0:10-0:20'
        col = 1 if key == 'q1' else 2
        block_labels[key] = [wb['充放电量'].cell(i + 2, col).value for i in range(6)]
        fingerprints[key] = {'filename': name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        wb.close()

    slots = []
    for t in range(144):
        slots.append({'t': t, 'source_sample_number': t + 1,
                      'source_endpoint': clock(t + 1),
                      'source_endpoint_minutes': (t + 1) * 10,
                      'physical_start': clock(t), 'physical_end': clock(t + 1),
                      'physical_start_minutes': t * 10, 'physical_end_minutes': (t + 1) * 10,
                      'template_label': labels['q2']['计划购电量'][t],
                      'template_labels': {k: {s: v[t] for s, v in sheets.items()}
                                          for k, sheets in labels.items()},
                      'excel_column_number': t + 2, 'excel_column_letter': get_column_letter(t + 2),
                      'q1_excel_row_number': t + 2, 'q1_excel_value_cell': f'B{t + 2}',
                      'four_hour_block': t // 24})
    blocks = [{'block': b, 'begin': b * 24, 'end': (b + 1) * 24,
               'physical_label': f'{clock(b * 24)}-{clock((b + 1) * 24)}',
               'template_labels': {k: v[b] for k, v in block_labels.items()}}
              for b in range(6)]
    mapping = {'schema_version': 1, 'slots_per_day': 144, 'minutes_per_slot': 10,
               'dt_hours': DT_HOURS, 'emergency_eps': EMERGENCY_EPS,
               'source_time_interpretation': '右端点平均功率，属于建模解释，不是题目明确规定的事实。',
               'mapping_rule': '数组严格按源样本列序写入原模板，不按模板显示区间平移；t=0 对应源 00:10、物理 00:00-00:10、模板 0:10-0:20。',
               'calendar_rule': '每个日期保留 144 段；最后一段属于当天物理 23:50-24:00，不额外读取次日数据。',
               'physical_boundaries': [clock(i) for i in range(145)],
               'original_templates': fingerprints, 'four_hour_blocks': blocks, 'slots': slots}
    MAPPING_PATH.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding='utf-8')
    csv_path = MAPPING_PATH.with_suffix('.csv')
    fields = ['t', 'source_sample_number', 'source_endpoint', 'physical_start', 'physical_end',
              'template_label', 'q1_template_label', 'excel_column_number', 'excel_column_letter',
              'q1_excel_row_number', 'q1_excel_value_cell', 'four_hour_block']
    with csv_path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in slots:
            flat = {k: row[k] for k in fields if k != 'q1_template_label'}
            flat['q1_template_label'] = row['template_labels']['q1']['计划购电量']
            writer.writerow(flat)
    load_time_mapping.cache_clear()
    load_time_mapping()
    print(f'Frozen 144 source/physical/template mappings: {MAPPING_PATH}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true', help='Read original labels and freeze JSON/CSV')
    args = parser.parse_args()
    if args.freeze:
        freeze_mapping()
    else:
        print(json.dumps(load_time_mapping()['slots'][0], ensure_ascii=False, indent=2))
        print(json.dumps(load_time_mapping()['slots'][-1], ensure_ascii=False, indent=2))
