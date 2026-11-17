/**
 * 附件 5 结果填写程序。
 * 输入：../data/results_export.json（求解器导出的 kWh / 元数值）。
 * 输出：../../附件/附件5/result*.xlsx；从 originals/附件5 的模板重新填写。
 * 依赖：工作环境提供的 @oai/artifact-tool，使用 bundled Node.js 执行。
 * 用法：node export_workbooks.mjs [--inspect] [--only q1|q2|q3|q4_2|q4_3]
 * --inspect 仅读取、检查和渲染原始模板，不填写结果。
 * 用户要求保留原模板全部时间标签；数组按源数据的采样列顺序一一填写，不移位。
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { FileBlob, SpreadsheetFile } from '@oai/artifact-tool';

const codeDir = path.dirname(fileURLToPath(import.meta.url));
const projectDir = path.resolve(codeDir, '..');
const templateDir = path.join(projectDir, 'originals', '附件5');
const outputDir = path.resolve(projectDir, '..', '附件', '附件5');
const previewDir = path.join(projectDir, 'outputs', 'workbook_previews');
const inputPath = path.join(projectDir, 'data', 'results_export.json');
const mappingPath = path.join(projectDir, 'data', 'time_mapping.json');
const timeMapping = JSON.parse(await fs.readFile(mappingPath, 'utf8'));
const args = process.argv.slice(2);
const inspectOnly = args.includes('--inspect');
const onlyIndex = args.indexOf('--only');
const only = onlyIndex >= 0 ? args[onlyIndex + 1] : null;
const targets = [
  { key: 'q1', filename: 'result1.xlsx', sheets: ['计划购电量', '充放电量'] },
  { key: 'q2', filename: 'result2.xlsx', sheets: ['计划购电量', '充放电量', '紧急购电量'] },
  { key: 'q3', filename: 'result3.xlsx', sheets: ['计划购电量', '调整购电量', '充放电量', '紧急购电量'] },
  { key: 'q4_2', filename: 'result4-2.xlsx', sheets: ['计划购电量', '充放电量', '紧急购电量'] },
  { key: 'q4_3', filename: 'result4-3.xlsx', sheets: ['计划购电量', '调整购电量', '充放电量', '紧急购电量'] },
].filter(t => !only || t.key === only);
if (!targets.length) throw new Error(`Unknown --only value: ${only}`);

const T = timeMapping.slots_per_day;
const EPS = timeMapping.emergency_eps;
const numberFormat = '#,##0.000;[Red]-#,##0.000;0.000';
const costFormat = '#,##0.00;[Red]-#,##0.00;0.00';
const dateFormat = 'yyyy-mm-dd';
function sum(xs) { return xs.reduce((total, x) => total + x, 0); }
function interval(a, b) {
  assert(Number.isInteger(a) && Number.isInteger(b) && a >= 0 && a < b && b <= T,
    `Invalid physical interval ${a}:${b}`);
  return `${timeMapping.physical_boundaries[a]}-${timeMapping.physical_boundaries[b]}`;
}
function dateValue(value) { return new Date(`${String(value).slice(0, 10)}T00:00:00.000Z`); }
function assert(condition, message) { if (!condition) throw new Error(message); }
assert(T === 144 && timeMapping.slots.length === T, 'Expected frozen 144-slot time mapping');
assert(timeMapping.physical_boundaries.length === T + 1, 'Expected 145 physical state boundaries');
timeMapping.slots.forEach((slot, t) => {
  assert(slot.t === t && slot.excel_column_number === t + 2 && slot.q1_excel_row_number === t + 2,
    `Frozen source/template position is inconsistent at ${t}`);
});
function validateTemplateLabels(workbook, target) {
  for (const sheetName of target.sheets.filter(s => s === '计划购电量' || s === '调整购电量')) {
    const sheet = workbook.worksheets.getItem(sheetName);
    const labels = target.key === 'q1' ? sheet.getRange('A2:A145').values.map(r => r[0])
      : sheet.getRange('B1:EO1').values[0];
    labels.forEach((label, t) => assert(label === timeMapping.slots[t].template_labels[target.key][sheetName],
      `${target.filename}/${sheetName}: original label differs from frozen mapping at ${t}`));
  }
}
function vector(v, n, name, nonnegative = true) {
  assert(Array.isArray(v) && v.length === n, `${name} must contain ${n} values`);
  v.forEach((x, i) => {
    assert(typeof x === 'number' && Number.isFinite(x), `${name}[${i}] is not finite`);
    if (nonnegative) assert(x >= -EPS, `${name}[${i}] is negative`);
  });
}
function validateCase(data, key) {
  if (key === 'q1') {
    ['plan', 'charge', 'discharge'].forEach(k => vector(data[k], T, `${key}.${k}`));
    vector(data.energy, T + 1, `${key}.energy`);
    assert(Number.isFinite(data.cost), 'q1.cost must be finite');
    return;
  }
  assert(data.dates?.length === 334, `${key} must contain 334 dates`);
  data.dates.forEach((d, i) => {
    const expected = new Date(Date.UTC(2025, 1, 1 + i)).toISOString().slice(0, 10);
    assert(String(d).slice(0, 10) === expected, `${key}.dates[${i}]: expected ${expected}`);
  });
  ['plan', 'adjusted', 'charge', 'discharge', 'energy', 'emergency'].forEach(k => {
    assert(data[k]?.length === 334, `${key}.${k} must contain 334 days`);
    data[k].forEach((v, i) => vector(v, k === 'energy' ? T + 1 : T, `${key}.${k}[${i}]`));
  });
  ['plan_cost', 'total_cost', 'adjustment_cost', 'emergency_cost'].forEach(k => vector(data[k], 334, `${key}.${k}`, false));
  if (key === 'q3' || key === 'q4_3') {
    data.dates.forEach((_, i) => {
      for (let t = 0; t < 36; t++) {
        assert(Math.abs(data.adjusted[i][t] - data.plan[i][t]) < 1e-5,
          `${key}: adjusted plan cannot change before 06:00 (day ${i}, slot ${t})`);
      }
    });
  }
}

// 保留模板工作表与列顺序，仅为填写范围设置可读的尺寸、单位、数值格式和冻结窗格。
function style(sheet, rowCount, columnCount, widths) {
  const used = sheet.getRangeByIndexes(0, 0, rowCount, columnCount);
  used.format.font = { name: '宋体', size: 10, color: '#111111' };
  used.format.rowHeight = 20;
  used.format.verticalAlignment = 'center';
  used.format.wrapText = false;
  used.format.borders = { preset: 'none' };
  sheet.getRangeByIndexes(0, 0, 1, columnCount).format = {
    font: { name: '宋体', size: 10, bold: true, color: '#111111' },
    fill: '#EDEFF2', horizontalAlignment: 'center', verticalAlignment: 'center',
    rowHeight: 25,
    borders: { bottom: { style: 'thin', color: '#B9C0CA' } },
  };
  widths.forEach(([start, count, width]) => {
    sheet.getRangeByIndexes(0, start, rowCount, count).format.columnWidth = width;
  });
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(1);
}

function fillQ1(workbook, data) {
  const plan = workbook.worksheets.getItem('计划购电量');
  plan.getRange('B2:B145').values = data.plan.map(x => [x]);
  plan.getRange('A147:B148').values = [['全天购电量（kWh）', null], ['全天购电费（元）', data.cost]];
  plan.getRange('B147').formulas = [['=SUM(B2:B145)']];
  style(plan, 148, 2, [[0, 1, 25], [1, 1, 20]]);
  plan.getRange('B2:B147').setNumberFormat(numberFormat);
  plan.getRange('B148').setNumberFormat(costFormat);
  plan.getRange('A147:B148').format.font.bold = true;

  const battery = workbook.worksheets.getItem('充放电量');
  const rows = [];
  for (const block of timeMapping.four_hour_blocks) rows.push([
    sum(data.charge.slice(block.begin, block.end)),
    sum(data.discharge.slice(block.begin, block.end)),
  ]);
  battery.getRange('B2:C7').values = rows;
  battery.getRange('E2:E3').values = [[data.energy[0]], [data.energy[T]]];
  style(battery, 7, 5, [[0, 1, 21], [1, 2, 19], [3, 1, 12], [4, 1, 19]]);
  battery.getRange('B2:C7').setNumberFormat(numberFormat);
  battery.getRange('E2:E7').setNumberFormat(numberFormat);
}

function fillSchedule(sheet, data, adjusted = false) {
  const quantities = adjusted ? data.adjusted : data.plan;
  const costs = adjusted ? data.total_cost : data.plan_cost;
  const rows = data.dates.map((date, i) => [dateValue(date), ...quantities[i], null, costs[i]]);
  sheet.getRange('A2:EQ335').values = rows;
  sheet.getRange('EP2').formulas = [['=SUM(B2:EO2)']];
  sheet.getRange('EP2:EP335').fillDown();
  style(sheet, 335, 147, [[0, 1, 23], [1, 144, 17], [145, 2, 24]]);
  sheet.getRange('A2:A335').setNumberFormat(dateFormat);
  sheet.getRange('B2:EP335').setNumberFormat(numberFormat);
  sheet.getRange('EQ2:EQ335').setNumberFormat(costFormat);
  sheet.getRange('EP1:EQ335').format.font.bold = true;
}

function fillBattery(sheet, data) {
  const rows = sheet.getRange('A1:F1').values;
  const blockLabels = sheet.getRange('B2:B7').values.map(r => r[0]);
  const stateLabels = sheet.getRange('E2:E3').values.map(r => r[0]);
  data.dates.forEach((date, i) => {
    for (const { block, begin, end } of timeMapping.four_hour_blocks) rows.push([
      dateValue(date), blockLabels[block],
      sum(data.charge[i].slice(begin, end)),
      sum(data.discharge[i].slice(begin, end)),
      block === 0 ? stateLabels[0] : block === 1 ? stateLabels[1] : null,
      block === 0 ? data.energy[i][0] : block === 1 ? data.energy[i][T] : null,
    ]);
  });
  sheet.getUsedRange().clear({ applyTo: 'contents' });
  sheet.getRangeByIndexes(0, 0, rows.length, 6).values = rows;
  style(sheet, rows.length, 6, [[0, 1, 17], [1, 1, 20], [2, 2, 20], [4, 1, 12], [5, 1, 20]]);
  sheet.getRange(`A2:A${rows.length}`).setNumberFormat(dateFormat);
  sheet.getRange(`C2:D${rows.length}`).setNumberFormat(numberFormat);
  sheet.getRange(`E2:E${rows.length}`).setNumberFormat('h:mm');
  sheet.getRange(`F2:F${rows.length}`).setNumberFormat(numberFormat);
}

function emergencyRows(data, header) {
  const rows = header;
  data.dates.forEach((date, i) => {
    let segments = 0;
    for (let t = 0; t < T;) {
      if (data.emergency[i][t] <= EPS) { t++; continue; }
      const begin = t;
      let energy = 0;
      while (t < T && data.emergency[i][t] > EPS) energy += data.emergency[i][t++];
      rows.push([dateValue(date), interval(begin, t), energy]);
      segments++;
    }
    if (!segments) rows.push([dateValue(date), '无', 0]);
  });
  assert(Math.abs(sum(rows.slice(1).map(r => r[2])) - sum(data.emergency.map(sum))) < 1e-4,
    'Emergency interval aggregation does not reconcile');
  return rows;
}

function fillEmergency(sheet, data) {
  const rows = emergencyRows(data, sheet.getRange('A1:C1').values);
  sheet.getUsedRange().clear({ applyTo: 'contents' });
  sheet.getRangeByIndexes(0, 0, rows.length, 3).values = rows;
  style(sheet, rows.length, 3, [[0, 1, 17], [1, 1, 23], [2, 1, 22]]);
  sheet.getRange(`A2:A${rows.length}`).setNumberFormat(dateFormat);
  sheet.getRange(`C2:C${rows.length}`).setNumberFormat(numberFormat);
  return rows.length - 1;
}

async function renderSheets(workbook, target, phase) {
  const imagePaths = [];
  const dir = path.join(previewDir, phase);
  await fs.mkdir(dir, { recursive: true });
  for (const sheetName of target.sheets) {
    const range = sheetName === '充放电量' ? 'A1:F8' : sheetName === '紧急购电量' ? 'A1:C8' : target.key === 'q1' ? 'A1:B8' : 'A1:F8';
    const preview = await workbook.render({ sheetName, range, scale: 1.5, format: 'png' });
    const previewPath = path.join(dir, `${target.key}_${sheetName}.png`);
    await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));
    imagePaths.push(previewPath);
  }
  return imagePaths;
}

await fs.mkdir(previewDir, { recursive: true });
const report = [];
const input = inspectOnly ? null : JSON.parse(await fs.readFile(inputPath, 'utf8'));
for (const target of targets) {
  console.log(`${inspectOnly ? 'Inspecting' : 'Filling'} ${target.filename}`);
  const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(path.join(templateDir, target.filename)));
  validateTemplateLabels(workbook, target);
  if (inspectOnly) {
    console.log((await workbook.inspect({ kind: 'sheet', include: 'id,name', maxChars: 1500 })).ndjson);
    const previews = await renderSheets(workbook, target, 'source');
    report.push({ ...target, previews });
    continue;
  }
  const data = target.key === 'q1' ? input.q1 : input.cases?.[target.key] ?? input[target.key];
  assert(data, `Missing ${target.key} in ${inputPath}`);
  validateCase(data, target.key);
  let emergencyCount = 0;
  if (target.key === 'q1') fillQ1(workbook, data);
  else {
    fillSchedule(workbook.worksheets.getItem('计划购电量'), data);
    if (target.sheets.includes('调整购电量')) fillSchedule(workbook.worksheets.getItem('调整购电量'), data, true);
    fillBattery(workbook.worksheets.getItem('充放电量'), data);
    emergencyCount = fillEmergency(workbook.worksheets.getItem('紧急购电量'), data);
  }
  workbook.recalculate();
  const inspection = await workbook.inspect({ kind: 'table', range: "'计划购电量'!A1:C4", include: 'values,formulas', tableMaxRows: 4, tableMaxCols: 3, maxChars: 1500 });
  console.log(inspection.ndjson);
  const errors = await workbook.inspect({ kind: 'match', searchTerm: '#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!', options: { useRegex: true, maxResults: 20 }, maxChars: 2000 });
  console.log(errors.ndjson);
  const previews = await renderSheets(workbook, target, 'filled');
  const output = await SpreadsheetFile.exportXlsx(workbook);
  const outputPath = path.join(outputDir, target.filename);
  await output.save(outputPath);
  // The library may write a large inspection sidecar next to the workbook.
  // Keep only the requested xlsx files in attachment 5; compact reports are below.
  await fs.rm(`${outputPath}.inspect.ndjson`, { force: true });
  report.push({ key: target.key, outputPath, inputPath, mappingPath, sheets: target.sheets, days: target.key === 'q1' ? 1 : 334,
    emergencyRows: emergencyCount, previews, inspection: inspection.ndjson, errorScan: errors.ndjson });
  console.log(`Saved ${outputPath}`);
}
await fs.writeFile(path.join(previewDir, inspectOnly ? 'source_inspection.json' : `export_report${only ? `_${only}` : ''}.json`), JSON.stringify(report, null, 2));
