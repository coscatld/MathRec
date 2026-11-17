"""Bundle source, preserved versions, inputs and the five filled workbooks."""
from pathlib import Path
import csv
import hashlib
import json
import os
import zipfile

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parent
TARGET=BASE/'论文与结果_分问图表版.zip'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    # Refuse to silently package stale or blank template copies as results.
    manifest=ROOT/'outputs/workbook_previews/workbook_file_manifest.csv'
    with manifest.open(encoding='utf-8-sig',newline='') as src:
        rows=list(csv.DictReader(src))
    print('Workbook manifest columns:',list(rows[0]))
    files=[]
    for directory,subdirs,names in os.walk(ROOT,followlinks=False):
        subdirs[:]=[d for d in subdirs if d not in ['node_modules','__pycache__','.git','.python_deps']]
        for name in names:
            path=Path(directory)/name
            if path.is_symlink() or name.endswith(('.pyc','.zip','.inspect.ndjson','.log','.aux','.toc','.out','.synctex.gz')):continue
            if name in ['delivery_manifest.json','delivery_package.json']:continue
            files.append(path)
    files += [BASE/'C题.pdf',*[BASE/'附件'/f'附件{i}.xlsx' for i in range(1,5)]]
    filled=[BASE/'附件/附件5'/name for name in ['result1.xlsx','result2.xlsx','result3.xlsx','result4-2.xlsx','result4-3.xlsx']]
    for path in filled:
        row=next(r for r in rows if path.name in r.values())
        assert sha(path) in [v.lower() for v in row.values()],f'Workbook changed after validation: {path}'
    files+=filled
    records=[{'path':p.relative_to(BASE).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)]
    inventory=ROOT/'data/delivery_manifest.json'
    inventory.write_text(json.dumps({'scope':'all delivered files except this manifest and the outer ZIP',
          'latex_compiled':False,'paper_review_performed':False,'files':records},ensure_ascii=False,indent=2),encoding='utf-8')
    files.append(inventory)
    prefix='数学建模修订交付/'
    with zipfile.ZipFile(TARGET,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as out:
        for p in sorted(files):out.write(p,prefix+p.relative_to(BASE).as_posix())
        out.writestr(prefix+'先读说明.txt',
            '论文入口：CUMCMThesis-master/main.tex。\n'
            '本版按四个问题分别编排分析、模型建立与求解，新增真实数据热力矩阵、桑基图、瀑布图与阴影带。\n'
            '结构与图表索引：CUMCMThesis-master/论文结构与图表说明.md。\n'
            '修订说明：CUMCMThesis-master/修订记录.md；复现与文件说明：CUMCMThesis-master/交付说明.md。\n'
            '已填写结果在附件/附件5/；originals/附件5/为原始空模板，勿混淆。\n'
            '完整代码、执行数组、一月冻结选择与原版备份均包含。\n'
            '按要求没有编译LaTeX、没有进行论文审稿；example.pdf仅为原模板示例。\n'
            '购电页保留原模板0:10–0:20至次日0:00–0:10标签，数据不移位。\n')
    with zipfile.ZipFile(TARGET) as check:
        assert check.testzip() is None
        for path in filled:
            blob=check.read(prefix+path.relative_to(BASE).as_posix())
            assert hashlib.sha256(blob).hexdigest()==sha(path)
    report={'path':str(TARGET),'bytes':TARGET.stat().st_size,'sha256':sha(TARGET),'files':len(files)+1,
            'filled_workbooks':5,'zip_integrity_check':'passed','latex_compiled':False}
    (ROOT/'data/delivery_package.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
