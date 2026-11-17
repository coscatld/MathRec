"""Check source dependencies, reference keys and requested chapter structure.

This is not LaTeX compilation or manuscript peer review.
"""
from pathlib import Path
import hashlib
import json
import re

ROOT=Path(__file__).resolve().parents[1]


def main():
    seen=set();missing=[];texts=[];figures=[]
    def visit(path):
        if path in seen:return
        seen.add(path);text=path.read_text(encoding='utf-8-sig');texts.append(text)
        for target in re.findall(r'\\input\{([^}]+)\}',text):
            p=ROOT/(target if Path(target).suffix else target+'.tex')
            if p.exists():visit(p)
            else:missing.append(str(p))
        for target in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',text):
            if not (ROOT/target).exists():missing.append(target)
            figures.append(target)
    visit(ROOT/'main.tex');body='\n'.join(texts)
    labels=re.findall(r'\\label\{([^}]+)\}',body)
    refs=re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',body)
    bib=set(re.findall(r'\\bibitem(?:\[[^]]*\])?\{([^}]+)\}',body))
    cites={k for group in re.findall(r'\\cite(?:\[[^]]*\])?\{([^}]+)\}',body) for k in group.split(',')}
    chapters=[]
    for index,num in enumerate('一二三四',start=1):
        p=ROOT/'sections'/f'question{index}.tex';s=p.read_text(encoding='utf-8-sig')
        section=re.findall(r'\\section\{([^}]+)\}',s)
        subs=re.findall(r'\\subsection\{([^}]+)\}',s)
        assert section==[f'问题{num}模型建立与求解'],section
        assert subs==[f'问题{num}分析',f'问题{num}模型建立',f'问题{num}求解'],subs
        chapters.append({'section_number':index+3,'section':section[0],
             'subsections':{f'{index+3}.{j+1}':v for j,v in enumerate(subs)},
             'subsubsection_count':len(re.findall(r'\\subsubsection\{',s))})
    baseline=json.loads((ROOT/'data/layout_revision_baseline.json').read_text(encoding='utf-8'))
    changed=[]
    for original,digest in baseline.items():
        # Relative fallback permits checking after unpacking onto another machine.
        p=Path(original)
        if not p.exists():
            parts=p.as_posix().split('/');offset=parts.index('CUMCMThesis-master') if 'CUMCMThesis-master' in parts else parts.index('附件')
            p=ROOT.parent/Path(*parts[offset:])
        if hashlib.sha256(p.read_bytes()).hexdigest()!=digest:changed.append(str(p))
    report={'tex_input_count':len(seen),'figure_count':len(figures),'unique_figure_count':len(set(figures)),
        'figures':figures,'chapters':chapters,'missing_files':missing,
        'unresolved_labels':sorted(set(refs)-set(labels)),
        'duplicate_labels':sorted({x for x in labels if labels.count(x)>1}),
        'missing_bibkeys':sorted(cites-bib),'bibliography_count':len(bib),
        'changed_numeric_or_workbook_files':changed,'latex_compiled':False,'paper_review_performed':False}
    for key in ['missing_files','unresolved_labels','duplicate_labels','missing_bibkeys','changed_numeric_or_workbook_files']:
        assert not report[key],f'{key}: {report[key]}'
    (ROOT/'data/source_bundle_checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='figures'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
