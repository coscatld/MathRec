"""Read-only manuscript checks; writes this review's manifest, not model results."""
from pathlib import Path
import collections
import hashlib
import json
import re
import shutil

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def closure(base):
    texts, missing = {}, []
    def visit(p):
        if p in texts:
            return
        if not p.exists():
            missing.append(str(p)); return
        txt = re.sub(r'(?<!\\)%[^\n]*', '', p.read_text())
        texts[p] = txt
        for name in re.findall(r'\\(?:input|include)\{([^}]+)\}', txt):
            target = base / name
            if not target.suffix:
                target = target.with_suffix('.tex')
            visit(target)
    visit(base / 'main.tex')
    return texts, missing

current, missing = closure(ROOT)
previous, old_missing = closure(OUT / 'before')
text = '\n'.join(current.values())
old_text = '\n'.join(previous.values())
labels = re.findall(r'\\label\{([^}]+)\}', text)
refs = re.findall(r'\\(?:ref|eqref|cref|Cref|pageref)\{([^}]+)\}', text)
duplicate = [k for k, n in collections.Counter(labels).items() if n > 1]
bibkeys = re.findall(r'\\bibitem(?:\[[^]]*\])?\{([^}]+)\}', text)
cites = [k.strip() for group in re.findall(r'\\cite\w*\*?(?:\[[^]]*\])*\{([^}]+)\}', text) for k in group.split(',')]
figs = re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', text)
old_eq = set(re.findall(r'\\label\{(eq:[^}]+)\}', old_text))
new_eq = set(re.findall(r'\\label\{(eq:[^}]+)\}', text))
table_checks = {}
for old in sorted((OUT / 'before' / 'tables').glob('*.tex')):
    now = ROOT / 'tables' / old.name
    normalize = lambda s: re.sub(r'(\\begin\{table\})\[[^]]*\]', r'\1', s).rstrip()
    table_checks[str(now.relative_to(ROOT))] = {
        'byte_identical': old.read_bytes() == now.read_bytes(),
        'content_identical_ignoring_float_placement': normalize(old.read_text()) == normalize(now.read_text())}
log = (ROOT / 'main.log').read_text(errors='replace')
aux = (ROOT / 'main.aux').read_text()
page_count = int(re.findall(r'Output written on .*?\((\d+) pages', log)[-1])
appendix_page = int(re.search(r'\\newlabel\{tab:files\}\{\{[^}]*\}\{(\d+)\}', aux).group(1))
warnings = {name: bool(re.search(pattern, log)) for name, pattern in {
    'overfull_box': r'Overfull \\[hv]box',
    'missing_character': r'Missing character:',
    'undefined_reference': r'(?:Reference|Citation) .*undefined|There were undefined references',
    'multiply_defined_label': r'multiply defined'}.items()}
result = {
    'scope': 'Targeted manuscript logic, formula layout and PDF review; not a full submission-compliance certificate.',
    'current_source_files': len(current), 'missing_current_inputs': missing,
    'snapshot_missing_inputs': old_missing,
    'labels': len(labels), 'references': len(refs),
    'duplicate_labels': duplicate, 'unresolved_references': sorted(set(refs)-set(labels)),
    'unresolved_citations': sorted(set(cites)-set(bibkeys)),
    'uncited_bibliography_entries': sorted(set(bibkeys)-set(cites)),
    'missing_graphics': [f for f in figs if not (ROOT/f).exists()],
    'equation_labels_before': len(old_eq), 'equation_labels_after': len(new_eq),
    'lost_equation_labels': sorted(old_eq-new_eq),
    'cases_before': old_text.count(r'\begin{cases}'), 'cases_after': text.count(r'\begin{cases}'),
    'table_checks': table_checks,
    'figures_total': text.count(r'\begin{figure}'),
    'figures_in_appendices': sum(t.count(r'\begin{figure}') for p,t in current.items() if p.name.startswith('appendix_')),
    'pdf': {'pages': page_count, 'abstract_pages': 1, 'appendix_first_page': appendix_page,
            'body_including_references_pages': appendix_page-2, 'bytes': (ROOT/'main.pdf').stat().st_size,
            'sha256': digest(ROOT/'main.pdf')},
    'compile_issues': warnings,
    'visual_review': 'All pages rendered; contact sheets and formula/transition detail pages visually inspected.',
    'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in current},
    'related_reports': ['numerical_recheck.json','protected_recheck.json','q1_q2_review.md','rolling_logic_review.md','relocation_review.md','figure_review.md'],
    'known_submission_preparation_items': ['Full runnable source text is not yet included in the PDF appendix.', 'Five bibliography entries lack in-text citations.', 'Compact specified-date tables should be checked against the literal requested table layout.'],
}
critical = [missing, duplicate, result['unresolved_references'], result['unresolved_citations'], result['missing_graphics'], result['lost_equation_labels']]
result['targeted_revision_status'] = 'PASS' if not any(critical) and not any(warnings.values()) and all(x['content_identical_ignoring_float_placement'] for x in table_checks.values()) else 'FAIL'
(OUT/'revision_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
shutil.copyfile('/tmp/cumcm-logic-build.log', OUT/'compile.log')
print(json.dumps({k:result[k] for k in ['targeted_revision_status','current_source_files','labels','cases_before','cases_after','pdf','compile_issues']},ensure_ascii=False,indent=2))
