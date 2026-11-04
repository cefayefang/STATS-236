"""Check manuscript provenance, critical numerical claims and final LaTeX output.

Usage: .venv/bin/python code/scripts/verify_report.py
Compile into tmp/report-build/ before this check. This validates the draft's
consistency with saved evidence; it does not validate observational assumptions.
"""
from pathlib import Path
import hashlib
import json
import re
import subprocess

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'report'
checks=[]
def require(condition,description):
    if not condition:raise AssertionError(description)
    checks.append(description)
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def table(name):return pd.read_csv(ROOT/'tables'/name)
source='\n'.join(p.read_text() for p in sorted((REPORT/'sec').glob('*.tex')))
main=source
entry=(REPORT/'main.tex').read_text()
asset=json.loads((REPORT/'generated/asset_manifest.json').read_text())
for filename,expected in {**asset['input_sha256'],**asset['source_sha256']}.items():
    require(digest(ROOT/filename)==expected,f'Input/source SHA256: {filename}')
for filename,expected in asset['asset_sha256'].items():
    require(digest(REPORT/filename)==expected,f'Figure/table SHA256: {filename}')
template=json.loads((REPORT/'template_manifest.json').read_text())
for name,expected in template['template_file_sha256'].items():
    require(digest(REPORT/name)==expected,f'Supplied template retained: {name}')
original=ROOT/template['source_archive']
if original.exists():
    require(digest(original)==template['archive_sha256'],'Original supplied template archive identified by SHA256')
require(not re.search('[\u4e00-\u9fff]',source),'Manuscript prose is English')
require(not re.search(r'\b(TODO|TBD|PLACEHOLDER|Pascal|BenchCal|SHBT261)\b',source),'No placeholders or previous-project text')
require(not re.search(r'\\appendix\b|\bappendix\b',entry+source,re.I),'Every report section is in the main body; no appendix')
core=['Spectral','SCORE','Louvain']
primary=table('tab06_main_test.csv').set_index('Method').loc[core]
comparison=table('tab11_method_comparison.csv').set_index('Method').loc[core]
require(primary.K.eq(11).all(),'Reported common real-data K is 11')
require(primary.B.eq(1999).all() and primary.exceedances.eq(0).all(),'Primary permutation counts and zero exceedances agree')
require(np.allclose(primary['p (perm)'],.0005) and np.allclose(primary.p_Holm_primary_3_methods,.0015),'Primary raw and Holm p-values agree')
for method in core:
    require(f'{primary.loc[method,"T"]:.4f}' in main,f'Reported equal contrast: {method}')
    require(f'{primary.loc[method,"ratio S_same/S_diff"]:.3f}' in main,f'Reported Jaccard ratio: {method}')
    require(f'{comparison.loc[method,"pair_weighted_T"]:.4f}' in main,f'Reported pair contrast: {method}')
require(primary['T'].gt(0).all() and comparison.loc['Spectral','pair_weighted_T']<0 and comparison.loc[['SCORE','Louvain'],'pair_weighted_T'].gt(0).all(),'Weighting-reversal claim agrees with saved estimates')
blocks=np.load(REPORT/'generated/block_summaries.npz')
for method in core:
    music=blocks[method+'_music'];social=blocks[method+'_social'];sizes=blocks[method+'_sizes']
    require(music.shape==social.shape==(11,11) and sizes.sum()==7431,f'Block figure population and dimensions: {method}')
    require(np.allclose(music,music.T) and np.allclose(social,social.T) and np.all((music>=0)&(music<=1)) and np.all((social>=0)&(social<=1)),f'Block figure symmetric similarities and densities are valid: {method}')
    a=np.diag(music);upper=np.triu_indices(11,1);between=music[upper]
    equal=a.mean()-between.mean()
    same=sizes*(sizes-1);cross=np.outer(sizes,sizes)[upper]
    pair=(same*a).sum()/same.sum()-(cross*between).sum()/cross.sum()
    require(np.isclose(equal,primary.loc[method,'T'],atol=1e-11) and np.isclose(pair,comparison.loc[method,'pair_weighted_T'],atol=1e-11),f'Heatmap means reproduce both reported contrasts: {method}')
binning=table('tab23_primary_binning_sensitivity.csv')
exact=binning[binning.scheme.eq('degree_exact_activity_q5')&binning.statistic.eq('equal')]
require(np.allclose(exact.p_Holm_3_methods_within_scheme_statistic,.006),'Exact-degree sensitivity Holm p agrees')
cal=table('tab26_primary_calibration_summary.csv')
stress=cal[cal.scenario.eq('smooth_degree_only_stress')&cal.method.eq('Spectral')&cal.statistic.eq('equal')].set_index('scheme')
require(np.isclose(stress.loc['degree_activity_q5','rejection_fraction'],.33) and np.isclose(stress.loc['degree_exact_activity_q5','rejection_fraction'],.065),'Reported continuous-degree stress rates agree')
interaction=table('tab29_degree_correction_interaction.csv')
for row in interaction[interaction.metric.eq('ARI')].itertuples():
    for value in [row.mean,row.ci_low,row.ci_high]:
        require(f'{value:.3f}' in main,f'Reported ARI interaction/interval value {value:.3f}')
for row in table('tab34_core_perturbation_summary.csv').itertuples():
    require(f'{row.NMI_mean:.3f}' in main,f'Reported shared-perturbation NMI: {row.method}')
require('30\\%' in source and '86.7\\%' in source,'Finite-grid K-match qualifications are present')
log=(ROOT/'tmp/report-build/main.log').read_text(errors='replace')
require(not re.search(r'Overfull|undefined|Missing character|LaTeX Error|Fatal error',log),'No overflow, missing glyph, unresolved reference or compile errors')
pdf=REPORT/'LastFM_STAT236_Report.pdf'
text=subprocess.check_output(['pdftotext','-layout',str(pdf),'-'],text=True)
pages=[page for page in text.split('\f') if page.strip()]
require(not re.search(r'\?\?|\[\?\]',text),'PDF contains no unresolved references or citations')
require(len(pages)<=8,'Entire report, including all analyses and references, fits within eight pages')
require('References' in text,'Bibliography is present in the report')
require(not re.search(r'\bAppendix\b',text),'Compiled report contains no appendix')
figure_paths=re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{(figures/[^}]+\.pdf)\}',entry+source)
expected={p for p in asset['asset_sha256'] if p.startswith('figures/') and p.endswith('.pdf')}
require(len(figure_paths)==6 and set(figure_paths)==expected,'All six selected scientific figures are included in the body')
require(len(list((REPORT/'figures').glob('*.pdf')))==6,'Six scientific figures exported without obsolete assets')
require(len(list((REPORT/'generated').glob('*results.tex')))==2,'Two numerical tables complement the expanded figures')
require('\\usepackage{annotate_equations}' in (REPORT/'preamble.tex').read_text(),'Original annotated-equation package is enabled')
require(source.count('\\eqnmarkbox')>=7 and source.count('\\annotate[')>=7,'Key equations retain coloured boxes and arrow annotations')
sources=[REPORT/'main.tex',REPORT/'preamble.tex',REPORT/'references.bib',REPORT/'cvpr.sty',REPORT/'ieeenat_fullname.bst',REPORT/'annotate_equations.sty',REPORT/'template_manifest.json']+sorted((REPORT/'sec').glob('*.tex'))
sources += [ROOT/'code/scripts'/name for name in ['build_report_assets.py','build_report.py','verify_report.py']]
result=dict(success=True,check_count=len(checks),checks=checks,total_pages=len(pages),
            main_and_reference_pages=len(pages),appendix_pages=0,figure_count=6,table_count=2,
            style='Supplied CVPR 10pt two-column template',
            source_sha256={str(p.relative_to(ROOT)):digest(p) for p in sources},
            pdf_sha256=digest(pdf),
            interpretation='Consistency and provenance checks; observational exchangeability is a substantive assumption.')
(REPORT/'generated/report_verification.json').write_text(json.dumps(result,indent=2))
print(f'PASS: {len(checks)} report checks; {len(pages)} pages; no appendix; 6 figures; original equation annotations')
