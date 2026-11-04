"""Compile, verify and package the English STAT 236 report.

Use --refresh-assets to regenerate figures/tables from the saved full analysis.
The source ZIP is self-contained for Overleaf and omits unrelated input-template
content and intermediate compilation files.
"""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
from zipfile import ZipFile, ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'report'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--refresh-assets',action='store_true')
args=parser.parse_args()
# The reference's arrow annotations need TikZ and tikzmark. Prefer an existing
# complete MacTeX installation when a minimal TeX installation shadows it.
tex_environment=os.environ.copy()
for tex_bin in [None,Path('/Library/TeX/texbin')]:
    if tex_bin is not None:
        if not (tex_bin/'latexmk').exists():continue
        tex_environment['PATH']=str(tex_bin)+os.pathsep+os.environ.get('PATH','')
    probe=subprocess.run(['kpsewhich','tikzlibrarytikzmark.code.tex'],
        env=tex_environment,capture_output=True,text=True)
    if probe.returncode==0:break
else:
    raise RuntimeError('The reference equation style requires PGF/TikZ and tikzmark.')
if args.refresh_assets:
    subprocess.run([sys.executable,str(ROOT/'code/scripts/build_report_assets.py')],check=True)
for name in ['latexmk','pdftotext']:
    if not shutil.which(name):raise RuntimeError(f'Required report tool is unavailable: {name}')
build=ROOT/'tmp/report-build'
build.mkdir(parents=True,exist_ok=True)
with (build/'compile.txt').open('w') as log:
    result=subprocess.run(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error',
        '-outdir='+str(build),'main.tex'],cwd=REPORT,env=tex_environment,
        stdout=log,stderr=subprocess.STDOUT)
if result.returncode:
    raise RuntimeError(f'LaTeX compilation failed. Read {build/"compile.txt"}')
shutil.copy2(build/'main.pdf',REPORT/'LastFM_STAT236_Report.pdf')
subprocess.run([sys.executable,str(ROOT/'code/scripts/verify_report.py')],check=True)
archive=REPORT/'LastFM_STAT236_Report_Source.zip'
selected=[REPORT/'README.md',REPORT/'main.tex',REPORT/'preamble.tex',REPORT/'references.bib',
          REPORT/'cvpr.sty',REPORT/'ieeenat_fullname.bst',REPORT/'annotate_equations.sty',
          REPORT/'template_manifest.json']
selected+=sorted((REPORT/'sec').glob('*.tex'))
assets=json.loads((REPORT/'generated/asset_manifest.json').read_text())
selected+=[REPORT/name for name in assets['asset_sha256']]
selected+=sorted((REPORT/'generated').glob('*.json'))
selected+=[REPORT/'LastFM_STAT236_Report.pdf']
with ZipFile(archive,'w',ZIP_DEFLATED,compresslevel=6) as z:
    for path in selected:z.write(path,path.relative_to(REPORT))
with ZipFile(archive) as z:
    if z.testzip():raise RuntimeError('Source archive integrity check failed')
print('PDF:',REPORT/'LastFM_STAT236_Report.pdf')
print('Overleaf source:',archive)
