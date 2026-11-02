"""Execute every notebook cell in-process and save rich outputs/checkpoints.

Usage: .venv/bin/python code/scripts/run_notebook.py [--smoke]
No Jupyter server or background kernel is needed. The notebook is still directly
executable with Jupyter Run All. Smoke output is isolated under tmp/smoke/.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import platform
from pathlib import Path
import sys
import time

CODE_DIR = Path(__file__).resolve().parents[1]
ROOT = CODE_DIR.parent
os.chdir(ROOT)
sys.path.insert(0, str(CODE_DIR))
parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true")
args = parser.parse_args()
os.environ["STAT236_PROFILE"] = "smoke" if args.smoke else "full"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("IPYTHONDIR", str(ROOT / "tmp" / "ipython"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["IPYTHONDIR"]).mkdir(parents=True, exist_ok=True)

import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import display
from IPython.utils.capture import capture_output
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits

source = CODE_DIR / "STAT236_LastFM_Community_Music.ipynb"
notebook = nbformat.read(source, as_version=4)
notebook.metadata.setdefault("language_info", {})["version"] = platform.python_version()
work = ROOT / ("tmp/smoke/outputs" if args.smoke else "outputs")
work.mkdir(parents=True, exist_ok=True)
checkpoint = work / "execution.checkpoint.ipynb"
destination = ROOT / "tmp/smoke/executed.ipynb" if args.smoke else source
for cell in notebook.cells:
    if cell.cell_type == "code":
        cell.outputs = []
        cell.execution_count = None

shell = InteractiveShell.instance()
shell.user_ns["display"] = display


def show_figures(*_args, **_kwargs):
    for number in plt.get_fignums():
        figure = plt.figure(number)
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
        display({"image/png": base64.b64encode(buffer.getvalue()).decode("ascii")}, raw=True)
        plt.close(figure)


plt.show = show_figures
log = []
started = time.perf_counter()
count = 0
with threadpool_limits(limits=4):
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        count += 1
        start = time.perf_counter()
        print(f"[{count}] cell {index}: {cell.source.splitlines()[0][:90]}", flush=True)
        with capture_output() as captured:
            result = shell.run_cell(cell.source, store_history=True)
        cell.execution_count = count
        if captured.stdout:
            cell.outputs.append(nbformat.v4.new_output("stream", name="stdout", text=captured.stdout))
        if captured.stderr:
            cell.outputs.append(nbformat.v4.new_output("stream", name="stderr", text=captured.stderr))
        for output in captured.outputs:
            cell.outputs.append(nbformat.v4.new_output("display_data", data=output.data, metadata=output.metadata))
        failure = result.error_before_exec or result.error_in_exec
        duration = time.perf_counter() - start
        log.append(dict(cell=index, execution_count=count, seconds=duration, success=failure is None))
        (work / "execution_log.json").write_text(json.dumps(dict(profile=os.environ["STAT236_PROFILE"],
            total_seconds=time.perf_counter() - started, cells=log), indent=2))
        nbformat.write(notebook, checkpoint)
        print(f"    {'FAILED' if failure else 'done'} in {duration:.1f}s", flush=True)
        if captured.stdout:
            print(captured.stdout[-1800:], flush=True)
        if failure:
            if captured.stderr:
                print(captured.stderr, file=sys.stderr)
            raise RuntimeError(f"Notebook cell {index} failed; checkpoint: {checkpoint}") from failure
nbformat.validate(notebook)
nbformat.write(notebook, destination)
print(f"SUCCESS: {count} code cells in {time.perf_counter() - started:.1f}s. Saved {destination}", flush=True)
