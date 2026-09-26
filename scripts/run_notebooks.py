"""Execute every notebook in order and save outputs in place.

    python scripts/run_notebooks.py

Run `python scripts/train.py` first: notebooks 06-09 compare their results
with the saved model artefacts.
"""
import subprocess
import sys
from pathlib import Path

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"

for nb in sorted(NB_DIR.glob("[0-9][0-9]_*.ipynb")):
    print(f"-> {nb.name}", flush=True)
    subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                    "--inplace", "--ExecutePreprocessor.timeout=1800", str(nb)], check=True)
print("All notebooks executed.")
